pragma Singleton
pragma ComponentBehavior: Bound

import qs.modules.common
import qs.modules.common.functions
import qs.services
import QtQuick
import Quickshell
import Quickshell.Io

Singleton {
    id: root

    property bool enabled: false
    property string caption: ""
    property string status: ""
    property string lastError: ""
    // How long to keep the last caption visible after the last update.
    // Set to 0 to never auto-clear.
    property int clearAfterMs: 45000
    property int statusClearAfterMs: 6000
    readonly property bool running: captionProc.running

    function toggleEnabled(force = null) {
        if (force === null) {
            root.enabled = !root.enabled;
        } else {
            root.enabled = force;
        }
    }

    function resetState() {
        root.caption = "";
        root.status = "";
        root.lastError = "";
        clearTimer.stop();
        statusClearTimer.stop();
    }

    function setCaption(text) {
        const cleaned = (text ?? "").toString().trim();
        if (cleaned.length === 0 || cleaned === root.caption) return;
        root.caption = cleaned;
        if (root.status.length > 0) root.status = "";
        if (root.lastError.length > 0) root.lastError = "";
        statusClearTimer.stop();
        if (root.clearAfterMs > 0) clearTimer.restart();
        else clearTimer.stop();
    }

    onEnabledChanged: {
        captionProc.running = root.enabled;
        resetState();
    }

    Timer {
        id: clearTimer
        interval: Math.max(root.clearAfterMs, 0)
        repeat: false
        onTriggered: {
            if (root.clearAfterMs <= 0) return;
            root.caption = "";
        }
    }

    Timer {
        id: statusClearTimer
        interval: root.statusClearAfterMs
        repeat: false
        onTriggered: root.status = ""
    }

    Process {
        id: captionProc
        running: false
        environment: ({
            LANG: "C",
            LC_ALL: "C"
        })
        command: [
            "python3",
            "-E",
            `${FileUtils.trimFileProtocol(Directories.scriptPath)}/liveCaption/live-caption.py`
        ]

        stdout: SplitParser {
            onRead: data => root.setCaption(data)
        }

        stderr: SplitParser {
            onRead: data => {
                const cleaned = (data ?? "").toString().trim();
                if (cleaned.length === 0) return;
                if (cleaned.startsWith("status:")) {
                    const nextStatus = cleaned.slice(7).trim();
                    if (nextStatus === root.status) return;
                    root.status = nextStatus;
                    if (root.statusClearAfterMs > 0 && !root.status.toLowerCase().startsWith("loading")) {
                        statusClearTimer.restart();
                    } else {
                        statusClearTimer.stop();
                    }
                    return;
                }
                if (cleaned === root.lastError) return;
                root.lastError = cleaned;
            }
        }

        onExited: (exitCode, exitStatus) => {
            if (!root.enabled) return;
            const errorSummary = root.lastError.length > 0 ? root.lastError : Translation.tr("Backend exited unexpectedly");
            root.enabled = false;
            if (exitCode !== 0) {
                Quickshell.execDetached([
                    "notify-send",
                    Translation.tr("Live captions stopped"),
                    `${errorSummary} (${Translation.tr("exit code")}: ${exitCode})`,
                    "-a",
                    "Shell"
                ]);
            }
        }
    }

    IpcHandler {
        target: "liveCaption"

        function toggle(): void {
            root.toggleEnabled();
        }

        function enable(): void {
            root.toggleEnabled(true);
        }

        function disable(): void {
            root.toggleEnabled(false);
        }
    }
}
