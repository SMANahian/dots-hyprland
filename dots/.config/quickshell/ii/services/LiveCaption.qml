pragma Singleton
pragma ComponentBehavior: Bound

import QtQuick
import Quickshell
import Quickshell.Io
import qs.modules.common

Singleton {
    id: root

    property bool enabled: (Config.ready ?? true) && (Config.options.liveCaption.enabled ?? false)
    property string modelDir: Config.options.liveCaption.modelDir ?? ""
    readonly property bool depsInstalled: Config.options.liveCaption.depsInstalled ?? false
    readonly property string effectiveSource: {
        const value = `${Config.options.liveCaption.source ?? "monitor"}`;
        if (value === "input") return "input";
        if (value === "both") return "both";
        return "monitor";
    }

    property string text: ""
    property string lastErrorLine: ""
    property int lastExitCode: 0
    property bool faulted: false
    property string setupStatusLine: ""
    property bool restarting: false
    // JS `trim()` doesn't remove some whitespace chars (e.g. U+0085) that Python's `strip()` does.
    // If the user pastes a "blank" modelDir containing such chars, the backend sees an empty modelDir.
    readonly property string effectiveModelDir: (root.modelDir ?? "").replace(/^[\\s\\u0085]+|[\\s\\u0085]+$/g, "")
    readonly property bool configured: root.effectiveModelDir.length > 0
    readonly property bool settingUp: depsInstallProc.running || modelDownloadProc.running
    readonly property bool shouldRun: root.enabled && root.configured && !root.faulted && !root.settingUp && !root.restarting
    readonly property bool running: captionProc.running

    function ensureSetup(): void {
        if (!root.enabled) return;
        if (!root.depsInstalled && !depsInstallProc.running) {
            depsInstallProc.statusLine = "";
            depsInstallProc.running = true;
        }
        if (!root.configured && !modelDownloadProc.running) {
            modelDownloadProc.statusLine = "";
            modelDownloadProc.running = true;
        }
    }

    function toggleEnabled(): void {
        const newEnabled = !(Config.options.liveCaption.enabled ?? false);
        if (newEnabled) {
            root.faulted = false;
            root.lastExitCode = 0;
            root.lastErrorLine = "";
        }
        Config.options.liveCaption.enabled = newEnabled;
    }

    function clear(): void {
        root.text = "";
    }

    function retry(): void {
        root.faulted = false;
        root.lastExitCode = 0;
        root.lastErrorLine = "";
        root.ensureSetup();
    }

    onEnabledChanged: {
        if (!enabled) {
            root.text = "";
            root.lastErrorLine = "";
            root.lastExitCode = 0;
            root.faulted = false;
            root.setupStatusLine = "";
            depsInstallProc.running = false;
            modelDownloadProc.running = false;
        } else {
            root.ensureSetup();
        }
    }

    onModelDirChanged: {
        if (!root.enabled) return;
        if (!root.configured) return;
        // If the user fixes the model path, allow the process to start again.
        root.faulted = false;
        root.lastExitCode = 0;
        root.lastErrorLine = "";
    }

    onEffectiveSourceChanged: {
        root.text = "";
        if (root.enabled) {
            root.restarting = true;
            restartTimer.restart();
        }
    }

    Timer {
        id: restartTimer
        interval: 200
        onTriggered: root.restarting = false;
    }

    Process {
        id: depsInstallProc
        running: false
        command: [`${Directories.scriptPath}/liveCaption/install-deps.sh`]

        property string statusLine: ""

        stdout: StdioCollector {
            onStreamFinished: {
                const version = (this.text ?? "").trim();
                if (version.length) {
                    Quickshell.execDetached([
                        "notify-send",
                        Translation.tr("Live caption deps installed"),
                        `${Translation.tr("sherpa-onnx version")}: ${version}`,
                        "-a",
                        "Shell",
                    ]);
                }
            }
        }
        stderr: SplitParser {
            onRead: line => {
                depsInstallProc.statusLine = (line ?? "").trim();
                root.setupStatusLine = depsInstallProc.statusLine;
            }
        }
        onExited: (exitCode, exitStatus) => {
            if (!root.enabled) return;
            if (exitCode === 0) {
                Config.options.liveCaption.depsInstalled = true;
                root.setupStatusLine = "";
                return;
            }
            root.lastExitCode = exitCode;
            root.lastErrorLine = depsInstallProc.statusLine;
            Quickshell.execDetached([
                "notify-send",
                Translation.tr("Live caption deps install failed"),
                `${Translation.tr("Exit code")}: ${exitCode}${depsInstallProc.statusLine.length ? `\n${depsInstallProc.statusLine}` : ""}`,
                "-a",
                "Shell",
            ]);
            root.faulted = true;
        }
    }

    Process {
        id: modelDownloadProc
        running: false
        command: [`${Directories.scriptPath}/liveCaption/download-model.sh`]

        property string statusLine: ""

        stdout: StdioCollector {
            onStreamFinished: {
                const modelDir = (this.text ?? "").trim();
                if (!modelDir.length) return;
                Config.options.liveCaption.modelDir = modelDir;
                root.setupStatusLine = "";
            }
        }
        stderr: SplitParser {
            onRead: line => {
                modelDownloadProc.statusLine = (line ?? "").trim();
                root.setupStatusLine = modelDownloadProc.statusLine;
            }
        }
        onExited: (exitCode, exitStatus) => {
            if (!root.enabled) return;
            if (exitCode === 0) return;
            root.lastExitCode = exitCode;
            root.lastErrorLine = modelDownloadProc.statusLine;
            Quickshell.execDetached([
                "notify-send",
                Translation.tr("Live caption download failed"),
                `${Translation.tr("Exit code")}: ${exitCode}${modelDownloadProc.statusLine.length ? `\n${modelDownloadProc.statusLine}` : ""}`,
                "-a",
                "Shell",
            ]);
            root.faulted = true;
        }
    }

    Process {
        id: captionProc
        running: root.shouldRun
        command: [
            "bash", "-c",
            `VENV="\${ILLOGICAL_IMPULSE_VIRTUAL_ENV:-\${XDG_STATE_HOME:-\$HOME/.local/state}/quickshell/.venv}"; VENV="\$(eval echo \$VENV)"; source "\$VENV/bin/activate" && exec python -E "${Directories.scriptPath}/liveCaption/live_caption.py" "$@"`,
            "--",
            "--model-dir",
            root.effectiveModelDir,
            "--source",
            root.effectiveSource,
            "--enable-endpoint",
            "--num-threads",
            `${Config.options.liveCaption.numThreads ?? 4}`,
            "--update-interval-ms",
            `${Config.options.liveCaption.updateIntervalMs ?? 120}`,
            "--history-chars",
            `${Config.options.liveCaption.historyChars ?? 320}`,
            "--no-history",
        ]
        stdout: SplitParser {
            onRead: line => {
                const t = (line ?? "").trim();
                root.text = t;
            }
        }
        stderr: SplitParser {
            onRead: line => {
                root.lastErrorLine = (line ?? "").trim();
            }
        }
        onExited: (exitCode, exitStatus) => {
            root.lastExitCode = exitCode;
            if (!root.enabled) return;
            if (exitCode === 0) return;
            Quickshell.execDetached([
                "notify-send",
                Translation.tr("Live caption stopped"),
                `${Translation.tr("Exit code")}: ${exitCode}${root.lastErrorLine.length ? `\n${root.lastErrorLine}` : ""}`,
                "-a",
                "Shell",
            ]);
            root.faulted = true;
        }
    }

    IpcHandler {
        target: "liveCaption"

        function toggle(): void {
            root.toggleEnabled();
        }

        function enable(): void {
            Config.options.liveCaption.enabled = true;
        }

        function disable(): void {
            Config.options.liveCaption.enabled = false;
        }

        function clear(): void {
            root.clear();
        }

        function retry(): void {
            root.retry();
        }
    }
}
