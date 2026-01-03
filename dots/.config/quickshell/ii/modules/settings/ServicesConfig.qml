import QtQuick
import QtQuick.Layouts
import Quickshell
import Quickshell.Io
import qs.services
import qs.modules.common
import qs.modules.common.widgets

ContentPage {
    id: root
    forceWidth: true

    readonly property string liveCaptionDownloadScriptPath: `${Directories.scriptPath}/liveCaption/download-model.sh`
    readonly property string liveCaptionInstallDepsScriptPath: `${Directories.scriptPath}/liveCaption/install-deps.sh`

    readonly property var availableModels: [
        // English models
        {
            name: "sherpa-onnx-streaming-zipformer-en-2023-06-26",
            displayName: "⭐ English (Zipformer) — Recommended",
            description: Translation.tr("Best balance of speed and accuracy. ~70MB. Good for most PCs."),
            language: "en",
            size: "~70MB",
            recommended: "balanced"
        },
        {
            name: "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
            displayName: "🚀 English (Small) — Fast",
            description: Translation.tr("Very small and fast. ~42MB. Best for older/low-end hardware."),
            language: "en",
            size: "~42MB",
            recommended: "low-end"
        },
        {
            name: "sherpa-onnx-lstm-en-2023-02-17",
            displayName: "💾 English (LSTM) — Lightweight",
            description: Translation.tr("Different architecture. ~366MB. Alternative if Zipformer has issues."),
            language: "en",
            size: "~366MB",
            recommended: ""
        },
        // Multilingual
        {
            name: "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
            displayName: "🌏 Chinese + English (Bilingual)",
            description: Translation.tr("Supports both Chinese and English. ~191MB."),
            language: "zh-en",
            size: "~191MB",
            recommended: ""
        },
        {
            name: "sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16",
            displayName: "🌏 Chinese + English (Small)",
            description: Translation.tr("Small bilingual model. ~48MB. Fast but less accurate."),
            language: "zh-en",
            size: "~48MB",
            recommended: ""
        },
        // Chinese
        {
            name: "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
            displayName: "🇨🇳 Chinese (Small)",
            description: Translation.tr("Small Chinese model. ~25MB. Very fast."),
            language: "zh",
            size: "~25MB",
            recommended: ""
        },
        // French
        {
            name: "sherpa-onnx-streaming-zipformer-fr-2023-04-14",
            displayName: "🇫🇷 French",
            description: Translation.tr("French speech recognition. ~123MB."),
            language: "fr",
            size: "~123MB",
            recommended: ""
        },
        // Korean
        {
            name: "sherpa-onnx-streaming-zipformer-korean-2024-06-16",
            displayName: "🇰🇷 Korean",
            description: Translation.tr("Korean speech recognition. ~125MB."),
            language: "ko",
            size: "~125MB",
            recommended: ""
        }
    ]
    property int selectedModelIndex: 0

    Process {
        id: liveCaptionDownloadProc

        property string statusLine: ""

        command: [root.liveCaptionDownloadScriptPath, root.availableModels[root.selectedModelIndex].name]
        stdout: StdioCollector {
            id: liveCaptionDownloadOut
            onStreamFinished: {
                const modelDir = liveCaptionDownloadOut.text.trim();
                if (modelDir.length === 0) {
                    Quickshell.execDetached([
                        "notify-send",
                        Translation.tr("Live caption"),
                        Translation.tr("Download finished but returned an empty model path."),
                        "-a",
                        "Shell",
                    ]);
                    return;
                }

                Config.options.liveCaption.modelDir = modelDir;
                Config.options.liveCaption.enabled = true;

                Quickshell.execDetached([
                    "notify-send",
                    Translation.tr("Live caption model ready"),
                    `${Translation.tr("Model directory")}: ${modelDir}`,
                    "-a",
                    "Shell",
                ]);
            }
        }
        stderr: SplitParser {
            onRead: line => {
                liveCaptionDownloadProc.statusLine = (line ?? "").trim();
            }
        }
        onExited: (exitCode, exitStatus) => {
            if (exitCode === 0) return;
            Quickshell.execDetached([
                "notify-send",
                Translation.tr("Live caption download failed"),
                `${Translation.tr("Exit code")}: ${exitCode}${liveCaptionDownloadProc.statusLine.length ? `\n${liveCaptionDownloadProc.statusLine}` : ""}`,
                "-a",
                "Shell",
            ]);
        }
    }

    Process {
        id: liveCaptionInstallDepsProc

        property string statusLine: ""

        command: [root.liveCaptionInstallDepsScriptPath]
        stdout: StdioCollector {
            id: liveCaptionInstallDepsOut
            onStreamFinished: {
                const version = liveCaptionInstallDepsOut.text.trim();
                Config.options.liveCaption.depsInstalled = true;
                Quickshell.execDetached([
                    "notify-send",
                    Translation.tr("Live caption deps installed"),
                    version.length ? `${Translation.tr("sherpa-onnx version")}: ${version}` : "",
                    "-a",
                    "Shell",
                ]);
            }
        }
        stderr: SplitParser {
            onRead: line => {
                liveCaptionInstallDepsProc.statusLine = (line ?? "").trim();
            }
        }
        onExited: (exitCode, exitStatus) => {
            if (exitCode === 0) return;
            Quickshell.execDetached([
                "notify-send",
                Translation.tr("Live caption deps install failed"),
                `${Translation.tr("Exit code")}: ${exitCode}${liveCaptionInstallDepsProc.statusLine.length ? `\n${liveCaptionInstallDepsProc.statusLine}` : ""}`,
                "-a",
                "Shell",
            ]);
        }
    }

    ContentSection {
        icon: "neurology"
        title: Translation.tr("AI")

        MaterialTextArea {
            Layout.fillWidth: true
            placeholderText: Translation.tr("System prompt")
            text: Config.options.ai.systemPrompt
            wrapMode: TextEdit.Wrap
            onTextChanged: {
                Qt.callLater(() => {
                    Config.options.ai.systemPrompt = text;
                });
            }
        }
    }

    ContentSection {
        icon: "music_cast"
        title: Translation.tr("Music Recognition")

        ConfigSpinBox {
            icon: "timer_off"
            text: Translation.tr("Total duration timeout (s)")
            value: Config.options.musicRecognition.timeout
            from: 10
            to: 100
            stepSize: 2
            onValueChanged: {
                Config.options.musicRecognition.timeout = value;
            }
        }
        ConfigSpinBox {
            icon: "av_timer"
            text: Translation.tr("Polling interval (s)")
            value: Config.options.musicRecognition.interval
            from: 2
            to: 10
            stepSize: 1
            onValueChanged: {
                Config.options.musicRecognition.interval = value;
            }
        }
    }

    ContentSection {
        icon: "subtitles"
        title: Translation.tr("Live caption")

        ConfigSwitch {
            buttonIcon: "subtitles"
            text: Translation.tr("Enable")
            checked: Config.options.liveCaption.enabled
            onCheckedChanged: {
                Config.options.liveCaption.enabled = checked;
            }
        }

        ContentSubsection {
            title: Translation.tr("Quick Setup")

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                RippleButtonWithIcon {
                    Layout.fillWidth: false
                    enabled: !liveCaptionInstallDepsProc.running && !liveCaptionDownloadProc.running
                    materialIcon: liveCaptionInstallDepsProc.running ? "downloading" : "terminal"
                    mainText: liveCaptionInstallDepsProc.running ? Translation.tr("Installing…") : Translation.tr("1. Install deps")
                    onClicked: {
                        liveCaptionInstallDepsProc.statusLine = "";
                        liveCaptionInstallDepsProc.running = true;
                    }
                    StyledToolTip {
                        text: Translation.tr("Installs sherpa-onnx into your Quickshell python venv.")
                    }
                }

                RippleButtonWithIcon {
                    Layout.fillWidth: false
                    enabled: !liveCaptionInstallDepsProc.running && !liveCaptionDownloadProc.running
                    materialIcon: liveCaptionDownloadProc.running ? "downloading" : "download"
                    mainText: liveCaptionDownloadProc.running ? Translation.tr("Downloading…") : Translation.tr("2. Download model")
                    onClicked: {
                        liveCaptionDownloadProc.statusLine = "";
                        liveCaptionDownloadProc.running = true;
                    }
                    StyledToolTip {
                        text: Translation.tr("Downloads selected model to ~/.local/share/sherpa-onnx/models")
                    }
                }
            }

            StyledText {
                Layout.fillWidth: true
                visible: liveCaptionDownloadProc.running && liveCaptionDownloadProc.statusLine.length > 0
                font.pixelSize: Appearance.font.pixelSize.smaller
                color: Appearance.colors.colSubtext
                text: liveCaptionDownloadProc.statusLine
                elide: Text.ElideRight
            }

            StyledText {
                Layout.fillWidth: true
                visible: liveCaptionInstallDepsProc.running && liveCaptionInstallDepsProc.statusLine.length > 0
                font.pixelSize: Appearance.font.pixelSize.smaller
                color: Appearance.colors.colSubtext
                text: liveCaptionInstallDepsProc.statusLine
                elide: Text.ElideRight
            }
        }

        ContentSubsection {
            title: Translation.tr("Model")

            StyledComboBox {
                Layout.fillWidth: true
                textRole: "displayName"
                model: root.availableModels
                currentIndex: root.selectedModelIndex
                onActivated: index => {
                    root.selectedModelIndex = index
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: modelInfoColumn.implicitHeight + 16
                radius: Appearance.rounding.small
                color: Appearance.colors.colSurfaceContainerLow

                ColumnLayout {
                    id: modelInfoColumn
                    anchors.fill: parent
                    anchors.margins: 8
                    spacing: 4

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 16

                        RowLayout {
                            spacing: 4
                            MaterialSymbol {
                                text: "storage"
                                iconSize: Appearance.font.pixelSize.normal
                                color: Appearance.colors.colSubtext
                            }
                            StyledText {
                                text: root.availableModels[root.selectedModelIndex].size
                                color: Appearance.colors.colSubtext
                                font.pixelSize: Appearance.font.pixelSize.small
                            }
                        }

                        RowLayout {
                            spacing: 4
                            MaterialSymbol {
                                text: "translate"
                                iconSize: Appearance.font.pixelSize.normal
                                color: Appearance.colors.colSubtext
                            }
                            StyledText {
                                text: root.availableModels[root.selectedModelIndex].language.toUpperCase()
                                color: Appearance.colors.colSubtext
                                font.pixelSize: Appearance.font.pixelSize.small
                            }
                        }

                        Rectangle {
                            visible: root.availableModels[root.selectedModelIndex].recommended === "balanced"
                            Layout.preferredHeight: recommendedText.implicitHeight + 4
                            Layout.preferredWidth: recommendedText.implicitWidth + 12
                            radius: height / 2
                            color: Appearance.colors.colPrimaryContainer

                            StyledText {
                                id: recommendedText
                                anchors.centerIn: parent
                                text: Translation.tr("Recommended")
                                color: Appearance.colors.colOnPrimaryContainer
                                font.pixelSize: Appearance.font.pixelSize.smaller
                                font.bold: true
                            }
                        }

                        Rectangle {
                            visible: root.availableModels[root.selectedModelIndex].recommended === "low-end"
                            Layout.preferredHeight: lowEndText.implicitHeight + 4
                            Layout.preferredWidth: lowEndText.implicitWidth + 12
                            radius: height / 2
                            color: Appearance.colors.colSecondaryContainer

                            StyledText {
                                id: lowEndText
                                anchors.centerIn: parent
                                text: Translation.tr("Best for low-end PCs")
                                color: Appearance.colors.colOnSecondaryContainer
                                font.pixelSize: Appearance.font.pixelSize.smaller
                                font.bold: true
                            }
                        }
                    }

                    StyledText {
                        Layout.fillWidth: true
                        text: root.availableModels[root.selectedModelIndex].description
                        color: Appearance.colors.colOnSurface
                        font.pixelSize: Appearance.font.pixelSize.small
                        wrapMode: Text.Wrap
                    }
                }
            }

            StyledText {
                Layout.fillWidth: true
                Layout.topMargin: 8
                text: Translation.tr("Or use a custom model:")
                color: Appearance.colors.colSubtext
                font.pixelSize: Appearance.font.pixelSize.smaller
            }

            MaterialTextArea {
                Layout.fillWidth: true
                placeholderText: Translation.tr("Custom model directory path (optional)")
                text: Config.options.liveCaption.modelDir
                wrapMode: TextEdit.Wrap
                onTextChanged: {
                    Config.options.liveCaption.modelDir = text;
                }
            }
        }

        ContentSubsection {
            title: Translation.tr("Audio Source")

            StyledComboBox {
                id: liveCaptionSourceSelector
                textRole: "displayName"

                model: [
                    { displayName: Translation.tr("System audio (speaker)"), value: "monitor", icon: "volume_up" },
                    { displayName: Translation.tr("Microphone"), value: "input", icon: "mic" },
                    { displayName: Translation.tr("Both (Mic + System)"), value: "both", icon: "compare_arrows" },
                ]

                currentIndex: {
                    const value = Config.options.liveCaption.source ?? "monitor";
                    const index = model.findIndex(item => item.value === value);
                    return index !== -1 ? index : 0;
                }

                onActivated: index => {
                    Config.options.liveCaption.source = model[index].value;
                }
            }
        }

        ContentSubsection {
            title: Translation.tr("Advanced")

            ConfigRow {
                uniform: true
                ConfigSpinBox {
                    icon: "memory"
                    text: Translation.tr("Threads")
                    value: Config.options.liveCaption.numThreads
                    from: 1
                    to: 16
                    stepSize: 1
                    onValueChanged: {
                        Config.options.liveCaption.numThreads = value;
                    }
                }
                ConfigSpinBox {
                    icon: "av_timer"
                    text: Translation.tr("Update interval (ms)")
                value: Config.options.liveCaption.updateIntervalMs
                from: 0
                to: 2000
                stepSize: 50
                onValueChanged: {
                    Config.options.liveCaption.updateIntervalMs = value;
                }
            }
        }

            ConfigSpinBox {
                icon: "history"
                text: Translation.tr("History (characters)")
                value: Config.options.liveCaption.historyChars
                from: 0
                to: 2000
                stepSize: 20
                onValueChanged: {
                    Config.options.liveCaption.historyChars = value;
                }
            }
        }
    }

    ContentSection {
        icon: "cell_tower"
        title: Translation.tr("Networking")

        MaterialTextArea {
            Layout.fillWidth: true
            placeholderText: Translation.tr("User agent (for services that require it)")
            text: Config.options.networking.userAgent
            wrapMode: TextEdit.Wrap
            onTextChanged: {
                Config.options.networking.userAgent = text;
            }
        }
    }

    ContentSection {
        icon: "memory"
        title: Translation.tr("Resources")

        ConfigSpinBox {
            icon: "av_timer"
            text: Translation.tr("Polling interval (ms)")
            value: Config.options.resources.updateInterval
            from: 100
            to: 10000
            stepSize: 100
            onValueChanged: {
                Config.options.resources.updateInterval = value;
            }
        }
        
    }

    ContentSection {
        icon: "file_open"
        title: Translation.tr("Save paths")

        MaterialTextArea {
            Layout.fillWidth: true
            placeholderText: Translation.tr("Video Recording Path")
            text: Config.options.screenRecord.savePath
            wrapMode: TextEdit.Wrap
            onTextChanged: {
                Config.options.screenRecord.savePath = text;
            }
        }
        
        MaterialTextArea {
            Layout.fillWidth: true
            placeholderText: Translation.tr("Screenshot Path (leave empty to just copy)")
            text: Config.options.screenSnip.savePath
            wrapMode: TextEdit.Wrap
            onTextChanged: {
                Config.options.screenSnip.savePath = text;
            }
        }
    }

    ContentSection {
        icon: "search"
        title: Translation.tr("Search")

        ConfigSwitch {
            text: Translation.tr("Use Levenshtein distance-based algorithm instead of fuzzy")
            checked: Config.options.search.sloppy
            onCheckedChanged: {
                Config.options.search.sloppy = checked;
            }
            StyledToolTip {
                text: Translation.tr("Could be better if you make a ton of typos,\nbut results can be weird and might not work with acronyms\n(e.g. \"GIMP\" might not give you the paint program)")
            }
        }

        ContentSubsection {
            title: Translation.tr("Prefixes")
            ConfigRow {
                uniform: true
                MaterialTextArea {
                    Layout.fillWidth: true
                    placeholderText: Translation.tr("Action")
                    text: Config.options.search.prefix.action
                    wrapMode: TextEdit.Wrap
                    onTextChanged: {
                        Config.options.search.prefix.action = text;
                    }
                }
                MaterialTextArea {
                    Layout.fillWidth: true
                    placeholderText: Translation.tr("Clipboard")
                    text: Config.options.search.prefix.clipboard
                    wrapMode: TextEdit.Wrap
                    onTextChanged: {
                        Config.options.search.prefix.clipboard = text;
                    }
                }
                MaterialTextArea {
                    Layout.fillWidth: true
                    placeholderText: Translation.tr("Emojis")
                    text: Config.options.search.prefix.emojis
                    wrapMode: TextEdit.Wrap
                    onTextChanged: {
                        Config.options.search.prefix.emojis = text;
                    }
                }
            }

            ConfigRow {
                uniform: true
                MaterialTextArea {
                    Layout.fillWidth: true
                    placeholderText: Translation.tr("Math")
                    text: Config.options.search.prefix.math
                    wrapMode: TextEdit.Wrap
                    onTextChanged: {
                        Config.options.search.prefix.math = text;
                    }
                }
                MaterialTextArea {
                    Layout.fillWidth: true
                    placeholderText: Translation.tr("Shell command")
                    text: Config.options.search.prefix.shellCommand
                    wrapMode: TextEdit.Wrap
                    onTextChanged: {
                        Config.options.search.prefix.shellCommand = text;
                    }
                }
                MaterialTextArea {
                    Layout.fillWidth: true
                    placeholderText: Translation.tr("Web search")
                    text: Config.options.search.prefix.webSearch
                    wrapMode: TextEdit.Wrap
                    onTextChanged: {
                        Config.options.search.prefix.webSearch = text;
                    }
                }
            }
        }
        ContentSubsection {
            title: Translation.tr("Web search")
            MaterialTextArea {
                Layout.fillWidth: true
                placeholderText: Translation.tr("Base URL")
                text: Config.options.search.engineBaseUrl
                wrapMode: TextEdit.Wrap
                onTextChanged: {
                    Config.options.search.engineBaseUrl = text;
                }
            }
        }
    }

    ContentSection {
        icon: "weather_mix"
        title: Translation.tr("Weather")
        ConfigRow {
            ConfigSwitch {
                buttonIcon: "assistant_navigation"
                text: Translation.tr("Enable GPS based location")
                checked: Config.options.bar.weather.enableGPS
                onCheckedChanged: {
                    Config.options.bar.weather.enableGPS = checked;
                }
            }
            ConfigSwitch {
                buttonIcon: "thermometer"
                text: Translation.tr("Fahrenheit unit")
                checked: Config.options.bar.weather.useUSCS
                onCheckedChanged: {
                    Config.options.bar.weather.useUSCS = checked;
                }
                StyledToolTip {
                    text: Translation.tr("It may take a few seconds to update")
                }
            }
        }
        
        MaterialTextArea {
            Layout.fillWidth: true
            placeholderText: Translation.tr("City name")
            text: Config.options.bar.weather.city
            wrapMode: TextEdit.Wrap
            onTextChanged: {
                Config.options.bar.weather.city = text;
            }
        }
        ConfigSpinBox {
            icon: "av_timer"
            text: Translation.tr("Polling interval (m)")
            value: Config.options.bar.weather.fetchInterval
            from: 5
            to: 50
            stepSize: 5
            onValueChanged: {
                Config.options.bar.weather.fetchInterval = value;
            }
        }
    }
}
