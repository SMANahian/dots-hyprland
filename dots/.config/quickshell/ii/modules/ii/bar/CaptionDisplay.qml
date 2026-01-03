import QtQuick
import Quickshell
import qs.modules.common

Item {
    id: root
    property string text: ""
    property font font: Qt.font({family: "Google Sans Flex", pixelSize: 14})
    property color color: "white"
    property int maxLines: 3

    property string committedHistory: ""
    property string lastReceivedText: ""
    property var displayLines: []

    TextMetrics {
        id: metrics
        font: root.font
    }

    onTextChanged: {
        if (text === "" && lastReceivedText !== "") {
            committedHistory += (committedHistory ? " " : "") + lastReceivedText
            // Prune history to keep it manageable (e.g. last 1000 chars)
            if (committedHistory.length > 1000) {
                var cut = committedHistory.indexOf(" ", committedHistory.length - 800)
                if (cut !== -1) committedHistory = committedHistory.substring(cut + 1)
            }
        }
        lastReceivedText = text
        updateDisplayLines()
    }

    onWidthChanged: updateDisplayLines()

    function updateDisplayLines() {
        if (root.width <= 0) return

        var fullText = committedHistory + (text ? (committedHistory ? " " : "") + text : "")
        var words = fullText.split(" ")
        var lines = []
        var currentLine = ""

        for (var i = 0; i < words.length; i++) {
            var word = words[i]
            if (word === "") continue
            
            var testLine = currentLine + (currentLine ? " " : "") + word
            metrics.text = testLine
            
            if (metrics.width > root.width) {
                if (currentLine) lines.push(currentLine)
                currentLine = word
            } else {
                currentLine = testLine
            }
        }
        if (currentLine) lines.push(currentLine)

        // Pad with empty lines if needed, or just take last N
        var result = []
        for (var j = 0; j < maxLines; j++) {
            var idx = lines.length - maxLines + j
            if (idx >= 0) {
                result.push(lines[idx])
            } else {
                result.push("")
            }
        }
        displayLines = result
    }

    Column {
        anchors.fill: parent
        spacing: 0 // Adjust if needed
        
        Repeater {
            model: root.maxLines
            delegate: Text {
                width: root.width
                height: root.height / root.maxLines
                text: root.displayLines[index] || ""
                font: root.font
                color: root.color
                horizontalAlignment: Text.AlignLeft
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
            }
        }
    }
}
