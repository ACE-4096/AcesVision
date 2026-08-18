pragma ComponentBehavior: Bound
// qmllint disable unqualified

import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

ApplicationWindow {
    id: window
    width: 1280
    height: 800
    minimumWidth: 980
    minimumHeight: 640
    visible: true
    title: "AcesVision"
    color: "#0e1014"
    Material.theme: Material.Dark
    Material.accent: accent

    property color panel: "#171a20"
    property color panelRaised: "#1e222a"
    property color border: "#303641"
    property color textMain: "#eef1f5"
    property color textMuted: "#929aa8"
    property color accent: "#8c6cff"
    property color good: "#4fd18b"
    property color warning: "#f5b942"
    property int currentPage: 0
    readonly property int pageMargin: 20

    component Card: Rectangle {
        color: window.panel
        border.color: window.border
        border.width: 1
        radius: 10
    }

    component NavButton: Button {
        id: navControl
        required property int page
        Layout.fillWidth: true
        checkable: true
        checked: window.currentPage === page
        onClicked: window.currentPage = page
        leftPadding: 16
        contentItem: Text {
            text: navControl.text
            color: navControl.checked ? window.textMain : window.textMuted
            font.pixelSize: 14
            font.weight: navControl.checked ? Font.DemiBold : Font.Normal
            verticalAlignment: Text.AlignVCenter
        }
        background: Rectangle {
            color: navControl.checked ? "#29243d" : (navControl.hovered ? "#20242b" : "transparent")
            radius: 7
            border.color: navControl.checked ? "#5b4c91" : "transparent"
        }
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.preferredWidth: 220
            Layout.fillHeight: true
            color: "#121419"
            border.color: window.border

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 14
                spacing: 7

                Text {
                    text: "AcesVision"
                    color: window.textMain
                    font.pixelSize: 22
                    font.bold: true
                    Layout.leftMargin: 8
                    Layout.topMargin: 8
                    Layout.bottomMargin: 18
                }

                NavButton { text: "Live"; page: 0 }
                NavButton { text: "Sources"; page: 1 }
                NavButton { text: "Overlay Studio"; page: 2 }
                NavButton { text: "Gestures and Rules"; page: 3 }
                NavButton { text: "People"; page: 4 }
                NavButton { text: "Models and Security"; page: 5 }
                Item { Layout.fillHeight: true }

                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: 74
                    color: window.panel
                    radius: 8
                    border.color: window.border
                    Column {
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 5
                        Text {
                            text: vision.status === "live" ? "Runtime live" : vision.status
                            color: vision.status === "live" ? window.good : window.warning
                            font.bold: true
                        }
                        Text {
                            width: parent.width
                            text: vision.sourceLabel
                            color: window.textMuted
                            elide: Text.ElideRight
                            font.pixelSize: 11
                        }
                        Text {
                            text: "Connectors: " + (vision.executableConnectors.length
                                ? vision.executableConnectors.join(", ")
                                : "none wired")
                            color: window.textMuted
                            font.pixelSize: 11
                        }
                    }
                }
            }
        }

        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: window.currentPage

            // Live
            ScrollView {
                id: livePage
                objectName: "page0"
                // Page gutter lives on the ScrollView, not on the ColumnLayout.
                // A ColumnLayout sized by `width: parent.width` has no anchors,
                // so anchors.margins on it was inert and every page ran its
                // titles and cards straight into the window edge.
                padding: window.pageMargin
                contentWidth: availableWidth
                ColumnLayout {
                    width: parent.width
                    spacing: 14

                    RowLayout {
                        Layout.fillWidth: true
                        Text { text: "Live"; color: window.textMain; font.pixelSize: 26; font.bold: true }
                        Item { Layout.fillWidth: true }
                        Text { text: "Frame " + vision.sequence; color: window.textMuted }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 10
                        Repeater {
                            model: [
                                { label: "Source", value: vision.sourceLabel },
                                { label: "Capture", value: vision.captureFps.toFixed(1) + " FPS" },
                                { label: "Inference", value: vision.inferenceFps.toFixed(1) + " FPS / " + vision.inferenceMs.toFixed(0) + " ms" },
                                { label: "Model", value: vision.modelSummary },
                                { label: "Security", value: "not verified" }
                            ]
                            delegate: Card {
                                required property var modelData
                                Layout.fillWidth: true
                                implicitHeight: 70
                                Column {
                                    anchors.fill: parent
                                    anchors.margins: 10
                                    spacing: 5
                                    Text { text: modelData.label; color: window.textMuted; font.pixelSize: 11 }
                                    Text {
                                        width: parent.width
                                        text: modelData.value
                                        color: window.textMain
                                        font.bold: true
                                        elide: Text.ElideRight
                                    }
                                }
                            }
                        }
                    }

                    RowLayout {
                        id: liveStage
                        objectName: "liveStage"
                        readonly property int tuningWidth: 320
                        // Video is 16:9, so key the stage to the width the VIDEO
                        // gets, and never let it grow past the viewport. Keying it
                        // to the row width alone made the page TALLER as the window
                        // got WIDER — at 1920x1080 the whole page scrolled and the
                        // OBS/gesture bar went off screen.
                        readonly property real videoWidth:
                            Math.max(240, width - tuningWidth - spacing)
                        // Everything else on this page measures ~205 px.
                        readonly property real viewportBudget:
                            livePage.availableHeight - 210
                        // Never shorter than the tuning panel needs; clipping the
                        // controls is worse than scrolling the page.
                        readonly property real minimumHeight:
                            Math.max(390, liveTuning.implicitHeight + 28)
                        Layout.fillWidth: true
                        Layout.preferredHeight: Math.max(
                            minimumHeight, Math.min(videoWidth * 9 / 16, viewportBudget))
                        spacing: 12

                        Card {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            color: "#050608"
                            clip: true
                            Image {
                                id: previewImage
                                objectName: "previewImage"
                                // Bumped only to force a refetch after a failure.
                                property int retryTick: 0
                                readonly property bool failed: status === Image.Error
                                anchors.fill: parent
                                anchors.margins: 1
                                // Nothing to fetch before the first frame exists —
                                // asking earlier just logged a 503 per attempt.
                                source: vision.sequence > 0
                                        ? vision.previewSource + "&retry=" + retryTick
                                        : ""
                                cache: false
                                asynchronous: true
                                fillMode: Image.PreserveAspectFit
                                onStatusChanged: if (status === Image.Error) previewRetry.restart()
                            }
                            // The preview server answers 503 until the first frame
                            // exists. Retry quietly instead of leaving a dead tile.
                            Timer {
                                id: previewRetry
                                interval: 1000
                                onTriggered: previewImage.retryTick++
                            }
                            // A stalled feed keeps the last good frame on screen and
                            // reads as live video. Scrim it and say what happened.
                            Rectangle {
                                anchors.fill: parent
                                anchors.margins: 1
                                visible: previewImage.failed || vision.previewStale
                                color: "#b0050608"
                            }
                            Text {
                                anchors.centerIn: parent
                                width: parent.width - 40
                                horizontalAlignment: Text.AlignHCenter
                                wrapMode: Text.Wrap
                                visible: previewImage.failed || vision.previewStale
                                         || vision.status !== "live"
                                color: previewImage.failed || vision.previewStale
                                       ? window.warning : window.textMuted
                                font.pixelSize: 18
                                text: previewImage.failed
                                      ? "Preview feed unavailable — retrying"
                                      : vision.status === "reconnecting"
                                        ? "Waiting for camera"
                                        : vision.previewStale
                                          ? "Frame is stale — this is not live video"
                                          : vision.status === "live"
                                            ? ""
                                            : "Starting vision runtime"
                            }
                        }

                        Card {
                            Layout.preferredWidth: 320
                            Layout.fillHeight: true
                            ColumnLayout {
                                id: liveTuning
                                objectName: "liveTuning"
                                anchors.fill: parent
                                anchors.margins: 14
                                spacing: 7

                                function applyTuning() {
                                    vision.setCameraTuning(
                                        autoExposure.checked,
                                        exposureControl.value,
                                        brightnessControl.value,
                                        contrastControl.value,
                                        gammaControl.value)
                                }

                                Text { text: "Image tuning"; color: window.textMain; font.bold: true; font.pixelSize: 16 }
                                Text {
                                    text: "Changes apply live. Automatic exposure is usually brighter; manual gives a steadier frame rate."
                                    color: window.textMuted
                                    wrapMode: Text.Wrap
                                    Layout.fillWidth: true
                                }
                                Switch {
                                    id: autoExposure
                                    text: vision.manualExposureSupported
                                          ? "Automatic exposure"
                                          : "Automatic exposure (required for this camera)"
                                    checked: true
                                    enabled: vision.manualExposureSupported
                                    onToggled: liveTuning.applyTuning()
                                }
                                // The backend can force auto back on when it catches
                                // a camera going black in manual mode, so mirror it.
                                Connections {
                                    target: vision
                                    function onCameraTuningChanged() {
                                        autoExposure.checked = vision.autoExposure
                                    }
                                }
                                Text {
                                    visible: vision.exposureNotice.length > 0
                                    text: vision.exposureNotice
                                    color: window.warning
                                    wrapMode: Text.Wrap
                                    Layout.fillWidth: true
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    Text { text: "Exposure"; color: window.textMuted; Layout.preferredWidth: 82 }
                                    SpinBox {
                                        id: exposureControl
                                        from: 3; to: 333; value: 166; editable: true
                                        enabled: vision.manualExposureSupported && !autoExposure.checked
                                        Layout.fillWidth: true
                                        onValueModified: liveTuning.applyTuning()
                                    }
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    Text { text: "Brightness"; color: window.textMuted; Layout.preferredWidth: 82 }
                                    SpinBox {
                                        id: brightnessControl
                                        from: -64; to: 64; value: 0; editable: true
                                        Layout.fillWidth: true
                                        onValueModified: liveTuning.applyTuning()
                                    }
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    Text { text: "Contrast"; color: window.textMuted; Layout.preferredWidth: 82 }
                                    SpinBox {
                                        id: contrastControl
                                        from: 0; to: 95; value: 0; editable: true
                                        Layout.fillWidth: true
                                        onValueModified: liveTuning.applyTuning()
                                    }
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    Text { text: "Gamma"; color: window.textMuted; Layout.preferredWidth: 82 }
                                    SpinBox {
                                        id: gammaControl
                                        from: 100; to: 300; value: 100; editable: true
                                        Layout.fillWidth: true
                                        onValueModified: liveTuning.applyTuning()
                                    }
                                }
                                RowLayout {
                                    Button {
                                        text: "Reset"
                                        onClicked: {
                                            autoExposure.checked = true
                                            exposureControl.value = 166
                                            brightnessControl.value = 0
                                            contrastControl.value = 0
                                            gammaControl.value = 100
                                            vision.setCameraTuning(true, 166, 0, 0, 100)
                                        }
                                    }
                                    Text { text: "Luma " + vision.imageMean.toFixed(1); color: window.textMuted }
                                    Item { Layout.fillWidth: true }
                                }
                                Text {
                                    visible: vision.imageWarning.length > 0
                                    text: vision.imageWarning
                                    color: window.warning
                                    wrapMode: Text.Wrap
                                    Layout.fillWidth: true
                                }
                                Item { Layout.fillHeight: true }
                            }
                        }
                    }

                    Card {
                        Layout.fillWidth: true
                        implicitHeight: controls.implicitHeight + 24
                        RowLayout {
                            id: controls
                            anchors.fill: parent
                            anchors.margins: 12
                            spacing: 18
                            Switch {
                                text: "OBS output"
                                checked: vision.obsEnabled
                                onToggled: vision.setObsEnabled(checked)
                            }
                            Switch {
                                text: "Gesture events"
                                checked: vision.eventsEnabled
                                onToggled: vision.setEventsEnabled(checked)
                            }
                            Item { Layout.fillWidth: true }
                            Text { text: vision.lastGesture; color: window.textMuted }
                        }
                    }

                    // Persists until another failure supersedes it or the operator
                    // dismisses it. It used to be erased by the next 100 ms
                    // refresh tick, so a failed action flashed for under a frame.
                    Card {
                        Layout.fillWidth: true
                        visible: vision.lastError.length > 0
                        color: "#2a1417"
                        border.color: "#7a3138"
                        implicitHeight: errorRow.implicitHeight + 24
                        RowLayout {
                            id: errorRow
                            objectName: "errorRow"
                            anchors.fill: parent
                            anchors.margins: 12
                            spacing: 12
                            Text {
                                text: vision.lastError
                                color: "#ff9a9a"
                                wrapMode: Text.Wrap
                                Layout.fillWidth: true
                            }
                            Button {
                                text: "Dismiss"
                                visible: vision.errorDismissable
                                onClicked: vision.dismissError()
                            }
                        }
                    }
                }
            }

            // Sources
            ScrollView {
                objectName: "page1"
                padding: window.pageMargin
                contentWidth: availableWidth
                ColumnLayout {
                    width: parent.width
                    spacing: 14
                    Text { text: "Sources"; color: window.textMain; font.pixelSize: 26; font.bold: true }
                    Text {
                        text: "Switch inputs without restarting inference or changing outputs."
                        color: window.textMuted
                    }
                    Card {
                        Layout.fillWidth: true
                        implicitHeight: webcamColumn.implicitHeight + 28
                        ColumnLayout {
                            id: webcamColumn
                            anchors.fill: parent
                            anchors.margins: 14
                            Text { text: "Physical webcam"; color: window.textMain; font.bold: true }
                            Text {
                                text: "Named devices are read from Linux without opening or locking the camera. Metadata-only nodes are hidden."
                                color: window.textMuted
                                wrapMode: Text.Wrap
                                Layout.fillWidth: true
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                ComboBox {
                                    id: webcamPicker
                                    model: vision.webcams
                                    textRole: "label"
                                    valueRole: "index"
                                    Layout.fillWidth: true
                                    displayText: count > 0 ? currentText : "No physical cameras found"
                                }
                                Button { text: "Refresh"; onClicked: vision.refreshWebcams() }
                                Button {
                                    text: "Use selected"
                                    enabled: webcamPicker.count > 0
                                    onClicked: vision.useWebcamIndex(webcamPicker.currentValue)
                                }
                            }
                        }
                    }
                    Card {
                        Layout.fillWidth: true
                        implicitHeight: droidColumn.implicitHeight + 28
                        ColumnLayout {
                            id: droidColumn
                            anchors.fill: parent
                            anchors.margins: 14
                            Text { text: "DroidCam"; color: window.textMain; font.bold: true }
                            Text {
                                text: "Scan checks only private local /24 networks on DroidCam port 4747. It runs only when you click Scan."
                                color: window.textMuted
                                wrapMode: Text.Wrap
                                Layout.fillWidth: true
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                ComboBox {
                                    id: droidPicker
                                    model: vision.droidCams
                                    textRole: "label"
                                    valueRole: "url"
                                    Layout.fillWidth: true
                                    displayText: count > 0 ? currentText : "No discovered DroidCam devices"
                                }
                                Button {
                                    text: vision.droidScanActive ? "Scanning..." : "Scan local network"
                                    enabled: !vision.droidScanActive
                                    onClicked: vision.scanDroidCams()
                                }
                                Button {
                                    text: "Use selected"
                                    enabled: droidPicker.count > 0
                                    onClicked: vision.useDroidCam(droidPicker.currentValue)
                                }
                            }
                            Text { text: vision.droidScanStatus; color: window.textMuted; font.pixelSize: 11 }
                            Text { text: "Manual URL"; color: window.textMuted; font.pixelSize: 11 }
                            RowLayout {
                                Layout.fillWidth: true
                                TextField {
                                    id: droidUrl
                                    placeholderText: "http://PHONE_IP:4747/video"
                                    Layout.fillWidth: true
                                }
                                Button { text: "Use URL"; onClicked: vision.useDroidCam(droidUrl.text) }
                            }
                            Text { text: "Network feeds are not trusted for privileged authorization."; color: window.warning }
                        }
                    }
                }
            }

            // Overlay Studio
            ScrollView {
                objectName: "page2"
                padding: window.pageMargin
                contentWidth: availableWidth
                ColumnLayout {
                    width: parent.width
                    spacing: 14
                    Text { text: "Overlay Studio"; color: window.textMain; font.pixelSize: 26; font.bold: true }
                    Text { text: "Profiles render independently from the shared raw frame."; color: window.textMuted }
                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2
                        columnSpacing: 12
                        rowSpacing: 12
                        Repeater {
                            // The custom profile gets a card as soon as one has
                            // been applied. Without it, "Apply custom" selected a
                            // profile no card knew about and all four read "Select".
                            model: {
                                var profiles = [
                                    { id: "clean", title: "Clean", body: "Raw video without annotations" },
                                    { id: "minimal", title: "Minimal", body: "Subtle boxes and compact labels" },
                                    { id: "broadcast", title: "Broadcast", body: "High-contrast OBS styling" },
                                    { id: "security", title: "Security", body: "Scores and security state" }
                                ]
                                if (vision.customOverlayReady)
                                    profiles.push({ id: "custom", title: "Custom",
                                                    body: "The box style you applied below" })
                                return profiles
                            }
                            delegate: Card {
                                id: overlayCard
                                required property var modelData
                                // The model list is rebuilt when the custom profile
                                // appears, and a delegate briefly sees an undefined
                                // modelData while that happens. Read it once, safely.
                                readonly property string profileId: modelData ? modelData.id : ""
                                readonly property string profileTitle: modelData ? modelData.title : ""
                                readonly property string profileBody: modelData ? modelData.body : ""
                                readonly property bool active:
                                    profileId !== "" && vision.overlayProfile === profileId
                                Layout.fillWidth: true
                                implicitHeight: 110
                                border.color: overlayCard.active ? window.accent : window.border
                                MouseArea {
                                    anchors.fill: parent
                                    onClicked: if (overlayCard.profileId !== "")
                                                   vision.setOverlayProfile(overlayCard.profileId)
                                }
                                Column {
                                    anchors.fill: parent
                                    anchors.margins: 14
                                    spacing: 7
                                    Text { text: overlayCard.profileTitle; color: window.textMain; font.bold: true; font.pixelSize: 16 }
                                    Text { text: overlayCard.profileBody; color: window.textMuted }
                                    Text {
                                        text: overlayCard.active ? "Active" : "Select"
                                        color: overlayCard.active ? window.accent : window.textMuted
                                    }
                                }
                            }
                        }
                    }
                    Card {
                        Layout.fillWidth: true
                        implicitHeight: styleEditor.implicitHeight + 28
                        ColumnLayout {
                            id: styleEditor
                            anchors.fill: parent
                            anchors.margins: 14
                            spacing: 10
                            Text { text: "Custom box style"; color: window.textMain; font.bold: true }
                            RowLayout {
                                Switch { id: showObjects; text: "Object boxes"; checked: true }
                                Switch { id: showFaces; text: "Face boxes"; checked: true }
                                Switch { id: showGestures; text: "Gesture boxes"; checked: true }
                                Item { Layout.fillWidth: true }
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                Text { text: "Line width"; color: window.textMuted }
                                Slider { id: lineWidth; from: 1; to: 8; stepSize: 1; value: 2; Layout.fillWidth: true }
                                Text { text: Math.round(lineWidth.value) + " px"; color: window.textMain }
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                Text { text: "Label size"; color: window.textMuted }
                                Slider { id: fontScale; from: 0.3; to: 2.0; stepSize: 0.1; value: 0.6; Layout.fillWidth: true }
                                Text { text: fontScale.value.toFixed(1); color: window.textMain }
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                TextField { id: objectColour; text: "#3cAAff"; placeholderText: "Objects"; Layout.fillWidth: true }
                                TextField { id: knownColour; text: "#00c800"; placeholderText: "Known"; Layout.fillWidth: true }
                                TextField { id: unknownColour; text: "#dc2828"; placeholderText: "Unknown"; Layout.fillWidth: true }
                                TextField { id: gestureColour; text: "#1ea0e6"; placeholderText: "Gesture"; Layout.fillWidth: true }
                                Button {
                                    text: "Apply custom"
                                    onClicked: vision.applyOverlayStyle(
                                        showObjects.checked, showFaces.checked, showGestures.checked,
                                        Math.round(lineWidth.value), fontScale.value,
                                        objectColour.text, knownColour.text,
                                        unknownColour.text, gestureColour.text)
                                }
                            }
                        }
                    }
                    Text {
                        text: "Landmarks, trails, zones, privacy effects, and saving or loading profiles are not available yet."
                        color: window.textMuted
                        wrapMode: Text.Wrap
                        Layout.fillWidth: true
                    }
                }
            }

            // Gestures and Rules
            ScrollView {
                objectName: "page3"
                padding: window.pageMargin
                contentWidth: availableWidth
                ColumnLayout {
                    width: parent.width
                    spacing: 14
                    Text { text: "Gestures and Rules"; color: window.textMain; font.pixelSize: 26; font.bold: true }
                    Card {
                        Layout.fillWidth: true
                        implicitHeight: 132
                        Column {
                            anchors.fill: parent
                            anchors.margins: 14
                            spacing: 8
                            Text { text: "Event output"; color: window.textMain; font.bold: true }
                            Text { text: vision.lastGesture; color: window.textMuted }
                            Text { text: vision.lastDecision; color: window.textMuted }
                            Text {
                                text: "Rules are dry-run until armed one at a time. "
                                    + "Failures show above and in the error banner."
                                color: window.warning
                            }
                        }
                    }

                    Card {
                        Layout.fillWidth: true
                        implicitHeight: ruleForm.implicitHeight + 28
                        ColumnLayout {
                            id: ruleForm
                            anchors.fill: parent
                            anchors.margins: 14
                            spacing: 10
                            Text { text: "Add rule (starts dry-run)"; color: window.textMain; font.bold: true }
                            RowLayout {
                                Layout.fillWidth: true
                                // Catalog-backed, not free text: a mistyped
                                // gesture or actor produced a rule that could
                                // never fire and never said so.
                                ComboBox {
                                    id: ruleGesture
                                    model: vision.gestureNames
                                    Layout.fillWidth: true
                                }
                                ComboBox {
                                    id: ruleActor
                                    model: vision.actorNames
                                    Layout.fillWidth: true
                                }
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                ComboBox {
                                    id: ruleConnector
                                    model: vision.connectorNames
                                    Layout.fillWidth: true
                                    onCurrentTextChanged: {
                                        ruleAction.model = vision.connectorActions(currentText)
                                        ruleAction.currentIndex = 0
                                    }
                                }
                                ComboBox {
                                    id: ruleAction
                                    model: vision.connectorActions(ruleConnector.currentText)
                                    Layout.fillWidth: true
                                }
                                Button {
                                    text: "Add rule"
                                    enabled: ruleGesture.currentText.length > 0
                                             && ruleConnector.currentText.length > 0
                                             && ruleAction.currentText.length > 0
                                    onClicked: {
                                        vision.addRule(ruleGesture.currentText,
                                                       ruleActor.currentText,
                                                       ruleConnector.currentText,
                                                       ruleAction.currentText)
                                    }
                                }
                            }
                        }
                    }

                    Text {
                        visible: vision.rules.length === 0
                        text: "No rules configured. Add one above — every new rule starts in dry-run until you arm it."
                        color: window.textMuted
                    }

                    Repeater {
                        model: vision.rules
                        delegate: Card {
                            required property var modelData
                            Layout.fillWidth: true
                            implicitHeight: 76
                            RowLayout {
                                anchors.fill: parent
                                anchors.margins: 12
                                ColumnLayout {
                                    Layout.fillWidth: true
                                    Text {
                                        text: modelData.gesture + "  ->  " + modelData.connector + "." + modelData.action
                                        color: window.textMain
                                        font.bold: true
                                    }
                                    Text {
                                        text: "Actor: " + modelData.actor
                                            + " | Risk: " + modelData.risk
                                            + " | " + (modelData.dryRun ? "Dry-run" : "ARMED")
                                            + (modelData.executable ? "" : " | no executor")
                                        color: modelData.dryRun ? window.textMuted : window.warning
                                    }
                                }
                                // Per-rule arming. Nothing else flips dry_run.
                                Button {
                                    text: modelData.dryRun ? "Arm" : "Disarm"
                                    enabled: modelData.executable || !modelData.dryRun
                                    onClicked: vision.setRuleDryRun(modelData.id, !modelData.dryRun)
                                }
                                Button { text: "Remove"; onClicked: vision.removeRule(modelData.id) }
                            }
                        }
                    }
                }
            }

            // People
            ScrollView {
                objectName: "page4"
                padding: window.pageMargin
                contentWidth: availableWidth
                ColumnLayout {
                    width: parent.width
                    spacing: 14
                    Text { text: "People"; color: window.textMain; font.pixelSize: 26; font.bold: true }
                    Card {
                        Layout.fillWidth: true
                        implicitHeight: 130
                        Column {
                            anchors.fill: parent
                            anchors.margins: 14
                            spacing: 8
                            Text { text: "Secure enrolment is not active yet"; color: window.warning; font.bold: true }
                            Text {
                                width: parent.width
                                text: "Enrolled faces are used for labelling only. AcesVision will not present them as verified authentication."
                                color: window.textMuted
                                wrapMode: Text.Wrap
                            }
                        }
                    }
                    Card {
                        Layout.fillWidth: true
                        implicitHeight: enrolledColumn.implicitHeight + 28
                        ColumnLayout {
                            id: enrolledColumn
                            anchors.fill: parent
                            anchors.margins: 14
                            spacing: 8
                            RowLayout {
                                Layout.fillWidth: true
                                Text { text: "Enrolled identities"; color: window.textMain; font.bold: true }
                                Item { Layout.fillWidth: true }
                                Button { text: "Rescan"; onClicked: vision.refreshActors() }
                            }
                            Text {
                                // Re-read while the app runs, so somebody enrolled
                                // now can be picked as a rule actor without a restart.
                                text: vision.actorNames.length > 1
                                      ? vision.actorNames.filter(function (name) { return name !== "*" }).join(", ")
                                      : "Nobody is enrolled yet. Rules can still target anyone with the * actor."
                                color: window.textMuted
                                wrapMode: Text.Wrap
                                Layout.fillWidth: true
                            }
                        }
                    }
                }
            }

            // Models and Security
            ScrollView {
                objectName: "page5"
                padding: window.pageMargin
                contentWidth: availableWidth
                ColumnLayout {
                    width: parent.width
                    spacing: 14
                    Text { text: "Models and Security"; color: window.textMain; font.pixelSize: 26; font.bold: true }
                    Card {
                        Layout.fillWidth: true
                        implicitHeight: modelControls.implicitHeight + 28
                        ColumnLayout {
                            id: modelControls
                            anchors.fill: parent
                            anchors.margins: 14
                            spacing: 8
                            Text { text: "Object detection and tracking"; color: window.textMain; font.bold: true; font.pixelSize: 17 }
                            Text {
                                text: "Models run locally on " + vision.computeDevice + ". Switching keeps capture and outputs alive while the new model warms up."
                                color: window.textMuted
                                wrapMode: Text.Wrap
                                Layout.fillWidth: true
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                ComboBox {
                                    id: objectModelPicker
                                    model: vision.modelOptions
                                    textRole: "label"
                                    valueRole: "id"
                                    Layout.fillWidth: true
                                }
                                Button { text: "Rescan"; onClicked: vision.refreshModels() }
                                Button {
                                    text: "Use model"
                                    enabled: objectModelPicker.count > 0
                                    onClicked: vision.setObjectModel(objectModelPicker.currentValue)
                                }
                            }
                            Text { text: "Active: " + vision.modelSummary; color: window.textMuted }
                        }
                    }
                    Card {
                        Layout.fillWidth: true
                        implicitHeight: 150
                        Column {
                            anchors.fill: parent
                            anchors.margins: 14
                            spacing: 8
                            Text { text: "Authorization locked"; color: window.warning; font.bold: true; font.pixelSize: 17 }
                            Text {
                                width: parent.width
                                text: "Liveness, RGB and IR pairing, physical-device trust, calibrated verification, and the attack suite must pass before security authorization can be enabled."
                                color: window.textMuted
                                wrapMode: Text.Wrap
                            }
                            Text { text: "Current gesture events always carry security_authorized=false."; color: window.textMuted }
                        }
                    }
                }
            }
        }
    }
}
