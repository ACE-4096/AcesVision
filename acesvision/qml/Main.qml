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
    // Low enough that the app is usable on an 800x600 projector and in a tall
    // half-screen tile. It used to refuse to go below 980x640, which made two
    // of the five window sizes this layout is held to unreachable — the app
    // simply could not be put in the shapes it was breaking at.
    minimumWidth: 560
    minimumHeight: 420
    visible: true
    title: "AcesVision"
    color: "#0e1014"
    Material.theme: Material.Dark
    Material.accent: accent

    property color panel: "#171a20"
    property color panelSunken: "#12141a"
    property color border: "#303641"
    property color textMain: "#eef1f5"
    property color textMuted: "#929aa8"
    property color accent: "#8c6cff"
    property color good: "#4fd18b"
    property color warning: "#f5b942"

    property int currentPage: 0
    // Which panel the tooling dock under the feed is showing. Separate from the
    // page so navigating away and back does not lose the operator's place.
    property int currentTool: 0
    // Advanced controls remain available, but start collapsed so their panel
    // never taxes the live video just to change a source or start recording.
    property bool controlsExpanded: false
    readonly property int pageCount: 4
    readonly property int toolCount: 5
    readonly property int pageMargin: 20

    // A narrow window collapses the rail automatically. At usable widths the
    // operator can still collapse it deliberately, which hands more space to
    // the Live feed without hiding navigation.
    readonly property bool wideLayout: width >= 1100
    property bool railExpanded: true
    readonly property bool showRailLabels: wideLayout && railExpanded
    readonly property int railWidth: showRailLabels ? 176 : 60

    component Card: Rectangle {
        color: window.panel
        border.color: window.border
        border.width: 1
        radius: 10
    }

    // Icons drawn from primitives rather than a font. The rail has to stay
    // legible on a machine that has none of the icon fonts installed, and a
    // missing glyph renders as a box — which is worse than no icon at all.
    component NavIcon: Item {
        id: glyph
        required property string kind
        property color tone: window.textMuted
        implicitWidth: 20
        implicitHeight: 20
        clip: true

        // Live: a screen with a capture dot in it.
        Item {
            anchors.fill: parent
            visible: glyph.kind === "feed"
            Rectangle {
                anchors.centerIn: parent
                width: 20; height: 15; radius: 3
                color: "transparent"
                border.width: 1.6
                border.color: glyph.tone
            }
            Rectangle {
                anchors.centerIn: parent
                width: 6; height: 6; radius: 3
                color: glyph.tone
            }
        }
        // Rules: a short list.
        Column {
            anchors.centerIn: parent
            visible: glyph.kind === "rules"
            spacing: 4
            Rectangle { width: 18; height: 2; radius: 1; color: glyph.tone }
            Rectangle { width: 12; height: 2; radius: 1; color: glyph.tone }
            Rectangle { width: 18; height: 2; radius: 1; color: glyph.tone }
        }
        // People: head and shoulders, the shoulders cut by the icon box.
        Item {
            anchors.fill: parent
            visible: glyph.kind === "person"
            Rectangle {
                x: 6; y: 1; width: 8; height: 8; radius: 4
                color: "transparent"; border.width: 1.6; border.color: glyph.tone
            }
            Rectangle {
                x: 2; y: 12; width: 16; height: 14; radius: 7
                color: "transparent"; border.width: 1.6; border.color: glyph.tone
            }
        }
        // Models and security: a chip.
        Item {
            anchors.fill: parent
            visible: glyph.kind === "chip"
            Rectangle {
                anchors.centerIn: parent
                width: 15; height: 15; radius: 3
                color: "transparent"; border.width: 1.6; border.color: glyph.tone
            }
            Rectangle {
                anchors.centerIn: parent
                width: 6; height: 6; radius: 1.5; color: glyph.tone
            }
        }
    }

    component NavButton: AbstractButton {
        id: navControl
        required property int page
        required property string glyph
        Layout.fillWidth: true
        Layout.preferredHeight: 42
        checkable: true
        checked: window.currentPage === page
        onClicked: window.currentPage = page
        hoverEnabled: true
        // The label is the only thing the compact rail drops, so it has to come
        // back somewhere. Hovering an icon says which page it is.
        ToolTip.visible: !window.showRailLabels && navControl.hovered
        ToolTip.text: navControl.text
        ToolTip.delay: 400
        background: Rectangle {
            color: navControl.checked ? "#29243d"
                                      : (navControl.hovered ? "#20242b" : "transparent")
            radius: 8
            border.color: navControl.checked ? "#5b4c91" : "transparent"
        }
        contentItem: Item {
            NavIcon {
                id: navGlyph
                kind: navControl.glyph
                tone: navControl.checked ? window.textMain : window.textMuted
                anchors.verticalCenter: parent.verticalCenter
                anchors.left: parent.left
                // Centred when there is no label beside it, indented when
                // there is. Computed rather than re-anchored, so nothing has to
                // assign `undefined` to an anchor at a breakpoint.
                anchors.leftMargin: window.showRailLabels
                                    ? 14 : Math.max(0, (parent.width - width) / 2)
            }
            Text {
                visible: window.showRailLabels
                text: navControl.text
                color: navControl.checked ? window.textMain : window.textMuted
                font.pixelSize: 14
                font.weight: navControl.checked ? Font.DemiBold : Font.Normal
                elide: Text.ElideRight
                verticalAlignment: Text.AlignVCenter
                anchors.left: navGlyph.right
                anchors.leftMargin: 12
                anchors.right: parent.right
                anchors.rightMargin: 10
                anchors.verticalCenter: parent.verticalCenter
            }
        }
    }

    // The dock's tab strip. A TabBar lays its tabs out in one unbreakable row
    // and clips the last of them in a narrow window; these live in a Flow and
    // wrap onto a second line instead, so no tool is ever off the edge.
    component ToolTab: AbstractButton {
        id: tab
        required property int slot
        checkable: true
        checked: window.currentTool === slot
        onClicked: {
            window.currentTool = slot
            window.controlsExpanded = true
        }
        hoverEnabled: true
        leftPadding: 14
        rightPadding: 14
        implicitHeight: 32
        background: Rectangle {
            radius: 16
            color: tab.checked ? "#2c2547" : (tab.hovered ? "#232830" : "transparent")
            border.width: 1
            border.color: tab.checked ? window.accent : window.border
        }
        contentItem: Text {
            text: tab.text
            color: tab.checked ? window.textMain : window.textMuted
            font.pixelSize: 13
            font.weight: tab.checked ? Font.DemiBold : Font.Normal
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
    }

    component Metric: Column {
        id: metric
        required property string label
        required property string value
        // A long value (a model name, a gesture line) must not be allowed to
        // set the width of the whole strip.
        property int maximumWidth: 220
        spacing: 3
        Text {
            width: Math.min(implicitWidth, metric.maximumWidth)
            text: metric.label
            color: window.textMuted
            font.pixelSize: 11
            elide: Text.ElideRight
        }
        Text {
            width: Math.min(implicitWidth, metric.maximumWidth)
            text: metric.value
            color: window.textMain
            font.pixelSize: 14
            font.bold: true
            elide: Text.ElideRight
        }
    }

    component SectionTitle: Text {
        color: window.textMain
        font.bold: true
        font.pixelSize: 15
        Layout.fillWidth: true
    }

    component Hint: Text {
        color: window.textMuted
        wrapMode: Text.Wrap
        font.pixelSize: 12
        Layout.fillWidth: true
    }

    // A tool panel: scrolls inside the dock rather than pushing the dock taller
    // than the window. Nothing in the dock can be made unreachable by a short
    // window; the worst case is that it scrolls.
    component ToolPanel: ScrollView {
        clip: true
        rightPadding: 6
        contentWidth: availableWidth
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // ---------------------------------------------------------------
        // App bar. The detailed live telemetry belongs beside the video, not
        // in a second dashboard above every page. Keep only the application
        // identity and runtime health here so the camera retains its height.
        // ---------------------------------------------------------------
        Rectangle {
            objectName: "statusBar"
            Layout.fillWidth: true
            implicitHeight: 42
            color: "#121419"
            border.color: window.border
            border.width: 1

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 12
                anchors.rightMargin: 12
                spacing: 10

                Text {
                    text: "AcesVision"
                    color: window.textMain
                    font.pixelSize: 16
                    font.bold: true
                }

                Rectangle {
                    width: stateRow.width + 20
                    height: 24
                    radius: 12
                    color: vision.status === "live" ? "#13301f" : "#302713"
                    border.width: 1
                    border.color: vision.status === "live" ? "#2f6b48" : "#6b5a2f"
                    Row {
                        id: stateRow
                        anchors.centerIn: parent
                        spacing: 7
                        Rectangle {
                            width: 8; height: 8; radius: 4
                            anchors.verticalCenter: parent.verticalCenter
                            color: vision.status === "live" ? window.good : window.warning
                        }
                        Text {
                            text: vision.status === "live" ? "Runtime live" : vision.status
                            color: vision.status === "live" ? window.good : window.warning
                            font.pixelSize: 12
                            font.bold: true
                        }
                    }
                }

                Item { Layout.fillWidth: true }
                Text {
                    text: "Local vision runtime"
                    color: window.textMuted
                    font.pixelSize: 12
                    visible: window.width >= 780
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            // -----------------------------------------------------------
            // Nav rail. Collapses to icons under the breakpoint instead of
            // holding 220 px of words hostage in a narrow window.
            // -----------------------------------------------------------
            Rectangle {
                objectName: "navRail"
                Layout.preferredWidth: window.railWidth
                Layout.fillHeight: true
                color: "#121419"
                border.color: window.border
                border.width: 1

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 8
                    anchors.topMargin: 12
                    spacing: 6

                    ToolButton {
                        objectName: "navCollapse"
                        Layout.fillWidth: true
                        Layout.preferredHeight: 30
                        text: window.showRailLabels ? "Collapse" : ">"
                        onClicked: window.railExpanded = !window.railExpanded
                        ToolTip.visible: hovered
                        ToolTip.text: window.showRailLabels
                                      ? "Collapse navigation"
                                      : "Expand navigation"
                    }
                    NavButton { objectName: "nav0"; text: "Live"; glyph: "feed"; page: 0 }
                    NavButton { objectName: "nav1"; text: "Rules"; glyph: "rules"; page: 1 }
                    NavButton { objectName: "nav2"; text: "People"; glyph: "person"; page: 2 }
                    NavButton { objectName: "nav3"; text: "Models"; glyph: "chip"; page: 3 }
                    Item { Layout.fillHeight: true }
                }
            }

            StackLayout {
                objectName: "pageStack"
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: window.currentPage

                // =======================================================
                // Live — the feed, and every routine control under it.
                // =======================================================
                ScrollView {
                    id: livePage
                    objectName: "page0"
                    clip: true
                    // Page gutter lives on the ScrollView, not on the
                    // ColumnLayout. A ColumnLayout sized by `width: parent.width`
                    // has no anchors, so anchors.margins on it was inert and
                    // every page ran its titles and cards into the window edge.
                    padding: window.pageMargin
                    contentWidth: availableWidth
                    ColumnLayout {
                        id: liveBody
                        objectName: "page0Body"
                        width: parent.width
                        // Fill the viewport when there is room, so the feed can
                        // take the slack; grow past it and scroll when there is
                        // not, so nothing is ever cut off instead.
                        height: Math.max(implicitHeight, livePage.availableHeight)
                        spacing: 12

                        // A compact session bar. The full dashboard used to
                        // consume more vertical room than the live controls it
                        // replaced; camera work needs the picture first.
                        Card {
                            id: dashboardCard
                            objectName: "dashboardCard"
                            Layout.fillWidth: true
                            implicitHeight: 48
                            RowLayout {
                                id: dashboardBody
                                anchors.fill: parent
                                anchors.margins: 8
                                spacing: 8
                                Text {
                                    text: vision.status === "live" ? "LIVE" : vision.status.toUpperCase()
                                    color: vision.status === "live" ? window.good : window.warning
                                    font.bold: true
                                    font.pixelSize: 12
                                }
                                Text {
                                    text: vision.sourceLabel
                                    color: window.textMain
                                    elide: Text.ElideRight
                                    Layout.preferredWidth: Math.min(240, dashboardCard.width * 0.27)
                                    Layout.minimumWidth: 90
                                }
                                Text {
                                    text: vision.captureFps.toFixed(0) + " / "
                                          + vision.inferenceFps.toFixed(0) + " FPS"
                                    color: window.textMuted
                                    font.pixelSize: 12
                                }
                                Text {
                                    text: vision.objectCount + " obj · " + vision.poseCount
                                          + " body · " + vision.faceCount + " face · "
                                          + vision.gestureCount + " gesture"
                                    color: window.textMuted
                                    font.pixelSize: 12
                                    visible: dashboardCard.width >= 760
                                }
                                Item { Layout.fillWidth: true }
                                Button {
                                    objectName: "quickSourceButton"
                                    text: "Source"
                                    onClicked: sourcePopup.open()
                                }
                                Button {
                                    objectName: "dashboardRecordButton"
                                    text: vision.recordingEnabled ? "Stop" : "Record"
                                    onClicked: vision.setRecordingEnabled(!vision.recordingEnabled)
                                }
                                Button {
                                    objectName: "quickAudioButton"
                                    text: "Mic"
                                    visible: dashboardCard.width >= 700
                                    onClicked: audioPopup.open()
                                }
                                Button {
                                    text: vision.workoutEnabled
                                          ? vision.workoutReps + " reps" : "Workout"
                                    onClicked: workoutPopup.open()
                                }
                                Button {
                                    text: window.controlsExpanded ? "Hide controls" : "Controls"
                                    onClicked: window.controlsExpanded = !window.controlsExpanded
                                }
                            }

                            // Source changes are a common live action, so this
                            // overlay never participates in the vertical split.
                            Popup {
                                id: sourcePopup
                                parent: Overlay.overlay
                                x: Math.max(12, (window.width - width) / 2)
                                y: Math.max(12, (window.height - height) / 2)
                                width: Math.min(560, window.width - 24)
                                padding: 14
                                modal: false
                                closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
                                background: Rectangle {
                                    color: window.panel
                                    border.color: window.border
                                    border.width: 1
                                    radius: 10
                                }
                                contentItem: ColumnLayout {
                                    spacing: 10
                                    SectionTitle { text: "Switch source" }
                                    Text {
                                        text: "Current: " + vision.sourceLabel
                                        color: window.textMuted
                                        elide: Text.ElideRight
                                        Layout.fillWidth: true
                                    }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        Text {
                                            text: "Orientation"
                                            color: window.textMuted
                                            font.pixelSize: 12
                                        }
                                        ComboBox {
                                            id: quickRotationPicker
                                            objectName: "quickRotationPicker"
                                            model: vision.rotationOptions
                                            textRole: "label"
                                            currentIndex: vision.rotationIndex
                                            Layout.fillWidth: true
                                            enabled: !vision.recordingEnabled
                                            onActivated: vision.setRotation(
                                                vision.rotationOptions[currentIndex].id)
                                        }
                                    }
                                    Hint {
                                        text: vision.recordingEnabled
                                              ? "Stop this recording before changing orientation."
                                              : "Rotation happens before detection, overlays, and recording — choose 90° for vertical footage."
                                    }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        ComboBox {
                                            id: quickWebcamPicker
                                            model: vision.webcams
                                            textRole: "label"
                                            valueRole: "index"
                                            Layout.fillWidth: true
                                            displayText: count > 0 ? currentText
                                                                   : "No physical cameras found"
                                        }
                                        Button { text: "Rescan"; onClicked: vision.refreshWebcams() }
                                        Button {
                                            text: "Use camera"
                                            enabled: quickWebcamPicker.count > 0
                                            onClicked: {
                                                vision.useWebcamIndex(quickWebcamPicker.currentValue)
                                                sourcePopup.close()
                                            }
                                        }
                                    }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        TextField {
                                            id: quickDroidUrl
                                            placeholderText: "http://PHONE_IP:4747/video"
                                            Layout.fillWidth: true
                                        }
                                        Button {
                                            text: "Use DroidCam"
                                            onClicked: {
                                                vision.useDroidCam(quickDroidUrl.text)
                                                sourcePopup.close()
                                            }
                                        }
                                    }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        Button {
                                            text: vision.droidScanActive ? "Scanning..." : "Scan DroidCam"
                                            enabled: !vision.droidScanActive
                                            onClicked: vision.scanDroidCams()
                                        }
                                        Item { Layout.fillWidth: true }
                                        Button {
                                            text: "Advanced source controls"
                                            onClicked: {
                                                sourcePopup.close()
                                                window.currentTool = 2
                                                window.controlsExpanded = true
                                            }
                                        }
                                        Button {
                                            text: "Recording audio"
                                            onClicked: {
                                                sourcePopup.close()
                                                audioPopup.open()
                                            }
                                        }
                                    }
                                    Hint { text: vision.droidScanStatus }
                                }
                            }

                            Popup {
                                id: audioPopup
                                parent: Overlay.overlay
                                x: Math.max(12, (window.width - width) / 2)
                                y: Math.max(12, (window.height - height) / 2)
                                width: Math.min(500, window.width - 24)
                                padding: 14
                                modal: false
                                closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
                                background: Rectangle {
                                    color: window.panel
                                    border.color: window.border
                                    border.width: 1
                                    radius: 10
                                }
                                contentItem: ColumnLayout {
                                    spacing: 10
                                    SectionTitle { text: "Recording audio" }
                                    ComboBox {
                                        objectName: "quickAudioSourcePicker"
                                        Layout.fillWidth: true
                                        model: vision.audioSources
                                        textRole: "label"
                                        currentIndex: vision.audioSourceIndex
                                        onActivated: vision.setRecordingAudioSource(
                                            vision.audioSources[currentIndex].id)
                                    }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        enabled: vision.recordingMicrophoneSelected
                                        Text {
                                            text: "Mic gain"
                                            color: window.textMuted
                                            font.pixelSize: 12
                                        }
                                        Slider {
                                            objectName: "quickAudioGain"
                                            from: 0; to: 150; stepSize: 1
                                            value: vision.recordingAudioGain
                                            Layout.fillWidth: true
                                            onMoved: vision.setRecordingAudioGain(value)
                                        }
                                        Text {
                                            text: vision.recordingAudioGain + "%"
                                            color: window.textMain
                                            font.bold: true
                                            Layout.preferredWidth: 42
                                        }
                                    }
                                    ProgressBar {
                                        objectName: "quickAudioMeter"
                                        Layout.fillWidth: true
                                        from: 0; to: 1; value: vision.recordingAudioLevel
                                        enabled: vision.recordingMicrophoneSelected
                                    }
                                    Text {
                                        text: vision.recordingMicrophoneSelected
                                              ? vision.recordingAudioLevelDb.toFixed(1) + " dBFS"
                                              : "Microphone gain is available only for microphone sources."
                                        color: window.textMain
                                        font.bold: vision.recordingMicrophoneSelected
                                    }
                                    Hint { text: vision.recordingAudioMeterStatus }
                                }
                            }

                            Popup {
                                id: workoutPopup
                                parent: Overlay.overlay
                                x: Math.max(12, (window.width - width) / 2)
                                y: Math.max(12, (window.height - height) / 2)
                                width: Math.min(500, window.width - 24)
                                padding: 14
                                modal: false
                                closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
                                background: Rectangle {
                                    color: window.panel
                                    border.color: window.border
                                    border.width: 1
                                    radius: 10
                                }
                                contentItem: ColumnLayout {
                                    spacing: 10
                                    SectionTitle { text: "Workout analysis" }
                                    Switch {
                                        objectName: "workoutEnabled"
                                        text: "Enable rep counting"
                                        checked: vision.workoutEnabled
                                        onToggled: vision.setWorkoutEnabled(checked)
                                    }
                                    ComboBox {
                                        objectName: "workoutExercise"
                                        Layout.fillWidth: true
                                        model: vision.workoutExercises
                                        textRole: "label"
                                        currentIndex: vision.workoutExerciseIndex
                                        onActivated: vision.setWorkoutExercise(
                                            vision.workoutExercises[currentIndex].id)
                                    }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        Button { text: "Reset reps"; onClicked: vision.resetWorkout() }
                                        Text {
                                            text: vision.workoutReps + " reps · "
                                                  + vision.workoutPhase + " · "
                                                  + vision.workoutAngle.toFixed(0) + "°"
                                            color: window.textMain
                                            font.bold: true
                                        }
                                        Item { Layout.fillWidth: true }
                                    }
                                    ProgressBar {
                                        Layout.fillWidth: true
                                        from: 0; to: 1; value: vision.workoutProgress
                                    }
                                    Hint { text: vision.workoutFeedback }
                                    Hint { text: vision.workoutFilter }
                                }
                            }
                        }

                        // The guard, and it is not decoration. Without a face
                        // box a shush is not merely dropped: MediaPipe labels
                        // the same hand Pointing_Up, which the example
                        // automations bind to `ledctl next-theme`. It used to
                        // live on a page the operator had to already be on to
                        // see it; it is a page-level banner here, above the
                        // dock, so no tab selection can hide it.
                        Card {
                            objectName: "shushWarningCard"
                            Layout.fillWidth: true
                            visible: vision.shushDegraded
                            color: "#2a2214"
                            border.color: "#7a6031"
                            implicitHeight: shushRow.implicitHeight + 24
                            RowLayout {
                                id: shushRow
                                anchors.fill: parent
                                anchors.margins: 12
                                spacing: 10
                                Rectangle {
                                    Layout.preferredWidth: 4
                                    Layout.fillHeight: true
                                    radius: 2
                                    color: window.warning
                                }
                                Text {
                                    objectName: "shushWarningText"
                                    text: vision.shushWarning
                                    color: window.warning
                                    wrapMode: Text.Wrap
                                    Layout.fillWidth: true
                                }
                            }
                        }

                        // Persists until another failure supersedes it or the
                        // operator dismisses it. It used to be erased by the
                        // next 100 ms refresh tick, so a failed action flashed
                        // for under a frame.
                        Card {
                            objectName: "errorCard"
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

                        // The divider belongs around the feed, not inside a
                        // settings tab: drag it to give live video or the
                        // controls the room the current task needs.
                        SplitView {
                            id: liveSplit
                            objectName: "liveSplit"
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            orientation: Qt.Vertical
                            handle: Rectangle {
                                implicitHeight: 10
                                color: "transparent"
                                Rectangle {
                                    anchors.centerIn: parent
                                    width: Math.min(72, parent.width * 0.22)
                                    height: 3
                                    radius: 2
                                    color: window.border
                                }
                            }

                        // ---- the feed itself ----------------------------
                        Card {
                            objectName: "feedCard"
                            // Leave enough room for an actual video frame even
                            // when the session dashboard and tool dock are both
                            // visible in a compact window.
                            SplitView.minimumHeight: 170
                            SplitView.fillHeight: true
                            // The 16:9 feed is the initial split. Once dragged,
                            // SplitView owns this preference rather than snapping
                            // the video back when the dock refreshes.
                            SplitView.preferredHeight:
                                Math.max(170, Math.min(width * 9 / 16,
                                                       livePage.availableHeight * 0.54))
                            color: "#050608"
                            clip: true
                            // Decode the next JPEG behind the current frame,
                            // then swap visibility only once it is ready. A
                            // single Image with a new URL clears its texture
                            // while decoding, which was the whole-preview
                            // flicker seen even with a stable camera source.
                            Item {
                                id: previewStack
                                anchors.fill: parent
                                anchors.margins: 1
                                property bool showA: true
                                property bool loading: false
                                property bool hasFrame: false
                                property bool failed: false
                                property string pendingSource: ""
                                property int retryTick: 0

                                function requestFrame(url) {
                                    if (!url.length) return
                                    pendingSource = url
                                    loadPending()
                                }
                                function loadPending() {
                                    if (loading || !pendingSource.length) return
                                    var target = showA ? previewB : previewA
                                    loading = true
                                    target.loading = true
                                    target.source = pendingSource
                                    pendingSource = ""
                                }
                                function decoded(image) {
                                    if (!image.loading) return
                                    image.loading = false
                                    loading = false
                                    showA = image === previewA
                                    hasFrame = true
                                    failed = false
                                    loadPending()
                                }
                                function decodeFailed(image) {
                                    if (!image.loading) return
                                    image.loading = false
                                    loading = false
                                    if (!hasFrame) failed = true
                                    previewRetry.restart()
                                }

                                Image {
                                    id: previewA
                                    objectName: "previewImage"
                                    property bool loading: false
                                    anchors.fill: parent
                                    visible: previewStack.showA
                                    cache: false
                                    asynchronous: true
                                    fillMode: Image.PreserveAspectFit
                                    onStatusChanged: {
                                        if (status === Image.Ready) previewStack.decoded(previewA)
                                        else if (status === Image.Error) previewStack.decodeFailed(previewA)
                                    }
                                }
                                Image {
                                    id: previewB
                                    property bool loading: false
                                    anchors.fill: parent
                                    visible: !previewStack.showA
                                    cache: false
                                    asynchronous: true
                                    fillMode: Image.PreserveAspectFit
                                    onStatusChanged: {
                                        if (status === Image.Ready) previewStack.decoded(previewB)
                                        else if (status === Image.Error) previewStack.decodeFailed(previewB)
                                    }
                                }
                                Connections {
                                    target: vision
                                    function onPreviewChanged() {
                                        if (vision.sequence > 0)
                                            previewStack.requestFrame(
                                                vision.previewSource + "&retry="
                                                + previewStack.retryTick)
                                    }
                                }
                                Component.onCompleted: {
                                    if (vision.sequence > 0)
                                        requestFrame(vision.previewSource + "&retry=" + retryTick)
                                }
                            }
                            // The preview server answers 503 until the first frame
                            // exists. Retry quietly instead of leaving a dead tile.
                            Timer {
                                id: previewRetry
                                interval: 1000
                                onTriggered: {
                                    previewStack.retryTick++
                                    previewStack.requestFrame(vision.previewSource + "&retry="
                                                              + previewStack.retryTick)
                                }
                            }
                            // A stalled feed keeps the last good frame on screen and
                            // reads as live video. Scrim it and say what happened.
                            Rectangle {
                                anchors.fill: parent
                                anchors.margins: 1
                                visible: previewStack.failed || vision.previewStale
                                color: "#b0050608"
                            }
                            Text {
                                anchors.centerIn: parent
                                width: parent.width - 40
                                horizontalAlignment: Text.AlignHCenter
                                wrapMode: Text.Wrap
                                visible: previewStack.failed || vision.previewStale
                                         || vision.status !== "live"
                                color: previewStack.failed || vision.previewStale
                                       ? window.warning : window.textMuted
                                font.pixelSize: 18
                                text: previewStack.failed
                                      ? "Preview feed unavailable — retrying"
                                      : vision.status === "reconnecting"
                                        ? "Waiting for camera"
                                        : vision.previewStale
                                          ? "Frame is stale — this is not live video"
                                          : vision.status === "live"
                                            ? ""
                                            : "Starting vision runtime"
                            }
                            // Badges, top left, over the video. Which overlay
                            // profile is baked into what OBS is receiving is a
                            // question the operator asks while looking at the
                            // picture, not while looking at a settings page.
                            Row {
                                anchors.left: parent.left
                                anchors.top: parent.top
                                anchors.margins: 12
                                spacing: 8
                                Rectangle {
                                    visible: vision.obsEnabled
                                    width: obsBadge.width + 16; height: 22; radius: 11
                                    color: "#cc13301f"
                                    border.width: 1; border.color: "#2f6b48"
                                    Text {
                                        id: obsBadge
                                        anchors.centerIn: parent
                                        text: "OBS"
                                        color: window.good
                                        font.pixelSize: 11
                                        font.bold: true
                                    }
                                }
                                Rectangle {
                                    width: overlayBadge.width + 16; height: 22; radius: 11
                                    color: "#cc12141a"
                                    border.width: 1; border.color: window.border
                                    Text {
                                        id: overlayBadge
                                        anchors.centerIn: parent
                                        text: vision.overlayProfile
                                        color: window.textMuted
                                        font.pixelSize: 11
                                    }
                                }
                            }
                        }

                        Item {
                            SplitView.minimumHeight: 230
                            SplitView.fillHeight: true

                            ColumnLayout {
                                anchors.fill: parent
                                spacing: 12

                        // ---- live measurements --------------------------
                        Card {
                            objectName: "metricStrip"
                            Layout.fillWidth: true
                            implicitHeight: metricFlow.implicitHeight + 24
                            color: window.panelSunken
                            Flow {
                                id: metricFlow
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.top: parent.top
                                anchors.margins: 12
                                spacing: 26
                                Metric {
                                    label: "Capture"
                                    value: vision.captureFps.toFixed(1) + " FPS"
                                }
                                Metric {
                                    label: "Inference"
                                    value: vision.inferenceFps.toFixed(1) + " FPS"
                                }
                                Metric {
                                    label: "Cycle (latest)"
                                    value: vision.latestInferenceMs.toFixed(0) + " ms"
                                }
                                Metric {
                                    label: "Cycle (avg 30)"
                                    value: vision.inferenceMs.toFixed(0) + " ms"
                                }
                                Metric {
                                    label: "Object model"
                                    value: vision.modelInferenceMs.toFixed(1) + " ms"
                                }
                                Metric {
                                    label: "Last gesture"
                                    value: vision.lastGesture
                                }
                            }
                        }

                        // ---- the tooling dock ---------------------------
                        Card {
                            objectName: "toolDock"
                            Layout.fillWidth: true
                            Layout.fillHeight: window.controlsExpanded
                            Layout.minimumHeight: window.controlsExpanded ? 170 : 44
                            Layout.maximumHeight: window.controlsExpanded ? 16777215 : 44
                            // Keyed to the space the page actually has, not to
                            // the window: the status bar wraps in a narrow window
                            // and takes a second row with it, and a dock sized off
                            // the window height ignored that and pushed the page
                            // into a scroll it did not need.
                            Layout.preferredHeight:
                                window.controlsExpanded
                                ? Math.max(170, Math.min(340,
                                                         livePage.availableHeight * 0.28))
                                : 44

                            ColumnLayout {
                                id: dockBody
                                anchors.fill: parent
                                anchors.margins: window.controlsExpanded ? 14 : 6
                                spacing: window.controlsExpanded ? 10 : 6

                                Flow {
                                    objectName: "toolTabs"
                                    Layout.fillWidth: true
                                    spacing: 6
                                    ToolTab { objectName: "tool0"; text: "Image"; slot: 0 }
                                    ToolTab { objectName: "tool1"; text: "Perception"; slot: 1 }
                                    ToolTab { objectName: "tool2"; text: "Source"; slot: 2 }
                                    ToolTab { objectName: "tool3"; text: "Overlay"; slot: 3 }
                                    ToolTab { objectName: "tool4"; text: "Output"; slot: 4 }
                                }

                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.preferredHeight: 1
                                    color: window.border
                                    visible: window.controlsExpanded
                                }

                                StackLayout {
                                    objectName: "toolStack"
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    currentIndex: window.currentTool
                                    visible: window.controlsExpanded

                                    // ---- 0: Image -----------------------
                                    ToolPanel {
                                        objectName: "toolPanel0"
                                        ColumnLayout {
                                            id: liveTuning
                                            objectName: "liveTuning"
                                            width: parent.width
                                            spacing: 8

                                            function applyTuning() {
                                                vision.setCameraTuning(
                                                    autoExposure.checked,
                                                    exposureControl.value,
                                                    brightnessControl.value,
                                                    contrastControl.value,
                                                    gammaControl.value)
                                            }

                                            RowLayout {
                                                Layout.fillWidth: true
                                                spacing: 16
                                                Switch {
                                                    id: autoExposure
                                                    objectName: "autoExposureSwitch"
                                                    text: "Automatic exposure"
                                                    checked: true
                                                    enabled: vision.manualExposureSupported
                                                    onToggled: liveTuning.applyTuning()
                                                }
                                                Item { Layout.fillWidth: true }
                                                Text {
                                                    text: "Luma " + vision.imageMean.toFixed(1)
                                                    color: window.textMuted
                                                }
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
                                            }
                                            // The backend can force auto back on when it
                                            // catches a camera going black in manual mode,
                                            // so mirror it.
                                            Connections {
                                                target: vision
                                                function onCameraTuningChanged() {
                                                    autoExposure.checked = vision.autoExposure
                                                }
                                            }
                                            Hint {
                                                text: vision.manualExposureSupported
                                                      ? "Automatic exposure is usually brighter. Manual gives a steadier frame rate."
                                                      : "This camera requires automatic exposure."
                                            }
                                            Hint {
                                                visible: vision.exposureNotice.length > 0
                                                text: vision.exposureNotice
                                                color: window.warning
                                            }

                                            GridLayout {
                                                Layout.fillWidth: true
                                                // Two field columns when there is room
                                                // for them, one when there is not — and
                                                // `columns` counts cells, so each field
                                                // costs two of them.
                                                columns: liveTuning.width >= 660 ? 4 : 2
                                                columnSpacing: 18
                                                rowSpacing: 8

                                                Text {
                                                    text: "Exposure"
                                                    color: window.textMuted
                                                    Layout.preferredWidth: 82
                                                }
                                                SpinBox {
                                                    id: exposureControl
                                                    from: 3; to: 333; value: 166; editable: true
                                                    enabled: vision.manualExposureSupported
                                                             && !autoExposure.checked
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 150
                                                    Layout.maximumWidth: 240
                                                    onValueModified: liveTuning.applyTuning()
                                                }
                                                Text {
                                                    text: "Brightness"
                                                    color: window.textMuted
                                                    Layout.preferredWidth: 82
                                                }
                                                SpinBox {
                                                    id: brightnessControl
                                                    from: -64; to: 64; value: 0; editable: true
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 150
                                                    Layout.maximumWidth: 240
                                                    onValueModified: liveTuning.applyTuning()
                                                }
                                                Text {
                                                    text: "Contrast"
                                                    color: window.textMuted
                                                    Layout.preferredWidth: 82
                                                }
                                                SpinBox {
                                                    id: contrastControl
                                                    from: 0; to: 95; value: 0; editable: true
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 150
                                                    Layout.maximumWidth: 240
                                                    onValueModified: liveTuning.applyTuning()
                                                }
                                                Text {
                                                    text: "Gamma"
                                                    color: window.textMuted
                                                    Layout.preferredWidth: 82
                                                }
                                                SpinBox {
                                                    id: gammaControl
                                                    from: 100; to: 300; value: 100; editable: true
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 150
                                                    Layout.maximumWidth: 240
                                                    onValueModified: liveTuning.applyTuning()
                                                }
                                            }

                                            Hint {
                                                visible: vision.imageWarning.length > 0
                                                text: vision.imageWarning
                                                color: window.warning
                                            }
                                            Item { Layout.fillHeight: true }
                                        }
                                    }

                                    // ---- 1: Perception ------------------
                                    ToolPanel {
                                        objectName: "toolPanel1"
                                        ColumnLayout {
                                            objectName: "stagePanel"
                                            width: parent.width
                                            spacing: 8

                                            Text {
                                                objectName: "cycleCost"
                                                text: "Cycle: " + vision.latestInferenceMs.toFixed(0)
                                                      + " ms latest, " + vision.inferenceMs.toFixed(0)
                                                      + " ms average over 30, "
                                                      + vision.inferenceFps.toFixed(1) + " FPS  ·  "
                                                      + vision.modelSummary
                                                color: window.textMuted
                                                wrapMode: Text.Wrap
                                                Layout.fillWidth: true
                                            }
                                            Hint {
                                                text: "All three stages run in sequence on one inference cycle, so the cycle cost is their sum. Switching a stage off or slowing it down buys frame rate back."
                                            }

                                            Repeater {
                                                // Driven off the fixed stage id list, never
                                                // off stageStats. A Repeater rebuilds every
                                                // delegate when its model changes, and
                                                // stageStats changes on the 100 ms poll —
                                                // binding to it would tear down and recreate
                                                // these switches and sliders ten times a
                                                // second and leave them undraggable.
                                                model: vision.stageIds
                                                delegate: Rectangle {
                                                    id: stageCard
                                                    required property int index
                                                    required property string modelData
                                                    readonly property var stats:
                                                        index < vision.stageStats.length
                                                        ? vision.stageStats[index] : null
                                                    readonly property bool degraded:
                                                        stats !== null && stats.warning.length > 0
                                                    objectName: "stageCard" + index
                                                    Layout.fillWidth: true
                                                    implicitHeight: stageBody.implicitHeight + 24
                                                    radius: 8
                                                    color: window.panelSunken
                                                    border.width: 1
                                                    border.color: stageCard.degraded
                                                                  ? window.warning : window.border
                                                    ColumnLayout {
                                                        id: stageBody
                                                        anchors.fill: parent
                                                        anchors.margins: 12
                                                        spacing: 6
                                                        RowLayout {
                                                            Layout.fillWidth: true
                                                            Switch {
                                                                text: stageCard.stats
                                                                      ? stageCard.stats.label : ""
                                                                checked: stageCard.stats
                                                                         ? stageCard.stats.enabled : false
                                                                onToggled: vision.setStageEnabled(
                                                                    stageCard.modelData, checked)
                                                            }
                                                            Item { Layout.fillWidth: true }
                                                            Text {
                                                                text: stageCard.stats
                                                                      ? stageCard.stats.ms.toFixed(1)
                                                                        + " ms  ·  "
                                                                        + stageCard.stats.hz.toFixed(1)
                                                                        + " Hz"
                                                                        + (stageCard.stats.refreshed
                                                                           ? "" : "  (idle)")
                                                                      : ""
                                                                color: window.textMain
                                                                font.bold: true
                                                            }
                                                        }
                                                        RowLayout {
                                                            Layout.fillWidth: true
                                                            enabled: stageCard.stats
                                                                     ? stageCard.stats.enabled : false
                                                            spacing: 10
                                                            Text {
                                                                text: stageCard.stats
                                                                      ? stageCard.stats.rateLabel : ""
                                                                color: window.textMuted
                                                                elide: Text.ElideRight
                                                                Layout.preferredWidth:
                                                                    Math.min(150, stageCard.width * 0.32)
                                                            }
                                                            Slider {
                                                                id: stageRate
                                                                objectName: "stageRate" + stageCard.index
                                                                from: stageCard.stats
                                                                      ? stageCard.stats.rateMin : 0
                                                                to: stageCard.stats
                                                                    ? stageCard.stats.rateMax : 1
                                                                stepSize: stageCard.stats
                                                                          ? stageCard.stats.rateStep : 1
                                                                snapMode: Slider.SnapAlways
                                                                value: stageCard.stats
                                                                       ? stageCard.stats.rate : 0
                                                                Layout.fillWidth: true
                                                                Layout.minimumWidth: 90
                                                                // onMoved, not onValueChanged: value
                                                                // is bound to the knob the backend
                                                                // publishes, so reacting to every
                                                                // change would echo the backend's own
                                                                // updates straight back at it.
                                                                onMoved: vision.setStageRate(
                                                                    stageCard.modelData, value)
                                                            }
                                                            Text {
                                                                text: stageCard.stats
                                                                      ? stageRate.value.toFixed(
                                                                            stageCard.stats.rateStep < 1
                                                                            ? 1 : 0)
                                                                        + stageCard.stats.rateSuffix
                                                                      : ""
                                                                color: window.textMain
                                                                horizontalAlignment: Text.AlignRight
                                                                Layout.preferredWidth: 74
                                                            }
                                                        }
                                                        Text {
                                                            text: stageCard.stats
                                                                  ? stageCard.stats.detail : ""
                                                            color: window.textMuted
                                                            font.pixelSize: 12
                                                            wrapMode: Text.Wrap
                                                            Layout.fillWidth: true
                                                        }
                                                        Text {
                                                            visible: stageCard.degraded
                                                            text: stageCard.stats
                                                                  ? stageCard.stats.warning : ""
                                                            color: window.warning
                                                            wrapMode: Text.Wrap
                                                            Layout.fillWidth: true
                                                        }
                                                    }
                                                }
                                            }
                                            Item { Layout.fillHeight: true }
                                        }
                                    }

                                    // ---- 2: Source ----------------------
                                    ToolPanel {
                                        objectName: "toolPanel2"
                                        ColumnLayout {
                                            objectName: "sourcePanel"
                                            width: parent.width
                                            spacing: 8

                                            SectionTitle { text: "Physical webcam" }
                                            RowLayout {
                                                Layout.fillWidth: true
                                                spacing: 8
                                                ComboBox {
                                                    id: webcamPicker
                                                    objectName: "webcamPicker"
                                                    model: vision.webcams
                                                    textRole: "label"
                                                    valueRole: "index"
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 130
                                                    displayText: count > 0 ? currentText
                                                                           : "No physical cameras found"
                                                }
                                                Button {
                                                    text: "Rescan"
                                                    onClicked: vision.refreshWebcams()
                                                }
                                                Button {
                                                    text: "Use"
                                                    enabled: webcamPicker.count > 0
                                                    onClicked: vision.useWebcamIndex(
                                                        webcamPicker.currentValue)
                                                }
                                            }
                                            Hint {
                                                text: "Named devices are read from Linux without opening or locking the camera. Metadata-only nodes are hidden."
                                            }

                                            SectionTitle { text: "Orientation" }
                                            RowLayout {
                                                Layout.fillWidth: true
                                                Text {
                                                    text: "Rotate clockwise"
                                                    color: window.textMuted
                                                    font.pixelSize: 12
                                                }
                                                ComboBox {
                                                    id: rotationPicker
                                                    objectName: "rotationPicker"
                                                    model: vision.rotationOptions
                                                    textRole: "label"
                                                    currentIndex: vision.rotationIndex
                                                    Layout.fillWidth: true
                                                    enabled: !vision.recordingEnabled
                                                    onActivated: vision.setRotation(
                                                        vision.rotationOptions[currentIndex].id)
                                                }
                                            }
                                            Hint {
                                                text: vision.recordingEnabled
                                                      ? "Stop this recording before changing orientation."
                                                      : "Applies before local perception, overlays and MP4 recording. Use 90° for vertical media."
                                            }

                                            Rectangle {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 6
                                                Layout.preferredHeight: 1
                                                color: window.border
                                            }

                                            SectionTitle { text: "DroidCam over the network" }
                                            RowLayout {
                                                Layout.fillWidth: true
                                                spacing: 8
                                                ComboBox {
                                                    id: droidPicker
                                                    objectName: "droidPicker"
                                                    model: vision.droidCams
                                                    textRole: "label"
                                                    valueRole: "url"
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 130
                                                    displayText: count > 0 ? currentText
                                                                           : "No discovered DroidCam devices"
                                                }
                                                Button {
                                                    text: vision.droidScanActive
                                                          ? "Scanning..." : "Scan"
                                                    enabled: !vision.droidScanActive
                                                    onClicked: vision.scanDroidCams()
                                                }
                                                Button {
                                                    text: "Use"
                                                    enabled: droidPicker.count > 0
                                                    onClicked: vision.useDroidCam(
                                                        droidPicker.currentValue)
                                                }
                                            }
                                            RowLayout {
                                                Layout.fillWidth: true
                                                spacing: 8
                                                TextField {
                                                    id: droidUrl
                                                    objectName: "droidUrlField"
                                                    placeholderText: "http://PHONE_IP:4747/video"
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 130
                                                }
                                                Button {
                                                    text: "Use URL"
                                                    onClicked: vision.useDroidCam(droidUrl.text)
                                                }
                                            }
                                            Hint { text: vision.droidScanStatus }
                                            // This program opens sockets on the operator's
                                            // own home network. Which network that is must
                                            // never be something they have to read the
                                            // source to find out, so it stays on the surface
                                            // that has the Scan button — not on a page they
                                            // would have to go looking for.
                                            Hint {
                                                text: "Scan checks only your own wired or wireless LAN, one /24, on DroidCam port 4747. VPN, container and virtual-machine networks are never scanned. It runs only when you click Scan."
                                            }
                                            Hint {
                                                objectName: "scanPlanText"
                                                text: "Will scan: " + vision.scanPlanTarget
                                            }
                                            Hint {
                                                text: "Network feeds are not trusted for privileged authorization."
                                                color: window.warning
                                            }
                                            Item { Layout.fillHeight: true }
                                        }
                                    }

                                    // ---- 3: Overlay ---------------------
                                    ToolPanel {
                                        objectName: "toolPanel3"
                                        ColumnLayout {
                                            id: overlayPanel
                                            objectName: "overlayPanel"
                                            width: parent.width
                                            spacing: 8

                                            SectionTitle { text: "Profile" }
                                            Flow {
                                                objectName: "overlayProfiles"
                                                Layout.fillWidth: true
                                                spacing: 6
                                                Repeater {
                                                    // The custom profile gets a chip as soon
                                                    // as one has been applied. Without it,
                                                    // "Apply custom" selected a profile no
                                                    // chip knew about and every chip read as
                                                    // unselected.
                                                    model: {
                                                        var profiles = ["clean", "minimal",
                                                                        "broadcast", "security"]
                                                        if (vision.customOverlayReady)
                                                            profiles.push("custom")
                                                        return profiles
                                                    }
                                                    delegate: AbstractButton {
                                                        id: profileChip
                                                        required property string modelData
                                                        readonly property bool active:
                                                            vision.overlayProfile === modelData
                                                        text: modelData
                                                        leftPadding: 14
                                                        rightPadding: 14
                                                        implicitHeight: 32
                                                        hoverEnabled: true
                                                        onClicked: vision.setOverlayProfile(modelData)
                                                        background: Rectangle {
                                                            radius: 16
                                                            color: profileChip.active ? "#2c2547"
                                                                   : (profileChip.hovered
                                                                      ? "#232830" : "transparent")
                                                            border.width: 1
                                                            border.color: profileChip.active
                                                                          ? window.accent : window.border
                                                        }
                                                        contentItem: Text {
                                                            text: profileChip.text
                                                            color: profileChip.active
                                                                   ? window.textMain : window.textMuted
                                                            font.pixelSize: 13
                                                            horizontalAlignment: Text.AlignHCenter
                                                            verticalAlignment: Text.AlignVCenter
                                                        }
                                                    }
                                                }
                                            }
                                            Hint {
                                                text: "Profiles render independently from the shared raw frame. Clean is the raw video; security adds scores and security state."
                                            }

                                            Rectangle {
                                                Layout.fillWidth: true
                                                Layout.topMargin: 6
                                                Layout.preferredHeight: 1
                                                color: window.border
                                            }

                                            SectionTitle { text: "Custom box style" }
                                            Flow {
                                                Layout.fillWidth: true
                                                spacing: 10
                                                Switch { id: showObjects; text: "Object boxes"; checked: true }
                                                Switch { id: showFaces; text: "Face boxes"; checked: true }
                                                Switch { id: showGestures; text: "Gesture boxes"; checked: true }
                                                Switch { id: showLandmarks; text: "Hand joints"; checked: true }
                                                Switch { id: showPose; text: "Body joints"; checked: true }
                                            }
                                            RowLayout {
                                                Layout.fillWidth: true
                                                spacing: 10
                                                Text {
                                                    text: "Line width"
                                                    color: window.textMuted
                                                    Layout.preferredWidth: 76
                                                }
                                                Slider {
                                                    id: lineWidth
                                                    from: 1; to: 8; stepSize: 1; value: 2
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 90
                                                }
                                                Text {
                                                    text: Math.round(lineWidth.value) + " px"
                                                    color: window.textMain
                                                    horizontalAlignment: Text.AlignRight
                                                    Layout.preferredWidth: 46
                                                }
                                            }
                                            RowLayout {
                                                Layout.fillWidth: true
                                                spacing: 10
                                                Text {
                                                    text: "Label size"
                                                    color: window.textMuted
                                                    Layout.preferredWidth: 76
                                                }
                                                Slider {
                                                    id: fontScale
                                                    from: 0.3; to: 2.0; stepSize: 0.1; value: 0.6
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 90
                                                }
                                                Text {
                                                    text: fontScale.value.toFixed(1)
                                                    color: window.textMain
                                                    horizontalAlignment: Text.AlignRight
                                                    Layout.preferredWidth: 46
                                                }
                                            }
                                            GridLayout {
                                                Layout.fillWidth: true
                                                columns: overlayPanel.width >= 520 ? 4 : 2
                                                columnSpacing: 8
                                                rowSpacing: 8
                                                TextField {
                                                    id: objectColour
                                                    text: "#3cAAff"; placeholderText: "Objects"
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 90
                                                }
                                                TextField {
                                                    id: knownColour
                                                    text: "#00c800"; placeholderText: "Known"
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 90
                                                }
                                                TextField {
                                                    id: unknownColour
                                                    text: "#dc2828"; placeholderText: "Unknown"
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 90
                                                }
                                                TextField {
                                                    id: gestureColour
                                                    text: "#1ea0e6"; placeholderText: "Gesture"
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 90
                                                }
                                            }
                                            RowLayout {
                                                Layout.fillWidth: true
                                                Item { Layout.fillWidth: true }
                                                Button {
                                                    text: "Apply custom"
                                                    onClicked: vision.applyOverlayStyle(
                                                        showObjects.checked, showFaces.checked,
                                                        showGestures.checked,
                                                        Math.round(lineWidth.value), fontScale.value,
                                                        objectColour.text, knownColour.text,
                                                        unknownColour.text, gestureColour.text,
                                                        showLandmarks.checked, showPose.checked)
                                                }
                                            }
                                            Hint {
                                                text: "Hand joints use MediaPipe's 21-point skeleton; Body joints use MediaPipe Pose's 33-point posture skeleton. Trails, zones and saved profiles are not available yet."
                                            }
                                            Item { Layout.fillHeight: true }
                                        }
                                    }

                                    // ---- 4: Output ----------------------
                                    ToolPanel {
                                        objectName: "toolPanel4"
                                        ColumnLayout {
                                            objectName: "outputPanel"
                                            width: parent.width
                                            spacing: 8

                                            SectionTitle { text: "Destinations" }
                                            Flow {
                                                objectName: "outputSwitches"
                                                Layout.fillWidth: true
                                                spacing: 18
                                                Switch {
                                                    objectName: "obsSwitch"
                                                    text: "OBS virtual camera"
                                                    checked: vision.obsEnabled
                                                    onToggled: vision.setObsEnabled(checked)
                                                }
                                                Switch {
                                                    objectName: "eventsSwitch"
                                                    text: "Gesture events"
                                                    checked: vision.eventsEnabled
                                                    onToggled: vision.setEventsEnabled(checked)
                                                }
                                                Switch {
                                                    objectName: "recordSwitch"
                                                    text: "Record video"
                                                    checked: vision.recordingEnabled
                                                    onToggled: vision.setRecordingEnabled(checked)
                                                }
                                            }
                                            Hint {
                                                text: "Current input: " + vision.sourceLabel
                                                      + " · " + vision.captureFps.toFixed(1) + " FPS"
                                            }
                                            Hint {
                                                text: vision.recordingStatus
                                            }
                                            SectionTitle { text: "Recording rate" }
                                            ComboBox {
                                                objectName: "recordingRatePicker"
                                                Layout.fillWidth: true
                                                model: vision.recordingRateOptions
                                                textRole: "label"
                                                currentIndex: vision.recordingRateIndex
                                                onActivated: vision.setRecordingRate(
                                                    vision.recordingRateOptions[currentIndex].id)
                                            }
                                            Hint {
                                                text: "Next recording: " + vision.recordingFps
                                                      + " FPS. Auto preserves a DroidCam feed at 60 FPS; choose a fixed rate only when you need a fixed delivery timeline."
                                            }
                                            SectionTitle { text: "Recording audio" }
                                            ComboBox {
                                                objectName: "audioSourcePicker"
                                                Layout.fillWidth: true
                                                model: vision.audioSources
                                                textRole: "label"
                                                currentIndex: vision.audioSourceIndex
                                                onActivated: vision.setRecordingAudioSource(
                                                    vision.audioSources[currentIndex].id)
                                            }
                                            RowLayout {
                                                Layout.fillWidth: true
                                                enabled: vision.recordingMicrophoneSelected
                                                Text {
                                                    text: "Mic gain"
                                                    color: window.textMuted
                                                    font.pixelSize: 12
                                                }
                                                Slider {
                                                    objectName: "audioGain"
                                                    from: 0; to: 150; stepSize: 1
                                                    value: vision.recordingAudioGain
                                                    Layout.fillWidth: true
                                                    onMoved: vision.setRecordingAudioGain(value)
                                                }
                                                Text {
                                                    text: vision.recordingAudioGain + "%"
                                                    color: window.textMain
                                                    font.bold: true
                                                    Layout.preferredWidth: 42
                                                }
                                            }
                                            ProgressBar {
                                                objectName: "audioMeter"
                                                Layout.fillWidth: true
                                                from: 0; to: 1; value: vision.recordingAudioLevel
                                                enabled: vision.recordingMicrophoneSelected
                                            }
                                            Text {
                                                text: vision.recordingMicrophoneSelected
                                                      ? vision.recordingAudioLevelDb.toFixed(1) + " dBFS"
                                                      : "Microphone gain is available only for microphone sources."
                                                color: window.textMain
                                                font.bold: vision.recordingMicrophoneSelected
                                            }
                                            Hint { text: vision.recordingAudioMeterStatus }
                                            Hint {
                                                text: "Audio is off by default. Choose a Microphone for narration or an explicit System audio monitor for desktop sound. Aim for spoken peaks around −12 to −6 dBFS; system-audio monitors are read-only here."
                                            }
                                            Hint {
                                                text: "Record video writes an MP4 and JSON sidecar outside this checkout. It uses this live pipeline, never a second camera."
                                            }
                                            Hint { text: vision.lastGesture }
                                            Hint { text: vision.lastDecision }
                                            Item { Layout.fillHeight: true }
                                        }
                                    }
                                }
                            }
                        }
                            }
                        }
                    }
                    }
                }

                // =======================================================
                // Gestures and Rules
                // =======================================================
                ScrollView {
                    objectName: "page1"
                    clip: true
                    padding: window.pageMargin
                    contentWidth: availableWidth
                    ColumnLayout {
                        objectName: "page1Body"
                        width: parent.width
                        spacing: 14
                        Text {
                            text: "Gestures and Rules"
                            color: window.textMain
                            font.pixelSize: 26
                            font.bold: true
                        }
                        Card {
                            Layout.fillWidth: true
                            implicitHeight: eventColumn.implicitHeight + 28
                            ColumnLayout {
                                id: eventColumn
                                anchors.fill: parent
                                anchors.margins: 14
                                spacing: 8
                                SectionTitle { text: "Event output" }
                                Hint { text: vision.lastGesture }
                                Hint { text: vision.lastDecision }
                                Hint {
                                    text: "Connectors: " + (vision.executableConnectors.length
                                          ? vision.executableConnectors.join(", ")
                                          : "none wired")
                                }
                                Hint {
                                    text: "Rules are dry-run until armed one at a time. Failures show here and in the error banner on Live."
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
                                SectionTitle { text: "Add rule (starts dry-run)" }
                                GridLayout {
                                    Layout.fillWidth: true
                                    columns: ruleForm.width >= 620 ? 4 : 2
                                    columnSpacing: 10
                                    rowSpacing: 10
                                    // Catalog-backed, not free text: a mistyped
                                    // gesture or actor produced a rule that could
                                    // never fire and never said so.
                                    ComboBox {
                                        id: ruleGesture
                                        model: vision.gestureNames
                                        Layout.fillWidth: true
                                        Layout.minimumWidth: 110
                                    }
                                    ComboBox {
                                        id: ruleActor
                                        model: vision.actorNames
                                        Layout.fillWidth: true
                                        Layout.minimumWidth: 110
                                    }
                                    ComboBox {
                                        id: ruleConnector
                                        model: vision.connectorNames
                                        Layout.fillWidth: true
                                        Layout.minimumWidth: 110
                                        onCurrentTextChanged: {
                                            ruleAction.model = vision.connectorActions(currentText)
                                            ruleAction.currentIndex = 0
                                        }
                                    }
                                    ComboBox {
                                        id: ruleAction
                                        model: vision.connectorActions(ruleConnector.currentText)
                                        Layout.fillWidth: true
                                        Layout.minimumWidth: 110
                                    }
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    Item { Layout.fillWidth: true }
                                    Button {
                                        text: "Add rule"
                                        enabled: ruleGesture.currentText.length > 0
                                                 && ruleConnector.currentText.length > 0
                                                 && ruleAction.currentText.length > 0
                                        onClicked: vision.addRule(ruleGesture.currentText,
                                                                  ruleActor.currentText,
                                                                  ruleConnector.currentText,
                                                                  ruleAction.currentText)
                                    }
                                }
                                Hint {
                                    Layout.fillWidth: true
                                    text: "For clean video, choose Overlay / toggle_clean, add the rule, then turn off Dry run. It hides boxes and labels; the next matching gesture restores your previous overlay."
                                }
                            }
                        }

                        Text {
                            visible: vision.rules.length === 0
                            text: "No rules configured. Add one above — every new rule starts in dry-run until you arm it."
                            color: window.textMuted
                            wrapMode: Text.Wrap
                            Layout.fillWidth: true
                        }

                        Repeater {
                            model: vision.rules
                            delegate: Card {
                                required property var modelData
                                Layout.fillWidth: true
                                implicitHeight: ruleRow.implicitHeight + 24
                                RowLayout {
                                    id: ruleRow
                                    anchors.fill: parent
                                    anchors.margins: 12
                                    spacing: 10
                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 4
                                        Text {
                                            text: modelData.gesture + "  ->  "
                                                  + modelData.connector + "." + modelData.action
                                            color: window.textMain
                                            font.bold: true
                                            elide: Text.ElideRight
                                            Layout.fillWidth: true
                                        }
                                        Text {
                                            text: "Actor: " + modelData.actor
                                                + " | Risk: " + modelData.risk
                                                + " | " + (modelData.dryRun ? "Dry-run" : "ARMED")
                                                + (modelData.executable ? "" : " | no executor")
                                            color: modelData.dryRun ? window.textMuted : window.warning
                                            wrapMode: Text.Wrap
                                            Layout.fillWidth: true
                                        }
                                    }
                                    // Per-rule arming. Nothing else flips dry_run.
                                    Button {
                                        text: modelData.dryRun ? "Arm" : "Disarm"
                                        enabled: modelData.executable || !modelData.dryRun
                                        onClicked: vision.setRuleDryRun(modelData.id,
                                                                        !modelData.dryRun)
                                    }
                                    Button {
                                        text: "Remove"
                                        onClicked: vision.removeRule(modelData.id)
                                    }
                                }
                            }
                        }
                    }
                }

                // =======================================================
                // People
                // =======================================================
                ScrollView {
                    objectName: "page2"
                    clip: true
                    padding: window.pageMargin
                    contentWidth: availableWidth
                    ColumnLayout {
                        objectName: "page2Body"
                        width: parent.width
                        spacing: 14
                        Text {
                            text: "People"
                            color: window.textMain
                            font.pixelSize: 26
                            font.bold: true
                        }
                        Card {
                            Layout.fillWidth: true
                            implicitHeight: enrolNotice.implicitHeight + 28
                            color: "#2a2214"
                            border.color: "#7a6031"
                            ColumnLayout {
                                id: enrolNotice
                                anchors.fill: parent
                                anchors.margins: 14
                                spacing: 8
                                SectionTitle {
                                    text: "Secure enrolment is not active yet"
                                    color: window.warning
                                }
                                Hint {
                                    text: "Enrolled faces are used for labelling only. AcesVision will not present them as verified authentication."
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
                                    SectionTitle { text: "Enrolled identities" }
                                    Button { text: "Rescan"; onClicked: vision.refreshActors() }
                                }
                                Hint {
                                    // Re-read while the app runs, so somebody enrolled
                                    // now can be picked as a rule actor without a restart.
                                    text: vision.actorNames.length > 1
                                          ? vision.actorNames.filter(
                                                function (name) { return name !== "*" }).join(", ")
                                          : "Nobody is enrolled yet. Rules can still target anyone with the * actor."
                                }
                            }
                        }
                    }
                }

                // =======================================================
                // Models and Security — the rare, consequential controls.
                // Switching detection model warms a new model up; it does not
                // belong one click from the video the way exposure does.
                // =======================================================
                ScrollView {
                    objectName: "page3"
                    clip: true
                    padding: window.pageMargin
                    contentWidth: availableWidth
                    ColumnLayout {
                        objectName: "page3Body"
                        width: parent.width
                        spacing: 14
                        Text {
                            text: "Models and Security"
                            color: window.textMain
                            font.pixelSize: 26
                            font.bold: true
                        }
                        Card {
                            Layout.fillWidth: true
                            implicitHeight: modelControls.implicitHeight + 28
                            ColumnLayout {
                                id: modelControls
                                anchors.fill: parent
                                anchors.margins: 14
                                spacing: 8
                                SectionTitle { text: "Object detection and tracking" }
                                Hint {
                                    text: "Models run locally on " + vision.computeDevice
                                          + ". Switching keeps capture and outputs alive while the new model warms up."
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8
                                    ComboBox {
                                        id: objectModelPicker
                                        objectName: "objectModelPicker"
                                        model: vision.modelOptions
                                        textRole: "label"
                                        valueRole: "id"
                                        Layout.fillWidth: true
                                        Layout.minimumWidth: 130
                                    }
                                    Button { text: "Rescan"; onClicked: vision.refreshModels() }
                                    Button {
                                        text: "Use model"
                                        enabled: objectModelPicker.count > 0
                                        onClicked: vision.setObjectModel(objectModelPicker.currentValue)
                                    }
                                }
                                Hint { text: "Active: " + vision.modelSummary }
                            }
                        }

                        Card {
                            Layout.fillWidth: true
                            implicitHeight: lockedColumn.implicitHeight + 28
                            ColumnLayout {
                                id: lockedColumn
                                anchors.fill: parent
                                anchors.margins: 14
                                spacing: 8
                                SectionTitle {
                                    text: "Authorization locked"
                                    color: window.warning
                                }
                                Hint {
                                    text: "Liveness, RGB and IR pairing, physical-device trust, calibrated verification, and the attack suite must pass before security authorization can be enabled."
                                }
                                Hint {
                                    text: "Current gesture events always carry security_authorized=false."
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
