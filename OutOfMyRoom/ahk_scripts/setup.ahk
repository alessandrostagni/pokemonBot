OpenMemoryViewer() {
	Send, {LAlt Down}
	Send, {t}
	Send, {LAlt Up}
	Send, {Down}
	Send, {Down}
	Send, {Down}
	Send, {Down}
	Send, {Enter}
	Send, {Shift Down}
    Sleep, 100
    Send, {Tab}
    Sleep, 100
    Send, {Tab}
    Send, {Shift Up}
    Sleep, 100
}
WinActivate VisualBoyAdvance
OpenMemoryViewer()
WinActivate, Memory viewer
Sleep, 100
WinActivate VisualBoyAdvance
Sleep, 100
ExitApp
Return

F3::
ExitApp
Return