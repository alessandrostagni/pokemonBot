; An action definition is expected to be placed here.
ConvertBase(InputBase, OutputBase, nptr)    ; Base 2 - 36
{
    static u := A_IsUnicode ? "_wcstoui64" : "_strtoui64"
    static v := A_IsUnicode ? "_i64tow"    : "_i64toa"
    VarSetCapacity(s, 66, 0)
    value := DllCall("msvcrt.dll\" u, "Str", nptr, "UInt", 0, "UInt", InputBase, "CDECL Int64")
    DllCall("msvcrt.dll\" v, "Int64", value, "Str", s, "UInt", OutputBase, "CDECL")
    return s
}


CloseWindow() {
	Send, {LAlt Down}
	Send, {F4}
	Send, {LAlt Up}
}

OpenMemoryViewer() {
	Send, {LAlt Down}
	Send, {t}
	Send, {LAlt Up}
	Send, {Down}
	Send, {Down}
	Send, {Down}
	Send, {Down}
	Send, {Enter}
}

Save(label, location) {
	Send, {Enter}
	Send, {Raw} %location%
	Send, {Tab}
	Send, {1}
	Send, {Tab}
	Send, {Enter}
	Sleep, 800
	Send, {Raw} %label%
	Send, {Enter}
}

SaveState(label) {
    OpenMemoryViewer()
    WinActivate, Memory viewer ahk_class #32770
    Sleep, 800
    Send, {Shift Down}
    Send, {Tab}
    Send, {Tab}
    Send, {Shift Up}

    WinActivate, Enter address and size ahk_class #32770
    Sleep, 400
    Save(label . "Y", "D361")

    Sleep, 400
    Save(label . "X", "D362")

    Sleep, 400
    Save(label . "Map", "D35E")

    Sleep, 400

    file := FileOpen("C:\Users\darth\Gameboy\states\" . label . "X.DMP", "r")
    X := file.read()
    file.close()
    file := FileOpen("C:\Users\darth\Gameboy\states\" . label . "Y.DMP", "r")
    Y := file.read()
    file.close()
    file := FileOpen("C:\Users\darth\Gameboy\states\" . label . "Map.DMP", "r")
    Map := file.read()
    file.close()

    X := ConvertBase(10, 16, Asc(X)) ; Convert String -> Decimal -> Hex
    Y := ConvertBase(10, 16, Asc(Y)) ; Convert String -> Decimal -> Hex
    map := ConvertBase(10, 16, Asc(map)) ; Convert String -> Decimal -> Hex

    file := FileOpen("C:\Users\darth\PycharmProjects\pokemonBot\states\" . label . "X.txt", "w")
    file.write(X)
    file.close()
    file := FileOpen("C:\Users\darth\PycharmProjects\pokemonBot\states\" . label . "Y.txt", "w")
    file.write(Y)
    file.close()
    file := FileOpen("C:\Users\darth\PycharmProjects\pokemonBot\states\" . label . "Map.txt", "w")
    file.write(map)
    file.close()
    CloseWindow()
}

Move(action) {
    Key := 1
    Switch action {
        Case 1:
            Key = Right
        Case 2:
            Key = Left
        Case 3:
            Key = Down
        Case 4:
            Key = Up
    }
    Send, {%Key% Down}
	Sleep, 50
	Send, {%Key% Up}
}

WinActivate ahk_exe VisualBoyAdvance.exe
Sleep, 100
SaveState("A")
Sleep, 100
WinMinimize ahk_exe VisualBoyAdvance.exe
ExitApp
Return

Escape::
ExitApp
Return