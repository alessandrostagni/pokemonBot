; An action definition is expected to be placed here OR written by a script, see README.
ConvertBase(InputBase, OutputBase, nptr)    ; Base 2 - 36
{
    static u := A_IsUnicode ? "_wcstoui64" : "_strtoui64"
    static v := A_IsUnicode ? "_i64tow"    : "_i64toa"
    VarSetCapacity(s, 66, 0)
    value := DllCall("msvcrt.dll\" u, "Str", nptr, "UInt", 0, "UInt", InputBase, "CDECL Int64")
    DllCall("msvcrt.dll\" v, "Int64", value, "Str", s, "UInt", OutputBase, "CDECL")
    return s
}

OpenMemoryViewer() {
	Send, {LAlt Down}
	Sleep, 100
	Send, {t}
	Sleep, 100
	Send, {LAlt Up}
	Sleep, 100
	Send, {Down}
	Sleep, 100
	Send, {Down}
	Sleep, 100
	Send, {Down}
	Sleep, 100
	Send, {Down}
	Sleep, 100
	Send, {Enter}
}

Save(label, location) {
	Send, {Enter}
	Sleep, 100
	Send, {Raw} %location%
	Sleep, 100
	Send, {Tab}
	Sleep, 100
	Send, {1}
	Sleep, 100
	Send, {Tab}
	Sleep, 100
	Send, {Enter}
	Sleep, 800
	Send, {Raw} %label%
	Send, {Enter}
}

SaveState(label) {
    ;OpenMemoryViewer()
    WinActivate, Memory viewer
    ;Send, {Shift Down}
    ;Sleep, 100
    ;Send, {Tab}
    ;Sleep, 100
    ;Send, {Tab}
    ;Send, {Shift Up}
    Sleep, 100
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

    Sleep, 100
    WinActivate Memory viewer
    ;Sleep, 100
    ;Send, {Esc}
    ;Sleep, 100
    WinActivate VisualBoyAdvance
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
WinActivate VisualBoyAdvance
Sleep, 100
SaveState("A")
Sleep, 100
ExitApp
Return

F3::
ExitApp
Return