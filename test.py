from ahk import AHK

ahk = AHK()

# Test reset AHK script

#ahk.run_script(open('ahk_scripts/reset.ahk').read())

# Test step AHK script
action = 1
ahk.run_script(f'action := {action}\n' + open('OutOfMyRoom/ahk_scripts/step.ahk').read())

# Test get_state AHK script

#ahk.run_script(open('ahk_scripts/get_state.ahk').read())
