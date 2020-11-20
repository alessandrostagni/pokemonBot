def print_state(state):
    print(f"""
                == == == == == == == == =
                Pokemon: {state[0].name}
                == == == == == == == == =
                Level: {state[0].level}
                Types: {state[0].types}
                HP: {state[0].current_hp}
                Speed: {state[0].speed}
                Attack: {state[0].attack}
                Defense: {state[0].defense}
                Sp.Attack: {state[0].special_attack}
                Sp.Defense: {state[0].special_defense}
                == == =
                Moves
                == == =
                {[(move.name, move.current_pp, move.pp) for move in state[0].moves]}
            """)
    print(f"""
                == == == == == == == == =
                Pokemon: {state[1].name}
                == == == == == == == == =
                Level: {state[1].level}
                Types: {state[1].types}
                HP: {state[1].current_hp}
                Speed: {state[1].speed}
                Attack: {state[1].attack}
                Defense: {state[1].defense}
                Sp.Attack: {state[1].special_attack}
                Sp.Defense: {state[1].special_defense}
                == == =
                Moves
                == == =
                {[(move.name, move.current_pp, move.pp) for move in state[1].moves]}
            """)
