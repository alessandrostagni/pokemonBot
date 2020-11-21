from pokemon_simulate import *

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def decode_state(battle):
    pokemon_a = battle[0]
    pokemon_b = battle[1]
    moves_data = []
    for move in pokemon_a.moves:
        if move.max_hits is None:
            move_max_hits = -1
        else:
            move_max_hits = move.max_hits

        if move.accuracy is None:
            move_accuracy = -1
        else:
            move_accuracy = move.accuracy

        if move.power is None:
            move_power = -1
        else:
            move_power = move.power

        moves_data.append((
            move.current_pp, move_max_hits, move_accuracy, move_power
        ))
    for i in range(4 - len(pokemon_a.moves)):
        moves_data.append((-1, -1, -1, -1, -1))

    state = [
        pokemon_a.types[0], pokemon_a.types[1],
        pokemon_a.attack, pokemon_a.defense,
        pokemon_a.special_attack, pokemon_a.special_defense,
        pokemon_a.speed,
        moves_data[0][0],
        moves_data[0][1],
        moves_data[0][2],
        moves_data[0][3],
        moves_data[1][0],
        moves_data[1][1],
        moves_data[1][2],
        moves_data[1][3],
        moves_data[2][0],
        moves_data[2][1],
        moves_data[2][2],
        moves_data[2][3],
        moves_data[3][0],
        moves_data[3][1],
        moves_data[3][2],
        moves_data[3][3],
        pokemon_b.current_hp, pokemon_b.types[0], pokemon_b.types[1]
    ]
    return state


def create_model(ACTION_SPACE_SIZE):
    model = Sequential()
    model.add(Dense(32, input_shape=(26,)))
    model.add(Dense(64))
    model.add(Dense(ACTION_SPACE_SIZE, activation='linear'))
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))
    return model


def check_move_feasibility(pokemon_a, action):
    # Unfeasible action:
    if action >= len(pokemon_a.moves):
        return 'fail_move'
    move = pokemon_a.moves[action]
    print(f'Pokemon a chooses {move.name}')

    # Check there are no moves with more than 0 pp that could be used by the agent
    if move.current_pp == 0:
        for move in pokemon_a.moves:
            if move.current_pp > 0:
                print(f'Finished pp for move {move.name}, can\'t use it!')
                return 'fail_move'
    return 'ok'


def agent_attack(pokemon_a, pokemon_b, action):
    # Assign attacker and defender roles
    attacker = pokemon_a
    defender = pokemon_b
    move = pokemon_a.moves[action]
    apply_move(attacker, defender, move)
    return defender


def check_draw(pokemon_a, pokemon_b):
    draw = True
    for move in pokemon_a.moves:
        if move.current_pp > 0:
            return False
    for move in pokemon_b.moves:
        if move.current_pp > 0:
            return False
    return draw


def bot_attack(pokemon_a, pokemon_b):
    attacker = pokemon_b
    defender = pokemon_a
    move = choose_move(attacker)
    if move is not None:
        print(f'Pokemon b chooses {move.name}')
        apply_move(attacker, defender, move)
    return defender


def decide_first_attacker(pokemon_a, pokemon_b):
    if pokemon_a.speed > pokemon_b.speed:
        return 'a'
    elif pokemon_a.speed < pokemon_b.speed:
        return 'b'
    return np.random.choice(['a', 'b'])