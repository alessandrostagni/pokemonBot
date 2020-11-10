## Code edited from: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/ ###

import os
import sys

import numpy as np
import random
import time
from tqdm import tqdm

from DQNAgent_battle import *


def print_state(current_state, action):
    print(f"""
                == == == == == == == == =
                Pokemon: {current_state[0].name}
                == == == == == == == == =
                Level: {current_state[0].level}
                Types: {current_state[0].types}
                HP: {current_state[0].current_hp}
                Speed: {current_state[0].speed}
                Attack: {current_state[0].attack}
                Defense: {current_state[0].defense}
                Sp.Attack: {current_state[0].special_attack}
                Sp.Defense: {current_state[0].special_defense}
                == == =
                Moves
                == == =
                {[(move.name, move.current_pp, move.pp) for move in current_state[0].moves]}
            """)
    print(f"""
                == == == == == == == == =
                Pokemon: {current_state[1].name}
                == == == == == == == == =
                Level: {current_state[1].level}
                Types: {current_state[1].types}
                HP: {current_state[1].current_hp}
                Speed: {current_state[1].speed}
                Attack: {current_state[1].attack}
                Defense: {current_state[1].defense}
                Sp.Attack: {current_state[1].special_attack}
                Sp.Defense: {current_state[1].special_defense}
                == == =
                Moves
                == == =
                {[(move.name, move.current_pp, move.pp) for move in current_state[1].moves]}
            """)


# Environment settings

# For more repetitive results

# Instantiate battles
N_BATTLES = 10000

agent = DQNAgent()
agent.load_model("C:\\Users\\darth\\PycharmProjects\\pokemonBot\\models_saved\\battle\\Yoyo__24557.64max_24557.64avg_24557.64min__1604790963.model")
env = BlobEnv(N_BATTLES)
env.load_battles(r'battles.pickle')
current_state = env.reset()

# Run battles

win = 0.0
draw = 0.0
lost = 0.0
fail_move = 0.0

end = False

while not end:

    # Update tensorboard step every episode
    # agent.tensorboard.step = episode

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        action = 0
        '''
        print('Action:', action)
        print('Before fighting:')
        print_state(current_state, action)
        '''
        action = np.argmax(agent.get_qs(current_state))

        new_state, reward, done, outcome = env.step(current_state, action)
        '''
        print('New state:', new_state)
        print('Reward: ', reward)
        print('Done: ', done)
        print('After fighting:')
        print_state(new_state, action)
        print('-------')
        '''
        if outcome == 'win':
            if env.battle_index + 1 < len(env.battles):
                env.battle_index += 1
            else:
                end = True
            current_state[0].reset()
            current_state[1].reset()
            win += 1.0
            current_state = env.battles[env.battle_index]
            print('WIN')
        elif outcome == 'lost':
            if env.battle_index + 1 < len(env.battles):
                env.battle_index += 1
            else:
                end = True
            current_state[0].reset()
            current_state[1].reset()
            lost += 1.0
            current_state = env.battles[env.battle_index]
            print('LOST')
        elif outcome == 'draw':
            if env.battle_index + 1 < len(env.battles):
                env.battle_index += 1
            else:
                end = True
            current_state[0].reset()
            current_state[1].reset()
            draw += 1.0
            current_state = env.battles[env.battle_index]
            print('DRAW')
        elif outcome == 'fail_move':
            if env.battle_index + 1 < len(env.battles):
                env.battle_index += 1
            else:
                end = True
            current_state[0].reset()
            current_state[1].reset()
            fail_move += 1.0
            current_state = env.battles[env.battle_index]
            print('FAIL_MOVE')
        else:
            current_state = new_state
    if env.battle_index > 1000:
        end = True
print('Fail moves:', fail_move)
print('Win:', win)
print('Lost:', lost)
print('Draw:', draw)
print('Win rate:', (win / (win + draw + lost) * 100))