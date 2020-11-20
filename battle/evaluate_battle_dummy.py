## Code edited from: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/ ###

import os
import sys

import numpy as np
import random
import time
from tqdm import tqdm

from DQNAgent_dummy import *


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
agent.load_model("C:\\Users\\darth\\PycharmProjects\\pokemonBot\\battle\\models_battle\\episode_270_reward_7577.13_time__1605779108.model")
env = BlobEnv(N_BATTLES, 0)
#env.create_battles('battles_dummy_10000_eval.pickle')
# env.load_battles(r'battles_1000_eval.pickle')
env.load_battles(r'battles_dummy_10000_eval.pickle')
current_state = env.reset()

# Run battles

win = 0.0
draw = 0.0
lost = 0.0
fail_move = 0.0

for battle in env.battles:
    current_state = battle
    outcome = 'ok'
    while battle[0].current_hp > 0 and battle[1].current_hp > 0 and outcome != 'fail_move':
        action = np.argmax(agent.get_qs(battle))
        # print_state(battle, action)
        new_state, reward, done, outcome = env.step_real_battle(battle, action)

    if battle[0].current_hp <= 0:
        lost += 1.0
    elif battle[1].current_hp <= 0:
        win += 1.0
    elif outcome == 'fail_move':
        fail_move += 1.0
    print(win)
    print(lost)


print('Fail moves:', fail_move)
print('Win:', win)
print('Lost:', lost)
print('Draw:', draw)
print('Win rate:', (win / (win + draw + lost) * 100))