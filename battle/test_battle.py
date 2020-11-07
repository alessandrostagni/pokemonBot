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
EPISODES = 6
N_BATTLES = 1

# For more repetitive results
random.seed(4)
np.random.seed(4)

# Instantiate battles
env = BlobEnv(N_BATTLES)
env.create_battles(r'battles_test.pickle')
env.load_battles(r'battles_test.pickle')
current_state = env.reset()

# Run battles
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    # agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset flag and start iterating until episode ends
    done = False
    action = 0
    print('Action:', action)
    print('Before fighting:')
    print_state(current_state, action)

    new_state, reward, done, outcome = env.step(current_state, action)

    print('New state:', new_state)
    print('Reward: ', reward)
    print('Done: ', done)
    print('Step: ', step)
    print('After fighting:')
    print_state(new_state, action)
    print('-------')

    if outcome == 'win':
        current_state[0].reset()
        current_state[1].reset()
        print('WIN')
    elif outcome == 'lost':
        current_state[0].reset()
        current_state[1].reset()
        print('LOST')
    elif outcome == 'draw':
        current_state[0].reset()
        current_state[1].reset()
        print('DRAW')
    elif outcome == 'fail_move':
        current_state[0].reset()
        current_state[1].reset()
        print('FAIL_MOVE')

    current_state = new_state
    if done:
        break