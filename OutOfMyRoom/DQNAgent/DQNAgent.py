"""
Code edited from:
https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/

Agent that interacts directly with the VBA emulator of the starting room in Pokemon Blue/Red.
"""

import random
from collections import deque

import numpy as np
from ahk import AHK
from keras.models import Sequential, load_model


from keras.layers import Dense
from keras.optimizers import Adam

ACTION_SPACE_SIZE = 4
REPLAY_MEMORY_SIZE = 50_000
MODEL_NAME = "Yoyo"
DISCOUNT = 0.99
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)


class DQNAgent:

    def __init__(self):
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Dense(64, input_shape=(2,)))
        model.add(Dense(64))

        model.add(Dense(ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def load_model(self, model_path):
        self.model = load_model(model_path)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        input_state = np.array(state).reshape(-1, len(state))
        print('Input:', input_state)
        prediction = self.model.predict(input_state)[0]
        print('Prediction:', prediction)
        return prediction


class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 4
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def __init__(self):
        self.ahk = AHK()
        self.location_memory = set()
        self.episode_step = 0

    def reset(self):
        self.episode_step = 0
        self.location_memory = set()
        self.ahk.run_script(open('ahk_scripts/get_state.ahk').read())
        pos_x = open('states/AX.txt').read()
        pos_y = open('states/AY.txt').read()
        observation = (float(pos_x), float(pos_y))
        map = open('states/AMap.txt').read()
        print('Current map id:', map)
        self.location_memory.add(observation)
        self.ahk.run_script(open('ahk_scripts/reset.ahk').read())
        return observation

    def step(self, current_state, action):
        self.episode_step += 1
        self.ahk.run_script(f'action := {action}\n' + open('ahk_scripts/step.ahk').read())
        pos_x = open('states/BX.txt').read()
        pos_y = open('states/BY.txt').read()
        new_observation = (float(pos_x), float(pos_y))
        map_id = int(open('states/BMap.txt').read())
        print('Map id after action:', map_id)
        if map_id == 25 or self.episode_step >= 200:
            reward = 0
            done = True
        elif new_observation == current_state:
            reward = -1.0
            done = False
        elif new_observation not in self.location_memory:
            reward = 0.3
            self.location_memory.add(new_observation)
            done = False
        else:
            done = False
            reward = -0.2

        return new_observation, reward, done
