## Code edited from: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/ ###

import os
import random
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from keras.callbacks import TensorBoard

ACTION_SPACE_SIZE = 4
MODEL_NAME = "Yoyo"
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 5000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MAX_STEPS = 200


class DQNAgent:

    def __init__(self):
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        # self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        # self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(2,)))
        model.add(Dense(64))
        model.add(Dense(ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))
        return model

    def load_model(self, model_path):
        self.model = load_model(model_path)

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        X = np.array(X)
        y = np.array(y)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            X, y, batch_size=MINIBATCH_SIZE, verbose=0,
            shuffle=False, callbacks=[TensorBoard(log_dir='logs')] if terminal_state else None
        )

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        input = np.array(state).reshape(-1, len(state))
        print('Input:', input)
        prediction = self.model.predict(input)[0]
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
        self.start_position = (3.0, 6.0)
        self.current_position = self.start_position
        self.location_memory = set()
        self.episode_step = 0
        self.x_limits = (0.0, 7.0)
        self.y_limits = (0.0, 7.0)
        self.blocks = {(3.0, 5.0), (3.0, 4.0), (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),
                       (0.0, 7.0), (0.0, 6.0), (6.0, 7.0), (6.0, 6.0)}
        self.final_state = (7.0, 1.0)

    def reset(self):
        self.episode_step = 0
        self.location_memory = set()
        self.current_position = self.start_position
        self.location_memory.add(self.current_position)
        return self.current_position

    def move(self, current_state, action):
        new_state = current_state
        if action == 0:
            new_pos = (current_state[0] + 1.0, current_state[1])
            if new_pos not in self.blocks and new_pos[0] <= self.x_limits[1]:
                new_state = new_pos
        if action == 1:
            new_pos = (current_state[0] - 1.0, current_state[1])
            if new_pos not in self.blocks and new_pos[0] >= self.x_limits[0]:
                new_state = new_pos
        if action == 2:
            new_pos = (current_state[0], current_state[1] + 1.0)
            if new_pos not in self.blocks and new_pos[1] <= self.y_limits[1]:
                new_state = new_pos
        if action == 3:
            new_pos = (current_state[0], current_state[1] - 1.0)
            if new_pos not in self.blocks and new_pos[1] >= self.y_limits[0]:
                new_state = new_pos
        return new_state

    def step(self, current_state, action):
        self.episode_step += 1
        same = False
        new_observation = self.move(current_state, action)
        if new_observation == self.final_state:
            reward = +100.0
            done = True
        elif new_observation == current_state:
            reward = -100.0
            done = True
        else:
            done = False
            reward = -1.0
        if new_observation == self.final_state:
            print('FOUND!')
        if self.episode_step >= MAX_STEPS:
            done = True
        print(self.location_memory)
        return new_observation, reward, done, same
