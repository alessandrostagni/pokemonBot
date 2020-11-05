## Code edited from: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/ ###

import os
import random
import time

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from keras.callbacks import TensorBoard

from pokemon_simulate import *

np.random.seed(1)

ACTION_SPACE_SIZE = 4
MODEL_NAME = "Yoyo"
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MAX_STEPS = 200

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
DATA_DIR = os.path.join(ROOT_DIR, '../data')
TYPE_MODS = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'type_modifiers.csv')).set_index('attack_type')
VERBOSE = True


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

    def decode_state(self, battle):
        pokemon_a = battle[0]
        pokemon_b = battle[1]
        moves_data = []
        for i in range(len(pokemon_a.moves)):
            moves_data.append((pokemon_a.moves[i].get_id(), pokemon_a.moves[i].current_pp))
        for i in range(4 - len(pokemon_a.moves)):
            moves_data.append((0, 0))

        state = [
            pokemon_a.types[0], pokemon_a.types[1],
            pokemon_a.hp, pokemon_a.attack, pokemon_a.defense,
            pokemon_a.special_attack, pokemon_b.special_defense,
            pokemon_a.speed,
            moves_data[0][0],
            moves_data[1][0],
            moves_data[2][0],
            moves_data[3][0],
            moves_data[0][1],
            moves_data[1][1],
            moves_data[2][1],
            moves_data[3][1],
            pokemon_b.hp, pokemon_b.types[0], pokemon_b.types[1]
        ]
        return state

    def create_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(19,)))
        model.add(Dense(64))
        model.add(Dense(ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))
        return model

    def load_model(self, model_path):
        self.model = load_model(model_path)

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append((
            self.decode_state(transition[0]), transition[1],
            transition[2], self.decode_state(transition[3]),
            transition[4]
        ))

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
        state = self.decode_state(state)
        input = np.array(state).reshape(-1, len(state))
        print('Input:', input)
        prediction = self.model.predict(input)[0]
        print('Prediction:', prediction)
        return prediction


class BlobEnv:

    def __init__(self, n_battles):
        self.n_battles = n_battles
        self.ACTION_SPACE_SIZE = 4
        self.battles = []
        self.same_state = 0
        self.last_state = None
        self.first_attack = True
        self.episode_step = 0
        self.battle_index = 0
        self.reset_index = 0
        self.current_state = None

    def create_battles(self, path):
        base_level = 0
        for i in range(0, self.n_battles):
            self.battles.append(get_random_battle(base_level))
            base_level += 1
            if base_level > 100:
                base_level = 1
        self.current_state = self.battles[0]
        pickle.dump(self.battles, open(path, 'wb'))

    def load_battles(self, path):
        self.battles = pickle.load(open(path, 'rb'))
        self.current_state = self.battles[0]

    def reset(self):
        self.battle_index = self.reset_index
        self.episode_step = 0
        self.current_state[0].reset()
        self.current_state[1].reset()
        self.battle_index = 0
        return self.current_state
    
    def win(self, current_state):
        self.last_state = None
        self.same_state = 0
        if self.battle_index < len(self.battles):
            self.battle_index += 1
            current_state = self.battles[self.battle_index]
            current_state[0].reset()
            current_state[1].reset()
            return current_state, +10.0, False
        return current_state, +10.0, True
    
    def defeat(self, current_state):
        if current_state == self.last_state:
            self.same_state += 1
        else:
            self.last_state = current_state
            self.same_state = 0
        if self.same_state > 5:
            self.battle_index += 1
            self.reset_index += 1
            current_state = self.battles[self.battle_index]
            current_state[0].reset()
            current_state[1].reset()
            return current_state, 0.0, False
        return current_state, -30.0, True

    def step(self, current_state, action):
        self.episode_step += 1
        if self.episode_step == 1:
            self.last_state = current_state
        pokemon_a = current_state[0]
        pokemon_b = current_state[1]
        pokemon_b_hp_start = pokemon_b.hp
        done = False
        attacker_label = None
        if pokemon_a.speed > pokemon_b.speed:
            attacker_label = 'a'
        if pokemon_a.speed < pokemon_b.speed:
            attacker_label = 'b'
        else:
            attacker_label = np.random.choice(['a', 'b'])

        if attacker_label == 'a':
            attacker = pokemon_a
            defender = pokemon_b
            if action >= len(pokemon_a.moves):
                return current_state, True, -10.0
            move = pokemon_a.moves[action]
            print(f'Pokemon a chooses {move.name}')
            if move.pp == 0:
                print(f'Finished pp for move {move.name}, can\'t use it!')
                return current_state, True, -10.0
        else:
            attacker = pokemon_b
            defender = pokemon_a
            move = choose_move(attacker)
            print(f'Pokemon b chooses {move.name}')

        if move is not None:
            apply_move(attacker, defender, move)
        else:
            moves_exhausted = True

        if defender.current_hp <= 0:
            if attacker_label == 'a':
                return self.win(current_state)
            else:
                return self.defeat(current_state)

        # Other pokemon attacks
        if attacker_label == 'a':
            attacker_label = 'b'
            attacker = pokemon_b
            defender = pokemon_a
            move = choose_move(attacker)
            print(f'Pokemon b chooses {move.name}')
        else:
            attacker_label = 'a'
            attacker = pokemon_a
            defender = pokemon_b
            if action >= len(pokemon_a.moves):
                return current_state, True, -10.0
            move = pokemon_a.moves[action]
            print(f'Pokemon a chooses {move.name}')
            if move.pp == 0:
                print(f'Finished pp for move {move.name}, can\'t use it!')
                return current_state, True, -10.0
        if move is not None:
            apply_move(attacker, defender, move)

        if defender.current_hp <= 0:
            if attacker_label == 'a':
                return self.win(current_state)
            else:
                return self.defeat(current_state)
        reward = (pokemon_b_hp_start - pokemon_b.current_hp) / pokemon_b.hp * 100
        return (pokemon_a, pokemon_b), reward, False
