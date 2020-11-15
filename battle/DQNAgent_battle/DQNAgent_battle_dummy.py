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
        for move in pokemon_a.moves:
            move_max_hits = None
            move_accuracy = None
            move_power = None
            if move.max_hits is None:
                move_max_hits = -1
            else:
                move_max_hits = move.max_hits

            if move.accuracy is None:
                move_accuracy = -1
            else:
                move_accuracy = move.accuracy

            move_power = move.power
            if move.power is None:
                move_power = -1
            else:
                move_power = move.power

            moves_data.append((
                move.get_id(), move.current_pp, move_max_hits, move_accuracy, move_power
            ))
        for i in range(4 - len(pokemon_a.moves)):
            moves_data.append((-1, -1, -1, -1, -1))

        state = [
            pokemon_a.types[0], pokemon_a.types[1],
            pokemon_a.attack, pokemon_a.defense,
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
            moves_data[0][2],
            moves_data[1][2],
            moves_data[2][2],
            moves_data[3][2],
            moves_data[0][3],
            moves_data[1][3],
            moves_data[2][3],
            moves_data[3][3],
            moves_data[0][4],
            moves_data[1][4],
            moves_data[2][4],
            moves_data[3][4],
            pokemon_b.current_hp, pokemon_b.types[0], pokemon_b.types[1]
        ]
        return state

    def create_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(30,)))
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
        self.first_attack = True
        self.episode_step = 0
        self.battle_index = 0
        self.reset_index = 0
        self.current_state = None

    def create_battles(self, path):
        base_level = 1
        while len(self.battles) < self.n_battles:
            ### HARDCODE LEVEL FOR TESTING
            b = get_random_battle(50)
            if is_battle_winnable(b[0], b[1]):
                self.battles.append(b)
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
        for b in self.battles:
            b[0].reset()
            b[1].reset()
        current_state = self.battles[self.battle_index]
        return self.current_state
    
    def win(self, current_state):
        self.battle_index += 1
        if self.battle_index < len(self.battles):
            current_state = self.battles[self.battle_index]
            return current_state, +100.0, False, 'win'
        self.battle_index = 0
        return current_state, +100.0, True, 'win'\

    def check_move_feasibility(self, current_state, pokemon_a, action):
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
            print('THIS SHOULD NOT HAPPEN!')
            stop
            return 'fail_move'
        return 'ok'

    def agent_attack(self, pokemon_a, pokemon_b, current_state, action):
        # Assign attacker and defender roles
        attacker = pokemon_a
        defender = pokemon_b
        move = pokemon_a.moves[action]
        apply_move(attacker, defender, move)
        return attacker, defender

    def check_winner(self, current_state, attacker, defender, pokemon_a, pokemon_b):
        if defender == pokemon_b:
            return self.win(current_state)
        else:
            return self.defeat(current_state)

    def check_draw(self, pokemon_a, pokemon_b):
        draw = True
        for move in pokemon_a.moves:
            if move.current_pp > 0:
                return False
        for move in pokemon_b.moves:
            if move.current_pp > 0:
                return False
        return draw

    def step(self, current_state, action):
        self.episode_step += 1
        pokemon_a = current_state[0]
        pokemon_b = current_state[1]
        pokemon_b_hp_start = pokemon_b.current_hp

        move_feasibility = self.check_move_feasibility(current_state, pokemon_a, action)
        if move_feasibility == 'fail_move':
            return current_state, -1000.0, True, 'fail_move'
        elif move_feasibility == 'ok':
            attacker, defender = self.agent_attack(pokemon_a, pokemon_b, current_state, action)

        if pokemon_b.current_hp <= 0:
            return self.win(current_state)

        reward = - (100.0 - ((pokemon_b_hp_start - pokemon_b.current_hp) / pokemon_b_hp_start) * 100) / 10
        return (pokemon_a, pokemon_b), reward, False, 'continue'

    # METHODS USED FOR EVALUATION ONLY #

    def win(self, current_state):
        self.battle_index += 1
        if self.battle_index < len(self.battles):
            current_state = self.battles[self.battle_index]
            return current_state, +100.0, False, 'win'
        self.battle_index = 0
        return current_state, +100.0, True, 'win'

    def defeat(self, current_state):
        self.battle_index += 1
        if self.battle_index < len(self.battles):
            current_state = self.battles[self.battle_index]
            return current_state, -100.0, False, 'lost'
        self.battle_index = 0
        return current_state, -100.0, True, 'lost'

    def bot_attack(self, pokemon_a, pokemon_b):
        attacker = pokemon_b
        defender = pokemon_a
        move = choose_move(attacker)
        if move is not None:
            print(f'Pokemon b chooses {move.name}')
            apply_move(attacker, defender, move)
        return attacker, defender

    def decide_first_attacker(self, pokemon_a, pokemon_b):
        attacker_label = None
        if pokemon_a.speed > pokemon_b.speed:
            return 'a'
        elif pokemon_a.speed < pokemon_b.speed:
            return 'b'
        return np.random.choice(['a', 'b'])

    def step_real_battle(self, current_state, action):
        pokemon_a = current_state[0]
        pokemon_b = current_state[1]
        pokemon_b_hp_start = pokemon_b.current_hp
        first_attacker_label = self.decide_first_attacker(pokemon_a, pokemon_b)
        print('FIRST ATTACKER:')
        print(first_attacker_label)

        if first_attacker_label == 'a':
            move_feasibility = self.check_move_feasibility(current_state, pokemon_a, action)
            if move_feasibility == 'fail_move':
                return current_state, -100.0, True, 'fail_move'
            elif move_feasibility == 'ok':
                attacker, defender = self.agent_attack(pokemon_a, pokemon_b, current_state, action)
        else:
            attacker, defender = self.bot_attack(pokemon_a, pokemon_b)

        if defender.current_hp <= 0:
            return self.check_winner(current_state, attacker, defender, pokemon_a, pokemon_b)

        # Other pokemon attacks
        if first_attacker_label == 'a':
            attacker, defender = self.bot_attack(pokemon_a, pokemon_b)
        else:
            move_feasibility = self.check_move_feasibility(current_state, pokemon_a, action)
            if move_feasibility == 'fail_move':
                return current_state, -100.0, True, 'fail_move'
            elif move_feasibility == 'ok':
                attacker, defender = self.agent_attack(pokemon_a, pokemon_b, current_state, action)

        if defender.current_hp <= 0:
            return self.check_winner(current_state, attacker, defender, pokemon_a, pokemon_b)

        # Check for draw
        draw = self.check_draw(pokemon_a, pokemon_b)
        if draw:
            return self.draw(current_state)

        reward = (pokemon_b_hp_start - pokemon_b.current_hp) / pokemon_b_hp_start * 100
        return (pokemon_a, pokemon_b), reward, False, 'continue'
