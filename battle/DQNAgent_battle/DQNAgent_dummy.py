"""
Code edited from:
https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/

DQNAgent trained by letting a pokemon play the same battle MAX_SAME_BATTLE times
before moving to the next battle.
Note that the Q values the agent is meant to learn are moving but also lower as related to only one battle.
"""

import pickle
from collections import deque

from keras.models import load_model
from keras.callbacks import TensorBoard

from dummy_helpers import *

ACTION_SPACE_SIZE = 4
MODEL_NAME = "Yoyo"
DISCOUNT = 0.1
REPLAY_MEMORY_SIZE = 100000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MAX_STEPS = 200
MAX_SAME_BATTLE = 70

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
DATA_DIR = os.path.join(ROOT_DIR, '../data')
TYPE_MODS = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'type_modifiers.csv')).set_index('attack_type')
VERBOSE = True


class DQNAgent:

    def __init__(self):
        # Main model
        self.model = create_model(ACTION_SPACE_SIZE)

        # Target network
        self.target_model = create_model(ACTION_SPACE_SIZE)
        # self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        # self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def load_model(self, model_path):
        self.model = load_model(model_path)

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append((
            decode_state(transition[0]), transition[1],
            transition[2], decode_state(transition[3]),
            transition[4]
        ))

    def train(self, terminal_state):
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

        x = []
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
            x.append(current_state)
            y.append(current_qs)

        x = np.array(x)
        y = np.array(y)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            x, y, batch_size=MINIBATCH_SIZE, verbose=0,
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
        state = decode_state(state)
        input_state = np.array(state).reshape(-1, len(state))
        print('Input state:', input_state)
        prediction = self.model.predict(input_state)[0]
        print('Prediction:', prediction)
        return prediction


class BlobEnv:

    def __init__(self, n_battles, start_index):
        self.n_battles = n_battles
        self.ACTION_SPACE_SIZE = 4
        self.battles = []
        self.first_attack = True
        self.battle_index = start_index
        self.reset_index = 0
        self.same_battle = 1

    def create_battles(self, path):
        base_level = 1
        while len(self.battles) < self.n_battles:
            # HARDCODE LEVEL FOR TESTING
            # b = get_random_battle(50)
            b = get_random_battle(base_level)
            if is_battle_winnable(b[0], b[1]):
                self.battles.append(b)
                base_level += 1
                if base_level > 100:
                    base_level = 1
        pickle.dump(self.battles, open(path, 'wb'))

    def load_battles(self, path):
        self.battles = pickle.load(open(path, 'rb'))

    def reset(self):
        current_state = self.battles[self.battle_index]
        if self.battle_index == 0:
            self.reset_battles()
        return current_state

    def reset_battles(self):
        for b in self.battles:
            b[0].reset()
            b[1].reset()

    def check_winner(self, current_state, defender, pokemon_b):
        if defender == pokemon_b:
            return self.win(current_state)
        else:
            return self.defeat(current_state)

    def step(self, current_state, action):
        pokemon_a = current_state[0]
        pokemon_b = current_state[1]
        # pokemon_b_hp_start = pokemon_b.current_hp

        move_feasibility = check_move_feasibility(pokemon_a, action)
        if move_feasibility == 'fail_move':
            return current_state, -100.0, True, 'fail_move'
        elif move_feasibility == 'ok':
            agent_attack(pokemon_a, pokemon_b, action)

        if pokemon_b.current_hp <= 0:
            return self.win_train(current_state)

        # More complex reward function for unfinished battles.
        # More realistic but also Q values are harder to learn so toggled off by now.
        # reward = (pokemon_b_hp_start - pokemon_b.current_hp) / pokemon_b_hp_start * 100
        reward = -1.0
        return (pokemon_a, pokemon_b), reward, False, 'continue'

    def win_train(self, current_state):
        if self.same_battle >= MAX_SAME_BATTLE:
            self.battle_index += 1
            self.same_battle = 1
            if self.battle_index < len(self.battles):
                current_state = self.battles[self.battle_index]
                return current_state, +100.0, True, 'win'
            else:
                self.battle_index = 0
        self.same_battle += 1.0
        new_state = tuple(current_state)
        current_state[0].reset()
        current_state[1].reset()
        return new_state, +100.0, True, 'win'

    # METHODS USED FOR EVALUATION ONLY

    def win(self, current_state):
        self.battle_index += 1
        if self.battle_index < len(self.battles):
            current_state = self.battles[self.battle_index]
            return current_state, +100.0, False, 'win'
        self.battle_index = 0
        return current_state, +100.0, True, 'win'

    def draw(self, current_state):
        self.battle_index += 1
        if self.battle_index < len(self.battles):
            current_state = self.battles[self.battle_index]
            return current_state, 0.0, False, 'draw'
        self.battle_index = 0
        return current_state, 0.0, True, 'draw'

    def defeat(self, current_state):
        self.battle_index += 1
        if self.battle_index < len(self.battles):
            current_state = self.battles[self.battle_index]
            return current_state, -100.0, False, 'lost'
        self.battle_index = 0
        return current_state, -100.0, True, 'lost'

    def step_real_battle(self, current_state, action):
        pokemon_a = current_state[0]
        pokemon_b = current_state[1]
        # pokemon_b_hp_start = pokemon_b.current_hp
        first_attacker_label = decide_first_attacker(pokemon_a, pokemon_b)
        print('FIRST ATTACKER:')
        print(first_attacker_label)

        defender = None
        if first_attacker_label == 'a':
            move_feasibility = check_move_feasibility(pokemon_a, action)
            if move_feasibility == 'fail_move':
                return current_state, -100.0, True, 'fail_move'
            elif move_feasibility == 'ok':
                defender = agent_attack(pokemon_a, pokemon_b, action)
        else:
            defender = bot_attack(pokemon_a, pokemon_b)

        if defender.current_hp <= 0:
            return self.check_winner(current_state, defender, pokemon_b)

        # Other pokemon attacks
        if first_attacker_label == 'a':
            attacker, defender = bot_attack(pokemon_a, pokemon_b)
        else:
            move_feasibility = check_move_feasibility(pokemon_a, action)
            if move_feasibility == 'fail_move':
                return current_state, -100.0, True, 'fail_move'
            elif move_feasibility == 'ok':
                defender = agent_attack(pokemon_a, pokemon_b, action)

        if defender.current_hp <= 0:
            return self.check_winner(current_state, defender, pokemon_b)

        # Check for draw
        draw = check_draw(pokemon_a, pokemon_b)
        if draw:
            return self.draw(current_state)

        # reward = (pokemon_b_hp_start - pokemon_b.current_hp) / pokemon_b_hp_start * 100
        reward = -10
        return (pokemon_a, pokemon_b), reward, False, 'continue'
