"""
Code edited from:
https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/

Trains a DQNAgent against a dummy, saves models snapshots and tensorboard logs.
"""

import tensorflow as tf
import time
from tqdm import tqdm

from DQNAgent_dummy import *
from train_helpers import print_state


# Environment settings
START_EPSILON = 0.90
EPISODES = 10000000
N_BATTLES = 10000
# Exploration settings
epsilon = START_EPSILON  # not a constant, going to be decayed
EPSILON_DECAY = 0.96
MIN_EPSILON = 0.001
MODEL_NAME = 'Yoyo'
START_INDEX = 0
START_EPISODE = 0

#  Stats settings
AGGREGATE_STATS_EVERY = 3  # episodes

agent = DQNAgent()
env = BlobEnv(N_BATTLES, START_INDEX)
# env.create_battles(r'battles_dummy_10000.pickle')
env.load_battles(r'battles_dummy_10000.pickle')

# For stats
ep_rewards = [-200]
win_battles = 0
lost_battles = 0
n_battles = 0


# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
reward_summary_writer = tf.summary.create_file_writer('logs/reward')
epsilon_summary_writer = tf.summary.create_file_writer('logs/epsilon')
win_summary_writer = tf.summary.create_file_writer('logs/win')
n_battles_summary_writer = tf.summary.create_file_writer('logs/n_battles')


# Create models folder
if not os.path.isdir('models_battle'):
    os.makedirs('models_battle')

# Iterate over episodes
for episode in tqdm(range(START_EPISODE, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    # agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    win = 0
    n_battles = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
            print('AGENT MOVE:')
        else:
            # Get random action
            action = np.random.randint(0, len(current_state[0].moves))
            print('RANDOM MOVE:')

        print('Action:', action)
        print('Before fighting:')
        print_state(current_state)

        old_battle_index = env.battle_index
        new_state, reward, done, outcome = env.step(current_state, action)
        new_battle_index = env.battle_index
        if new_battle_index != old_battle_index:
            epsilon = START_EPSILON
            n_battles += 1.0

        print('AFTER fighting')
        print('BATTLE INDEX:', env.battle_index + 1)
        print('SAME BATTLE:', env.same_battle)
        print_state(new_state)
        print_state(new_state)
        print('---------------')

        # Transform new continuous state to new discrete state and count reward
        episode_reward += reward

        # if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
        #   env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)

        if outcome == 'win':
            win += 1.0
            print('WIN')
        elif outcome == 'fail_move':
            print('FAIL_MOVE')

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    with reward_summary_writer.as_default():
        tf.summary.scalar('reward', episode_reward, step=episode)
    with epsilon_summary_writer.as_default():
        tf.summary.scalar('epsilon', epsilon, step=episode)
    with win_summary_writer.as_default():
        tf.summary.scalar('win', win, step=episode)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.model.save(
            f'models_battle/episode_{episode}_reward_{average_reward:_>7.2f}_time__{int(time.time())}.model'
        )

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
