"""
Code edited from:
https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/

Trains a DQNAgent against real battles, saves models snapshots and tensorboard logs.
"""

import time

import tensorflow as tf
from tqdm import tqdm

from DQNAgent_battle import *
from DQNAgent_battle import create_model
from train_helpers import print_state


# Environment settings
EPISODES = 10000
N_BATTLES = 100

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
MODEL_NAME = 'Yoyo'
# epsilon = 0
# EPSILON_DECAY = 0
# MIN_EPSILON = 0

#  Stats settings
AGGREGATE_STATS_EVERY = 1  # episodes


agent = DQNAgent()
env = BlobEnv(N_BATTLES)
env.create_battles(r'battles_100.pickle')
# env.load_battles(r'battles_100.pickle')

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
lost_summary_writer = tf.summary.create_file_writer('logs/lost')
draw_summary_writer = tf.summary.create_file_writer('logs/draw')
n_battles_summary_writer = tf.summary.create_file_writer('logs/n_battles')

# Create models folder
if not os.path.isdir('models_battle'):
    os.makedirs('models_battle')

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    # agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    win = 0
    lost = 0
    draw = 0
    n_battles = 0

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
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)
            print('RANDOM MOVE:')

        print('Action:', action)
        print('Before fighting:')
        print_state(current_state)

        new_state, reward, done, outcome = env.step(current_state, action)

        print('BATTLE INDEX:', env.battle_index + 1)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
        #   env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)

        print('New state:', new_state)
        print('Reward: ', reward)
        print('Done: ', done)
        print('Step: ', step)
        print('Epsilon: ', epsilon)
        print('After fighting:')
        print_state(new_state)
        print('-------')

        if outcome == 'win':
            win += 1.0
            print('WIN')
        elif outcome == 'lost':
            lost += 1.0
            print('LOST')
        elif outcome == 'draw':
            draw += 1.0
            print('DRAW')
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
    with lost_summary_writer.as_default():
        tf.summary.scalar('lost', lost, step=episode)
    with draw_summary_writer.as_default():
        tf.summary.scalar('draw', draw, step=episode)
    with n_battles_summary_writer.as_default():
        tf.summary.scalar('n_battles', n_battles, step=episode)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

    agent.model.save(f'models_battle/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f} \
                     avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
