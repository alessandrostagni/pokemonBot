## Code edited from: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/ ###

import os

from ahk import AHK
import numpy as np
import random
import tensorflow as tf
import time
from tqdm import tqdm

from DQNAgent_simulation import DQNAgent
from DQNAgent_simulation import BlobEnv


# Environment settings
EPISODES = 50000

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


m = DQNAgent().create_model()
agent = DQNAgent()
env = BlobEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models_simulation'):
    os.makedirs('models_simulation')

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    # agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()
    path = [(current_state[0], current_state[1], None)]

# Reset flag and start iterating until episode ends
    done = False
    while not done:
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        print('Current state:', current_state)
        print('Location memory:', env.location_memory)
        print('Action:', action)

        new_state, reward, done, same = env.step(current_state, action)
        path.append((new_state[0], new_state[1], action))

        print('New state:', new_state)
        print('Reward: ', reward)
        print('Done: ', done)
        print('Step: ', step)
        print('Path:', path)
        print('Epsilon: ', epsilon)
        
        print('-------')


        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
        #   env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    #if not episode % AGGREGATE_STATS_EVERY or episode == 1:
    average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
    min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
    max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
    # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

    # Save model, but only when min reward is greater or equal a set value
    #if min_reward >= MIN_REWARD:
    agent.model.save(f'models_simulation/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)