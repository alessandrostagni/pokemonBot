## Code edited from: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/ ###

import os
import sys

import numpy as np
import random
import tensorflow as tf
import time
from tqdm import tqdm

from DQNAgent_battle import *


# Environment settings
EPISODES = 100000
N_BATTLES = 10000

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
env = BlobEnv(N_BATTLES)
env.create_battles(r'battles.pickle')
env.load_battles(r'battles.pickle')

# For stats
ep_rewards = [-200]
win_battles = 0
lost_battles = 0


# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
reward_summary_writer = tf.summary.create_file_writer('logs/reward')
epsilon_summary_writer = tf.summary.create_file_writer('logs/epsilon')
win_summary_writer = tf.summary.create_file_writer('logs/win')
lost_summary_writer = tf.summary.create_file_writer('logs/lost')
draw_summary_writer = tf.summary.create_file_writer('logs/draw')

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

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


    # Reset environment and get initial state
    current_state = env.reset()

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

        print('Action:', action)
        print('Before fighting:')
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

        new_state, reward, done, outcome = env.step(current_state, action)

        print('New state:', new_state)
        print('Reward: ', reward)
        print('Done: ', done)
        print('Step: ', step)
        print('Epsilon: ', epsilon)
        print('After fighting:')
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
        print('-------')

        if outcome == 'win':
            current_state[0].reset()
            current_state[1].reset()
            win += 1.0
            print('WIN')
        elif outcome == 'lost':
            current_state[0].reset()
            current_state[1].reset()
            lost += 1.0
            print('LOST')
        elif outcome == 'draw':
            current_state[0].reset()
            current_state[1].reset()
            draw += 1.0
            print('DRAW')
        elif outcome == 'fail_move':
            current_state[0].reset()
            current_state[1].reset()
            print('FAIL_MOVE')

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
    ep_rewards.append(episode_reward)
    #if not episode % AGGREGATE_STATS_EVERY or episode == 1:
    average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
    min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
    max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
    # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

    # Save model, but only when min reward is greater or equal a set value
    #if min_reward >= MIN_REWARD:
    agent.model.save(f'models_battle/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)