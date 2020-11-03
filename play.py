import os

from ahk import AHK
import numpy as np
import random
import tensorflow as tf
import time
from tqdm import tqdm

from DQNAgent import DQNAgent
from DQNAgent import BlobEnv

#  Stats settings
AGGREGATE_STATS_EVERY = 1  # episodes
SHOW_PREVIEW = False

ahk = AHK()
ahk.run_script(open('ahk_scripts/setup.ahk').read())

agent = DQNAgent()
agent.load_model(
    "models_simulation\\Yoyo____92.00max___92.00avg___92.00min__1604357490.model"
)

env = BlobEnv()

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
# Memory fraction, used mostly when training multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

current_state = env.reset()
step = 0
done = False

while not done:
    print('Current state:', current_state)
    action = np.argmax(agent.get_qs(current_state))
    print('Step:', step)
    print('Current state:', current_state)
    print('Action:', action)
    new_state, reward, done = env.step(current_state, action)
    print('New state:', new_state)
    print(reward)
    print(done)
    print('------')
    current_state = new_state
    step += 1