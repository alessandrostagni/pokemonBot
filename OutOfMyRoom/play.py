"""
Code edited from:
https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/

Plays on the visual boy advance emulator by reading the memory values.
Since it runs an ahk script, make sure you have everything set up as explained in the README!

Would be amazing to have a fast interaction mechanism on the emulator, but AHK Script + Memory Viewer is the
only one I have found so far.
"""


from ahk import AHK
import numpy as np
import random
import tensorflow as tf

from OutOfMyRoom.DQNAgent.DQNAgent import DQNAgent
from OutOfMyRoom.DQNAgent.DQNAgent import BlobEnv

#  Stats settings
AGGREGATE_STATS_EVERY = 1  # episodes
SHOW_PREVIEW = False

ahk = AHK()
ahk.run_script(open('ahk_scripts/setup.ahk').read())

agent = DQNAgent()
agent.load_model(
    "..\\models_saved\\OutOfMyRoom\\Yoyo____92.00max___92.00avg___92.00min__1604357490.model"
)

env = BlobEnv()

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

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
