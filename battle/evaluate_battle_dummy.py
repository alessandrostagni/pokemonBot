"""
Code edited from:
https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/

Evaluate an agent through a series of battles.
Print win rate on screen.
"""
from DQNAgent_dummy import *

# Instantiate battles
N_BATTLES = 10000

agent = DQNAgent()
# agent.load_model(f"C:\\Users\\darth\\PycharmProjects\\pokemonBot\\battle\\models_battle\\episode_146169_reward___97.00_time__1605901938.model")
agent.load_model(f"C:\\Users\\darth\\PycharmProjects\\pokemonBot\\"
                 f"battle\\models_battle\\episode_242418_reward___96.00_time__1606076034.model")
env = BlobEnv(N_BATTLES, 0)
#env.create_battles(r'battles_dummy_10000_multilevel_eval.pickle')
env.load_battles(r'battles_dummy_10000_multilevel_eval.pickle')
current_state = env.reset()

# Run battles

win = 0.0
draw = 0.0
lost = 0.0
fail_move = 0.0

for battle in env.battles:
    current_state = battle
    outcome = 'ok'
    while battle[0].current_hp > 0 and battle[1].current_hp > 0 and outcome != 'fail_move':
        action = np.argmax(agent.get_qs(battle))
        new_state, reward, done, outcome = env.step_real_battle(battle, action)

    if battle[0].current_hp <= 0:
        lost += 1.0
    elif battle[1].current_hp <= 0:
        win += 1.0
    elif outcome == 'fail_move':
        fail_move += 1.0
    print(win)
    print(lost)


print('Fail moves:', fail_move)
print('Win:', win)
print('Lost:', lost)
print('Draw:', draw)
print('Win rate:', (win / (win + draw + lost) * 100))
