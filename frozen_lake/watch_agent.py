import gym
import numpy as np
import time
import os
from constants import q_params, Q_TABLE_FILENAME

def clear_output():
    os.system('cls')

def play_game(env, q_table):
    for i_episode in range(3):
        state = env.reset()
        print(f'*** EPISODE {i_episode + 1} ***\n\n\n')
        time.sleep(1)

        for i_step in range(q_params['max_steps_per_episode']):
            clear_output()
            env.render()
            time.sleep(0.3)

            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if done:
                clear_output()
                env.render()
                if reward == 1:
                    print('You reached the goal!')
                    time.sleep(3)
                else:
                    print('You fell through a hole!')
                    time.sleep(3)
                clear_output()
                break

# Load learned Q-learning params
q_table = np.load(Q_TABLE_FILENAME)

# Create an environment
env = gym.make('FrozenLake-v0')

play_game(env, q_table)