import matplotlib.pyplot as plt
import numpy as np
import time
import gym
import os
from tqdm import tqdm

from constants import q_params, Q_TABLE_FILENAME

def train_agent(env, config):
    # Retrieve action space and observation space
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    # Construct a Q table
    q_table = np.zeros((state_space_size, action_space_size))
    
    # Extract Q learning parameters
    num_episodes = config['num_episodes']
    max_steps_per_episode = config['max_steps_per_episode']
    learning_rate = config['learning_rate']
    discount_rate = config['discount_rate']
    exploration_rate = config['exploration_rate']
    max_exploration_rate = config['max_exploration_rate']
    min_exploration_rate = config['min_exploration_rate']
    exploration_decay_rate = config['exploration_decay_rate']

    rewards_total_episodes = []

    for i_episode in tqdm(range(num_episodes)):
        state = env.reset()
        rewards_current_episode = 0

        for i_step in range(max_steps_per_episode):
            # Render the environment
            # env.render()
            exploration_rate_threshold = np.random.uniform()
            if exploration_rate_threshold > exploration_rate and ~np.all(q_table[state] == 0):
                action = np.argmax(q_table[state])
            else:
                action = env.action_space.sample()    
            
            # Take action and get an observation and a reward
            new_state, reward, done, info = env.step(action)

            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            state = new_state
            rewards_current_episode += reward

            if done:
                break
        
        # Exploration rate decay
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * i_episode)

        # Save the rewards of the current episode
        rewards_total_episodes.append(rewards_current_episode)

    return q_table, rewards_total_episodes

# Create the environment
env = gym.make('FrozenLake-v0')

# Train an agent
q_table, rewards = train_agent(env, q_params)
np.save(Q_TABLE_FILENAME, q_table)

# Print the rewards
rewards_per_thousand = np.split(np.array(rewards), q_params['num_episodes'] / 1000)
for i, r in enumerate(rewards_per_thousand):
    print(f'Average reward after {(i + 1) * 1000} episodes: {r.mean()}')

env.close()