# G: && cd G:\Il mio Drive\RL\CartPole
import gymnasium as gym
import CartPole as cp
import pickle

# Training ----------------------------------------------------

env = gym.make('CartPole-v1')
agent = cp.Agent(env)
agent.train(env, n_episodes=1500)

with open('cache/agent.pickle', 'wb') as file:
    pickle.dump(agent, file)

# Evaluation --------------------------------------------------

"""
with open('cache/agent.pickle', 'rb') as file:
    agent = pickle.load(file)
"""

performances = []
for i in range(1000):
    print('# Test round:', i)
    performances.append(cp.play(agent))

with open('cache/performances.pickle', 'wb') as file:
    pickle.dump(performances, file)
