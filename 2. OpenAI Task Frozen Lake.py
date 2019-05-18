import gym
import numpy as np
from time import sleep

env = gym.make('FrozenLake8x8-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Declaring Params
eta = .6
gma = .9
episodes = 5000
rewards = []

for i in range(episodes):
    s = env.reset()
    rAll = 0
    d = False
    j = 0

    # while j < 99:
    #     env.render()
    #     j += 1
    #
    #     a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
    #     s1, r, d, _ = env.step(a)
    #     Q[s, a] = (1 - eta) * Q[s, a] + eta * (r + gma * np.max(Q[s1, :]))
    #
    #     rAll += r
    #     s = s1
    #     if d:
    #         break
    # Reset environment
    s = env.reset()
    d = False
    # The Q-Table learning algorithm
    while d != True:
        env.render()
        # Choose action from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        # Get new state & reward from environment
        s1, r, d, _ = env.step(a)
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + eta * (r + gma * np.max(Q[s1, :]) - Q[s, a])
        s = s1
        sleep(0.5)
    # Code will stop at d == True, and render one state before it
    rewards.append(rAll)
    env.render()
env.close()
print("Average Reward : ", sum(rewards) / episodes)
