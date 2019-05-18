import gym
env = gym.make('MountainCarContinuous-v0')
env.reset()

for _ in range(500):
    env.render()
    env.step(env.action_space.sample())

# print(gym.envs.registry.all())
