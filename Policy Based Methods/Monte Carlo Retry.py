import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np

LEARNING_RATE = 1e-2
TRAIN_EPISODES = 100
GAMMA = 0.99


class Reinforce(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def act(self, state):
        # noinspection PyArgumentList
        state = Variable(torch.Tensor(state).unsqueeze(0))
        probs = self.forward(state).squeeze(0)
        action = np.random.choice(2, p=probs.detach().numpy())
        return action, torch.log(probs[action])

    # noinspection PyShadowingNames
    @staticmethod
    def update_weights(experience, optimizer):
        discounter = Variable(torch.Tensor([0]))
        for (state, action, reward, log_prob) in experience:
            discounter = Variable(torch.Tensor([reward])) + GAMMA * discounter
            loss = (-1.0) * discounter * log_prob
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(0)
    agent = Reinforce()
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    for epi in range(TRAIN_EPISODES):
        state = env.reset()
        experience = []
        epi_reward = 0
        done = False
        while not done:
            action, log_prob = agent.act(state)
            next_state, reward, done, info = env.step(action)
            experience.append([state, action, reward, log_prob])
            epi_reward += reward
            state = next_state
            if done:
                print(f"Episode : {epi + 1}   Reward : {epi_reward}")

        agent.update_weights(experience, optimizer)
    env.close()
