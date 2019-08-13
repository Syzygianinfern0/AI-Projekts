import gym
import tensorflow as tf
import numpy as np
from collections import deque
import math

env = gym.make('CartPole-v0')
env.seed(0)


# noinspection PyAbstractClass
class Policy(tf.keras.Model):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super().__init__()
        self.a_size = a_size
        self.d1 = tf.keras.layers.Dense(h_size, activation='relu')
        self.d2 = tf.keras.layers.Dense(a_size, activation='softmax')

    # noinspection PyMethodOverriding
    def call(self, x):
        x = self.d1(x)
        return self.d2(x)

    def act(self, state):
        probs = self.call(tf.expand_dims(state, axis=0))
        choice = np.random.choice(self.a_size, p=np.squeeze(probs))
        log_prob = np.log(probs.numpy().squeeze()[choice])
        return choice, log_prob


# noinspection PyShadowingNames
def reinforce(policy, opt, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    # noinspection PyShadowingNames
    tape = None
    scores = []
    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        dis_rewards = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * dis_rewards)

        with tf.GradientTape() as tape:
            action, log_prob = policy.act(state)
            loss_value = tf.reduce_sum(policy_loss)

        grads = tape.gradient(loss_value, policy.trainable_variables)
        opt.apply_gradients(zip(grads, policy.trainable_variables))

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            break

    return scores


if __name__ == '__main__':
    policy = Policy()
    opt = tf.keras.optimizers.Adam(lr=0.01)
    reinforce(policy, opt)
