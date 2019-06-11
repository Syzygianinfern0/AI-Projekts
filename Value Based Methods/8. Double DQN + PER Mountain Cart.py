# Library Imports
import tensorflow as tf
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import os
import time

# Basic Stuff
env = gym.make('MountainCar-v0')

# Constants
TRAIN_EPIS = 250			# Number of episodes to train on
REPLAY_SIZE = 1_00_000		# Experience memory size
EPSILON = 1					# Initial value
EPSILON_DECAY = 0.997		# Decay rate
EPSILON_MIN = 0.01			# Lowest Value
LEARNING_RATE = 0.005		# LR for Adam 
GAMMA = 0.96				# Discounter
BATCH_SIZE = 64				# Size of training Batch
PLOT_FREQ = 5				# Log frequency
TAU = 0.01					# Target network copy bias
ALPHA = 0.65				# Damper for probabilities
OFFSET = 0.1				# For calculating the new priority

# Path to save weights
PATH = f'results/weights/Double DQN + PER Mountain Cart Retry/lr-{LEARNING_RATE}_gma-{GAMMA}_epsdk-{EPSILON_DECAY}-' \
    f'{int(time.time())}'
os.makedirs(PATH, exist_ok=True)


# Agent
class Network:
    """Handles the brain"""

    def __init__(self, env):
    	"""
    	Init all class variables
    	:param env: Environment to work on
    	"""
        self.inputs = env.observation_space.shape[0]
        self.outputs = env.action_space.n
        self.eps = EPSILON
        self.eps_decay = EPSILON_DECAY
        self.eps_min = EPSILON_MIN
        self.learning_rate = LEARNING_RATE
        self.local_model = self.make_model()
        self.target_model = self.make_model()
        self.size = REPLAY_SIZE
        self.exp_memory = deque(maxlen=self.size)
        self.priorities = deque(maxlen=self.size)
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.goal_pts = np.arange(start=-0.4,	# Custom reward checkpoints to encourage
                                  stop=0.625,	# exploration and therefore attain the 
                                  step=0.025)	# target point
        self.i = 0								# Checkpoints reached
        self.alpha = ALPHA
        self.offset = OFFSET

    def update_target_network(self, tau=TAU):
        """
        Updates Target Network
        :return: None
        """
        local_wts = np.array(self.local_model.get_weights())
        target_wts = np.array(self.target_model.get_weights())
        self.target_model.set_weights(target_wts + tau * (local_wts - target_wts))

    def anneal_hps(self):
        """
        Updates Hyper-params
        :return: None
        """
        self.eps *= self.eps_decay if self.eps > self.eps_min else 1

    def get_action(self, state, evaluate=False):
        """
        Returns action based on epsilon greedy policy
        :param state: State present in
        :param evaluate: Evaluation flag
        :return: The action to be done
        """
        if evaluate:
            q = self.local_model.predict(state)
            action = np.argmax(q[0])
            return action
        if random.random() < self.eps:
            return random.randint(0, self.outputs - 1)
        else:
            q = self.local_model.predict(state)
            action = np.argmax(q[0])
            return action

    def append_experience(self, state, action, reward, next_state, done):
        """
        Adds to memory
        :param state: State in
        :param action: Action performed
        :param reward: Reward gained
        :param next_state: Resultant State
        :param done: Terminal flag
        :return: None
        """
        self.exp_memory.append((state, action, reward, next_state, done))
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self):
        probabilities = np.array(self.priorities) ** self.alpha
        p_sum = sum(probabilities)
        probabilities = probabilities / p_sum
        return probabilities

    def custom_reward(self, state):
        """
        Custom reward function generator
        :param state: State in
        :return: Reward
        """
        reward = 0
        while self.goal_pts[self.i] < state[0]:
            self.i += 1
            reward += 10		# For every checkpoint reached award 10
        if state[0] >= 0.5:
            print("REACHED GOAL!!!")
            reward += 30		# For level completion
            plt.scatter(epi, epi_reward + 30)
        return reward

    def make_model(self):
        """
        Makes the model with tf
        :return: The compiled model
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(100, 'relu', input_dim=self.inputs),
            tf.keras.layers.Dense(self.outputs, 'linear')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
            loss='mse'
        )
        return model

    def train(self):
        """
        Trains the Double DQN with the replay buffer
        :return: None
        """
        if len(self.exp_memory) > self.batch_size:
            # indices = random.choices(self.exp_memory, self.batch_size)
            probabilities = self.get_probabilities()
            indices = np.random.choice(
                a=range(len(self.exp_memory)),
                size=self.batch_size,
                p=probabilities,			# Samples based on propabilitiies
            )
            imp_samp_weights = 1 / (len(self.exp_memory) * probabilities[indices])
            imp_samp_weights = imp_samp_weights / max(imp_samp_weights)
            imp_samp_weights = imp_samp_weights ** (1-self.eps)		# ISW for the loss function
            batch = np.array(self.exp_memory)[indices]		
            states = np.array([exp[0] for exp in batch])			# Unpack experiences
            actions = np.array([exp[1] for exp in batch])
            rewards = np.array([exp[2] for exp in batch])
            next_states = np.array([exp[3] for exp in batch])
            dones = np.array([exp[4] for exp in batch])
            states = np.squeeze(states)								# Preprocessing
            next_states = np.squeeze(next_states)
            q_values_target = rewards + (1 - dones) * self.gamma * self.target_model.predict_on_batch(next_states)[0][
                np.argmax(self.local_model.predict_on_batch(next_states), axis=1)]		# Bellman Equation
            q_values_current = self.local_model.predict_on_batch(states)[[np.arange(self.batch_size)], [actions]]
            for i, p in zip(indices, np.squeeze(abs(q_values_target - q_values_current) + self.offset)):
                self.priorities[i] = p 		# Set new priorities
            q_tuples_current = self.local_model.predict_on_batch(states)
            q_tuples_target = q_tuples_current
            q_tuples_target[[np.arange(self.batch_size)], [actions]] = q_values_target	# Update favoured values to train upon

            self.local_model.fit(states, q_tuples_target, verbose=0, sample_weight=imp_samp_weights)
        else:
            return


def evaluate_agent(agent):
    """
    Evaluate agent based on q values
    :param agent: Object of Network Class
    :return: None
    """
    done = False
    state = env.reset()
    state = np.array([state])
    epi_reward = 0
    while not done:
        action = agent.get_action(state, evaluate=True)
        next_state, reward, done, info = env.step(action)
        env.render()
        next_state = np.array([next_state])
        epi_reward += reward
        agent.append_experience(state, action, reward, next_state, done)
        state = next_state
    print(f"Evaluation Score : {epi_reward}")


if __name__ == '__main__':
    agent = Network(env)
    rewards = []
    agg_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}	# To store progress
    timesteps = 0
    for epi in range(TRAIN_EPIS):
        agent.i = 0			# Checkpoint Index
        done = False
        state = env.reset()
        best_ht = state[0]	# To know progress
        state = np.array([state])
        epi_reward = 0
        while not done:
            timesteps += 1
            action = agent.get_action(state)	# Based on Epsilon greedy
            next_state, reward, done, info = env.step(action)
            if best_ht < next_state[0]:			# To track progress
                best_ht = next_state[0]
            env.render()
            reward = agent.custom_reward(next_state)	# Compute custom reward function encouraging exploration
            next_state = np.array([next_state])
            epi_reward += reward
            agent.append_experience(state, action, reward, next_state, done)
            agent.train()
            agent.update_target_network()
            state = next_state
            if timesteps % 20 == 0:
                agent.anneal_hps()		# Update hyper params
        rewards.append(epi_reward)
        print(f"Episode : {epi + 1}    Reward : {epi_reward}    Epsilon : {agent.eps}    Best Ht : {best_ht}")
        if not epi % PLOT_FREQ:
            avg_reward = sum(rewards[-PLOT_FREQ:]) / PLOT_FREQ
            agg_ep_rewards['ep'].append(epi)
            agg_ep_rewards['avg'].append(avg_reward)
            agg_ep_rewards['max'].append(max(rewards[-PLOT_FREQ:]))
            agg_ep_rewards['min'].append(min(rewards[-PLOT_FREQ:]))
            agent.local_model.save(f'{PATH}/{epi}.h5')
    plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['avg'], label="average rewards")
    plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['max'], label="max rewards")
    plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['min'], label="min rewards")
    plt.title(f"lr:{LEARNING_RATE} gma:{GAMMA}  eps_dk:{EPSILON_DECAY}")
    plt.legend(loc=4)
    plt.savefig(f'{PATH}/graph.png', bbox_inches='tight')
    plt.show()
