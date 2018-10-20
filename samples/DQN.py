# coding: utf-8
""" DQN で重要な4つのポイント
1, Experience Replay
2, Fixed Target Q-Network
3, Reward Clipping
4, Huber Loss
"""
import random
import gym
import numpy as np
from collections import deque, namedtuple
import tensorflow as tf
import tensorflow.nn as nn
from keras.layers import Dense
from keras.models import Model
from tensorflow.train import AdamOptimizer
from tensorflow.losses import huber_loss
from keras.activations import linear


Transition = namedtuple("Transition", ("state", "action", "state_next", "reward"))
ENV = "CartPole-v0"
MAX_STEPS = 200
NUM_EPISODES = 500
BATCH_SIZE = 32
CAPACITY = 10000
GAMMA = 0.99


class ExperienceMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        #if len(self.memory) < self.capacity:
        #    self.memory.append(None)

        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(Model):
    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()
        self.fc1 = Dense(32, input_shape=(num_states, ))
        self.fc2 = Dense(32, activation=None)
        self.fc3 = Dense(num_actions, activation=None)

    def call(self, inputs, training=False, mask=None):
        x = nn.relu(self.fc1(inputs))
        x = nn.relu(self.fc2(x))
        x = linear(self.fc3(x))
        return x


class Brain:
    def __init__(self, num_states, num_actions):
        self.memory = ExperienceMemory(CAPACITY)
        self.num_states, self.num_actions = num_states, num_actions

        self.batch = None
        self.state_batch = None
        self.action_batch = None
        self.states_next_batch = None
        self.reward_batch = None

        self.optimizer = AdamOptimizer()
        self.main_q_network = Net(num_states, num_actions)
        self.main_q_network.compile(loss=huber_loss, optimizer=self.optimizer)
        self.target_q_network = Net(num_states, num_actions)
        self.target_q_network.compile(loss=huber_loss, optimizer=self.optimizer)

#    def replay(self):
#        if len(self.memory) < BATCH_SIZE:
#            return

#        self.batch, self.state_batch, self.action_batch, self.states_next_batch, self.reward_batch =\
#            self.make_minibatch()

#        mainQ = self.main_q_network.predict(self.state_batch)
#        next_action = np.argmax(mainQ)
#        target = self.reward_batch + GAMMA * self.target_q_network.predict(self.states_next_batch)[next_action]
#        self.main_q_network.fit(self.num_stetes, target, epochs=1, verbose=0)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        inputs = np.zeros((BATCH_SIZE, self.num_states))
        targets = np.zeros((BATCH_SIZE, self.num_actions))
        transitions = self.memory.sample(BATCH_SIZE)

        for i, (state_batch, action_batch, next_state_batch, reward_batch) in enumerate(transitions):
            inputs[i:i+1] = state_batch
            target = reward_batch

            if not (next_state_batch == np.zeros(state_batch.shape)).all(axis=1):
                mainQ = self.main_q_network.predict(state_batch)[0]
                next_action = np.argmax(mainQ)
                target = reward_batch + GAMMA * self.target_q_network.predict(next_state_batch)[0][next_action]

            targets[i] = self.main_q_network.predict(state_batch)
            targets[i][action_batch] = target
            self.main_q_network.fit(inputs, targets, epochs=1, verbose=0)

    def make_minibatch(self, batch_size=BATCH_SIZE):
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = tf.concat(batch.state, axis=0)
        action_batch = tf.concat(tf.cast(batch.action, tf.int32), axis=0)
        states_next_batch = tf.concat(batch.state_next, axis=0)
        reward_batch = tf.concat(tf.cast(batch.reward, tf.int32), axis=0)

        return batch, state_batch, action_batch, states_next_batch, reward_batch

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.main_q_network.predict(state))
        else:
            action = np.random.choice(self.num_actions)
        return action


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        return self.brain.decide_action(state, episode)

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)


class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.agent = Agent(self.num_states, self.num_actions)

    def run(self):
        complete_episodes = 0
        episode_final = False

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()
            state = observation
            state = np.reshape(state, [1, self.num_states])

            self.agent.brain.target_q_network = self.agent.brain.main_q_network

            for step in range(MAX_STEPS):
                #self.env.render()
                action = self.agent.get_action(state, episode)
                next_state, _, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.num_states])

                if done:
                    next_state = np.zeros(state.shape)
                    if observation[0] < 0.5:
                        # if step < 195:

                        reward = -1
                        complete_episodes = 0
                    else:
                        reward = 1
                        complete_episodes += 1
                else:
                    reward = 0

                self.agent.memorize(state, action, next_state, reward)
                self.agent.update_q_function()

                state = next_state

                if done:
                    print("{} Episode: Finished after {} steps: complete_episodes: {}".format(
                        episode, step + 1, complete_episodes))
                    break

                if episode_final:
                    self.env.render()

                    break

                if complete_episodes >= 10:
                    print("10回成功")
                    episode_final = True


if __name__ == '__main__':
    env = Environment()
    env.run()
