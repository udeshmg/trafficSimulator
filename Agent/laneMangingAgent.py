"""Training the agent"""

import random
from collections import deque
import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(0)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras import backend as K
K.set_epsilon(1e-03)
import sys
import matplotlib.pyplot as plt

# q_table = np.zeros([env.observation_space.n, env.action_space.n])

# implement a queue table here for a single intersection



NEGATIVE_REWARD = 2
SAVE_N_RESTORE = False
SAVE = True
DEBUG = 3
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)

def decompose_action(action_val, num_roads):
    action = action_val
    # Internal params
    if num_roads == 4:
        action_set = np.array([4, 3, 3, 3, 3])
    elif num_roads == 3:
        action_set = np.array([3, 3, 3, 3])

    decomposed_action = np.zeros(shape=len(action_set)).astype(int)
    for i in range(len(action_set)):
        decomposed_action[-i - 1] = action % action_set[-i - 1]
        action = int(action / action_set[-i - 1])
    return decomposed_action

def get_index(phase, roads, step=5):
    index = int(phase * (step ** len(roads)))
    for rd in range(0, len(roads)):
        index += int((roads[rd] * (step ** (len(roads)-rd-1))))
    return index

class DQNA_laneManager:
    def __init__(self, state_size, action_size, road,  id=1):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100)
        self.gamma = 0.75    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 #0.995
        self.learning_rate = 0.001 #0.001

        self.states = np.zeros(shape=(1, self.state_size))

        self.model, self.target_model = self._build_model()
        self.iter = 0
        self.debugLvl = 3
        self.id = id
        self.road = road

        #stat agent
        self.reward = np.empty(shape=0)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        #random.seed(0)
        #np.random.seed(0)
        #tf.set_random_seed(0)
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(24, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                    optimizer=Adam(lr=self.learning_rate, epsilon=1e-03))

        target_model = Sequential()
        target_model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        target_model.add(Dense(24, activation='tanh'))
        target_model.add(Dense(self.action_size, activation='linear'))
        target_model.compile(loss='mse',
                    optimizer=Adam(lr=self.learning_rate, epsilon=1e-03))


        #model._make_predict_function()
        #target_model._make_predict_function()
        #

        return model, target_model

    def copy_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() < self.epsilon:
            if self.debugLvl > 2:
                print("Agent ID", self.id, " Rondom Action ", self.id)
                return random.randrange(self.action_size)

        act_values = self.model.predict(state)[0]
        return  np.argmax(act_values) # returns action

    def replay(self, batch_size):

            random.seed(self.iter)
            minibatch = random.sample(self.memory, batch_size)

            for state, action, reward, next_state, done in minibatch:

                target = self.model.predict(state)
                if not done:
                    max_action = np.argmax(self.model.predict(next_state)[0])
                    t = self.target_model.predict(next_state)[0]
                    target[0][action] = reward + self.gamma * t[max_action]

                self.model.fit(state, target, epochs=1, verbose=0, shuffle=False, batch_size=16)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


    def load(self, name):
            self.epsilon = 0
            print("Agent: ", self.id, " Weights: ",name)
            self.model.load_weights(name)
            self.target_model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



    def actOnRoad(self):
        self.action = self.act(self.states)
        self.road.act(self.action)
        print("Action: ", self.action)

    def getOutputData(self):

        next_states, reward, is_done = self.road.getStates()
        next_states = np.reshape(next_states, [1, self.state_size])
        self.reward = np.append(self.reward, reward)

        print("Next state: ", next_states)
        self.remember(self.states, self.action, reward, next_states, is_done)

        self.states = next_states

        if self.iter > 32:
            self.replay(16)

        if self.iter%2 == 0 and self.iter != 0:
            self.copy_model()

        self.iter += 1

    def plotReward(self):
        plt.scatter([i for i in range(len(self.reward))], self.reward)
        plt.show()
