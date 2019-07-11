"""Training the agent"""
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as k
import tensorflow as tf
import sys

# q_table = np.zeros([env.observation_space.n, env.action_space.n])

# implement a queue table here for a single intersection


NEGATIVE_REWARD = 2
SAVE_N_RESTORE = False
SAVE = True
DEBUG = 3

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

class DQNAgent:
    def __init__(self, state_size, action_size,intersection, num_roads, id=1, lanechange=True, guided=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100)
        self.gamma = 0.75    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99 #0.995
        self.learning_rate = 0.001 #0.001

        self.model, self.target_model = self._build_model()
        self.iter = 0
        self.intersection = intersection
        self.num_roads = num_roads

        self.guided = guided

        self.laneChange = lanechange
        if self.laneChange:
            self.curr_road_loads =  np.zeros(shape=num_roads*5)
        else:
            self.curr_road_loads =  np.zeros(shape=num_roads*2)


        self.curr_phase = 0
        self.states = np.zeros(shape=(1, self.state_size))
        self.action = 0
        self.id = id


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(24, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                    optimizer=Adam(lr=self.learning_rate))

        target_model = Sequential()
        target_model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        target_model.add(Dense(24, activation='tanh'))
        target_model.add(Dense(self.action_size, activation='linear'))
        target_model.compile(loss='mse',
                    optimizer=Adam(lr=self.learning_rate))


        model._make_predict_function()
        target_model._make_predict_function()


        return model, target_model

    def copy_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print("Rondom Action ", self.id)

            if (self.laneChange and self.guided):
                action = random.randrange(self.num_roads)
                return action * (3**self.num_roads)
            else:
                return random.randrange(self.action_size)

        #act_values = self.ignore_invalid_actions(self.model.predict(state)[0], state)
        act_values = self.model.predict(state)[0]
        #print(act_values)
        #print("Phase selection", act_values[0], act_values[81],act_values[162],act_values[243])

        if self.laneChange:
            for i in range(len(act_values)):
                out = decompose_action(i,self.num_roads)

                imb_index = self.num_roads*4
                conf_index = self.num_roads*3

                for j in range(len(out)-1):
                    if (out[j+1] != 0 and state[0][imb_index+j] == 0) or \
                       (out[j+1] != 0 and state[0][conf_index+j] == 3 and state[0][imb_index+j] == 2) or \
                       (out[j+1] != 0 and state[0][conf_index+j] == 1 and state[0][imb_index+j] == 1):
                        act_values[i]  -= 100
                    elif out[j+1] == self.intersection.getChangeAllow(j) and out[j+1] != 0:
                        act_values[i]  -= 100

                    if out[j+1] == self.intersection.getAllowChange(j) and out[j+1] != 0:
                        print("Allowed the change")
                        act_values[i]  += 100

        return np.argmax(act_values)  # returns action

    def replay(self, batch_size):
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:

                target = self.model.predict(state)
                if not done:
                    max_action = np.argmax(self.model.predict(next_state)[0])
                    t = self.target_model.predict(next_state)[0]
                    target[0][action] = reward + self.gamma * t[max_action]


                self.model.fit(state, target, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


    def load(self, name):
            if self.laneChange and self.guided:
                self.epsilon = 0.2
            else:
                self.epsilon = 0.2
            self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def ignore_invalid_actions(self, action_set, state):
        for index, iter in enumerate(action_set):
            is_invalid = False
            separated_actions = decompose_action(index, self.num_roads)
            for i,j in enumerate(separated_actions[1:]):
                if (j != 0 and state[0][13+i] != 0):
                    is_invalid = True
                    if DEBUG > 4:
                        print("Invalid move found", separated_actions, state[0])
            if(is_invalid):
                action_set[index] -= NEGATIVE_REWARD
        if DEBUG > 4:
            print("Modified action set", action_set)
        return action_set

    def actOnIntersection(self):
        self.states = np.insert(self.curr_road_loads, 0, self.curr_phase)
        self.states = np.reshape(self.states, [1, self.state_size])
        self.action = self.act(self.states)

        if self.laneChange:
            print("action: ", self.id, decompose_action(self.action,self.num_roads))
            self.intersection.step(self.action)
        else:
            if self.num_roads == 4:
                print("action: ", self.id, decompose_action(self.action*81,self.num_roads))
                self.intersection.step(self.action*81)
            elif self.num_roads == 3:
                print("action: ", self.id, decompose_action(self.action*27,self.num_roads))
                self.intersection.step(self.action*27)


    def getOutputData(self):

        next_phase, next_road_loads, reward, is_done = self.intersection.getStates()
        if self.laneChange:
            next_states = np.insert(next_road_loads, 0, next_phase)
        else:
            next_states = np.insert(next_road_loads[0:8], 0, next_phase)

        if DEBUG > 2:
            print("Agent", self.id, "Next state: ", next_states)

        if not self.laneChange:
            print("Road Config: ", next_road_loads[8:12])

        next_states = np.reshape(next_states, [1, self.state_size])


        self.remember(self.states, self.action, reward, next_states, is_done)

        self.curr_phase = next_phase
        if self.laneChange:
            self.curr_road_loads = next_road_loads
        else:
            self.curr_road_loads = next_road_loads[0:8]

        if self.iter > 32:
            self.replay(16)


        if self.iter%2 == 0 and self.iter != 0:
            self.copy_model()


        self.iter += 1

'''
DEBUG = 3

phases = 4
num_roads = 4
vehicle_step = 5
actions = phases


action = np.zeros(phases)

#  Parameters for Q function

traffic_env = Intersection(id)
#traffic_env.enable_reset()

# updating values
curr_phase = 0
curr_road_loads = np.zeros(shape=num_roads*2+12)
next_road_loads = np.zeros(shape=num_roads*2+12)
signal_pattern = np.zeros(shape=(1,100),dtype=np.int)
moving_avg = 0
moving_que = deque(maxlen=100)
dp = Display()

iterations = 25000

state_size = 21
action_size = 324
agent = DQNAgent(state_size,action_size)

if SAVE_N_RESTORE:
    agent.load("weights")

reward_buffer = np.empty(shape=0, dtype=float)



for i in range(0, iterations):
    print(i)

    states = np.insert(curr_road_loads, 0, curr_phase)
    states = np.reshape(states, [1, state_size])
    action = agent.act(states)

    next_phase, next_road_loads, reward, is_done = traffic_env.step(action)
    reward_buffer =  np.append(reward_buffer, reward)
    next_states = np.insert(next_road_loads, 0, next_phase)
    if DEBUG > 2:
        print("Next state: ", next_states)
    next_states = np.reshape(next_states, [1, state_size])

    agent.remember(states, action, reward, next_states, is_done)

    curr_phase = next_phase
    curr_road_loads = next_road_loads


    if i > 50:
        agent.replay(32)

# moving average **** TODO: Declare a separate function

    if SAVE:
        agent.save("weights")

    # dp.figure(moving_avg, i)
    # moving average ****
if SAVE:
    agent.save("weights")
traffic_env.writeTofile('Multilane_2')
traffic_env.printArrayData()
dp.single_arr(reward_buffer)
#dp.colorMap(signal_pattern)

print("Training finished.\n") '''


