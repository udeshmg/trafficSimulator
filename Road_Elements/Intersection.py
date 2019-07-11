import numpy as np
from GUI.Display import Display
from collections import deque
from enum import Enum
import pandas as pd
from tkinter import *
from GUI.GUI import SimulatorTraffic
from collections import deque
from Road_Elements.Lane import Lane
from Road_Elements.Vehicles import VehicleBlock
from Road_Elements.Road import Road

alpha = 0.2 # reducing this diverge the results in the case of Multilane conf.
beta = 0.01
TIME_STEP = 10
DEBUG = 3
LANE_CAPACITY = 5
SIZE = 100
abs_SIZE = 300
maxVehicleGaP = 120
checkConsistent = 100
enable_check = False
GAP = 50
NUM_ROADS = 4
LENGTH = 5

ENABLE_DISPLAY = False
ENABLE_REPORTING = False



def info(level, a):
    if DEBUG < level:
        print(a)

que = deque(maxlen=LENGTH)

class Que():

    def __init__(self):
        self.que = deque(maxlen=LENGTH)

    def append(self, val):
        self.que.append(val)

    def popleft(self):
        out = self.que.popleft()
        return out

class Intersection:
    q_length_step_size = 6
    num_phases = 4
    num_roads = 4
    curr_waiting_time = 0
    curr_phase = 0
    dp = Display()
    if ENABLE_DISPLAY:
        gui = Tk()
        gui.geometry("600x600")
        gui.title("First title")
        env_sim = SimulatorTraffic(gui)

    def __init__(self, id):
        self.intersectionID = id
        self.road = np.empty(shape=0, dtype=Road)

        self.roadConnection = np.array([[2, 1, 3], [3, 2, 0], [0, 3, 1], [1, 0, 2]])

        self.average_qLen_buffer = np.empty(shape=NUM_ROADS*3, dtype=deque)
        self.average_qLen = np.zeros(shape=NUM_ROADS*3, dtype=float)

        for i in range(NUM_ROADS*3):
            self.average_qLen_buffer[i] = deque(maxlen=LENGTH)


        if ENABLE_DISPLAY:
            self.env_sim.simulator()

        self.gap_indicator = 1
        self.wait_time = 0
        self.vehicles = 0
        self.iter = 0
        self.highest_wait_time = deque(maxlen=10)
        self.waiting_time_que = deque(maxlen=SIZE)
        self.vehicle_num_que = deque(maxlen=SIZE)
        self.wait_time_buffer = np.empty(shape=0, dtype=float)
        self.throughput_buffer = np.empty(shape=0, dtype=float)
        self.disable_reset = True
        self.max_wait_time = 1000
        self.semi_queue_length = np.zeros(shape=4, dtype=int)
        self.lane_conf = np.zeros(shape=5)
        self.enable_check = False
        self.prev_wait_time = 0
        self.neg_reward = 0

        self.abs_time = 0
        self.abs_vehicles = 0
        self.abs_time_que = deque(maxlen=abs_SIZE)
        self.abs_vehicle_num_que = deque(maxlen=abs_SIZE)
        self.abs_time_buffer = np.empty(shape=0, dtype=float)
        self.abs_throughput_buffer = np.empty(shape=0, dtype=float)

        self.inst_time = 0
        self.inst_vehi = 0

        self.configMap = [[0,2,2,2,2]]
        self.enable_report = False
        self.laneChange = True

        self.vehicles_to_transfer = []

        self.request_buffer = []
        self.request_queue = []

        self.hold_buffer = np.zeros(shape=(4,2), dtype=int)
        self.allow_buffer = np.zeros(shape=(4,2), dtype=int)
        self.counter = 30
        self.allow_counter = 10

        self.local_view = True

    # step function advance the time in environment
    # When agent performs the action, step() method simulate
    # the environment

    def addRoad(self, road):
        self.road = np.append(self.road, road)
        print("Road added",self.intersectionID, self.road[-1].id)

    def setRoadConnection(self):
        # Road MAP
        self.road[0].set_straight_road(self.road[2])
        self.road[1].set_straight_road(self.road[3])
        self.road[2].set_straight_road(self.road[0])
        self.road[3].set_straight_road(self.road[1])

        self.road[0].set_right_road(self.road[1])
        self.road[1].set_right_road(self.road[2])
        self.road[2].set_right_road(self.road[3])
        self.road[3].set_right_road(self.road[0])

    def step(self, action):
        # transfer vehicles
        abs_time = 0
        abs_vehicles = 0
        while self.vehicles_to_transfer:
            road_details = self.vehicles_to_transfer.pop(0)
            tmp_a, tmp_b = self.road[road_details[1]].set_outgoing_vb(road_details[0],self.intersectionID)
            abs_time += tmp_a
            abs_vehicles += tmp_b


        #  Pre reward calculation

        self.prev_wait_time, tmp = self.total_wait_time()
        action_set = self.decompose_action(action)
        if self.enable_report:
            print(action_set)

        is_first = False
        if (self.curr_phase != action_set[0]):
            is_first = True
            #self.local_step(3)

        self.curr_phase = action_set[0]
        #print("Phase selected: ", action_set[0])

        ###  Execute actions : Begin
        if action_set[0] < 2:
            wait_time, vehicles = self.removeBlock(action_set[0], 'S', is_first)
            tmp_a, tmp_b        = self.removeBlock(action_set[0]+2, 'S', is_first)
            wait_time += tmp_a
            vehicles += tmp_b


        elif 2 <= action_set[0] and action_set[0] < 4:
            wait_time, vehicles = self.removeBlock(action_set[0], 'R', is_first)
            tmp_a, tmp_b        = self.removeBlock(action_set[0]-2, 'R', is_first)
            wait_time += tmp_a
            vehicles += tmp_b


        self.neg_reward = 0

        changed = False
        for index, iter in enumerate(action_set[1:]):
            if iter == 1:

                if self.road[index].get_traffic_imbalance(self.intersectionID) == 2 and self.road[index].get_change_counter() < 2:
                    if DEBUG > 1:
                        print("Change lane direction: road id", index, " Increase")
                    # check for whole network


                    if self.local_view:
                        if self.road[index].change_direction('IN', self.intersectionID):
                            changed = True

                    else:
                        self.sendLaneChangeRequest(
                            self.road[index].getChangeDirectionFromRoad('IN', self.intersectionID), self.road[index].id)

                self.neg_reward += 1
            elif iter == 2:

                if self.road[index].get_traffic_imbalance(self.intersectionID) == 1 and self.road[index].get_change_counter() < 2:
                    if DEBUG > 1:
                        print("Change lane direction: road id", index, " Decrease")



                    if self.local_view:
                        if self.road[index].change_direction('OUT', self.intersectionID):
                            changed = True

                    else:
                        self.sendLaneChangeRequest(
                            self.road[index].getChangeDirectionFromRoad('OUT', self.intersectionID),
                            self.road[index].id)

                self.neg_reward += 1

        if changed:
            self.configMap.append([self.iter, self.road[0].get_in_lanes_num(self.intersectionID),
                               self.road[1].get_in_lanes_num(self.intersectionID),
                               self.road[2].get_in_lanes_num(self.intersectionID),
                               self.road[3].get_in_lanes_num(self.intersectionID)])


        if not self.local_view:
            print("Using Guidance")
            for j in range(len(self.request_buffer)):
                id = self.get_local_road_id(self.request_buffer[j][0])
                self.road[id].change_direction(self.request_buffer[j][1], self.road[id].upstream_id)
            self.request_buffer.clear()

        for i in self.hold_buffer:
            if i[1] > 0:
                i[1] -= 1
            else:
                i[1] -= 0
                i[0] = 0

        for i in self.allow_buffer:
            if i[1] > 0:
                i[1] -= 1
            else:
                i[1] -= 0
                i[0] = 0

        ###  Execute actions : End

        # Remove outgoing traffic from a selected road at step
        #self.road[self.iter%4].remove_outgoing_vehicles(int(self.road[self.iter%4].capacity('OUT')*1.5))
        #self.road[(self.iter+2)%4].remove_outgoing_vehicles(int(self.road[(self.iter+2)%4].capacity('OUT')))


        self.highest_wait_time.append(wait_time/max(vehicles, 1))

        out_wait_highest = self.deque_highest_val(self.highest_wait_time)
        '''Figure ONE: wait_time per vehicle'''
        # self.dp.figure_time(wait_time/max(vehicles, 1), self.iter, 1)


        self.inst_time = self.moving_avg(wait_time,4,self.iter,self.waiting_time_que,self.wait_time)
        self.inst_vehi = self.moving_avg(vehicles,4,self.iter,self.vehicle_num_que,self.vehicles)

        self.wait_time = self.moving_avg(wait_time,SIZE,self.iter,self.waiting_time_que,self.wait_time)
        self.vehicles = self.moving_avg(vehicles,SIZE,self.iter,self.vehicle_num_que,self.vehicles)

        avg_wait_time = 0
        if self.vehicles != 0:
            avg_wait_time = self.wait_time / self.vehicles

        '''Figure TWO: vehicle throughput per time'''
        #self.dp.figure_time(self.vehicles, self.iter, 2)
        self.throughput_buffer = np.append(self.throughput_buffer, self.vehicles)
        '''Figure THREE: average vehicle wait time (last 1000 cycles)'''
        if self.iter > 0:
            #self.dp.figure_time(avg_wait_time, self.iter, 3)
            self.wait_time_buffer = np.append(self.wait_time_buffer,avg_wait_time)

        self.abs_time = self.moving_avg(abs_time, abs_SIZE, self.iter, self.abs_time_que, self.abs_time)
        self.abs_vehicles = self.moving_avg(abs_vehicles, abs_SIZE, self.iter, self.abs_vehicle_num_que, self.abs_vehicles)

        self.abs_time_buffer = np.append(self.abs_time_buffer, self.abs_time)
        self.abs_throughput_buffer = np.append(self.abs_throughput_buffer, self.abs_vehicles)

        self.iter += 1
        vehicles_list = np.array([self.road[0].get_num_vehicles(self.intersectionID, 'R'),
                                  self.road[1].get_num_vehicles(self.intersectionID, 'R'),
                                  self.road[2].get_num_vehicles(self.intersectionID, 'R'),
                                  self.road[3].get_num_vehicles(self.intersectionID, 'R'),
                                  self.road[0].get_num_vehicles(self.intersectionID, 'S'),
                                  self.road[1].get_num_vehicles(self.intersectionID, 'S'),
                                  self.road[2].get_num_vehicles(self.intersectionID, 'S'),
                                  self.road[3].get_num_vehicles(self.intersectionID, 'S')])

        state_vector = []
        #state_vector.append(self.curr_phase)
        for i in range(0, 4):

            state_vector.append(min(50, int(vehicles_list[i+4])))
            state_vector.append(min(50, int(vehicles_list[i])))
            #state_vector.append(self.map_state(self.road[i].get_num_vehicles('S')))
            state_vector.append(int(self.road[i].get_outgoing_vehicles(self.intersectionID)))

        #if self.enable_report:
        #    print("Intersection id", self.intersectionID, state_vector)

        ### Statistics and Graph usage END###

        #self.update_env()


    def getStates(self):
        vehicles_list = np.array([self.road[0].get_num_vehicles(self.intersectionID, 'R'), self.road[1].get_num_vehicles(self.intersectionID, 'R'),
                                  self.road[2].get_num_vehicles(self.intersectionID, 'R'), self.road[3].get_num_vehicles(self.intersectionID, 'R'),
                                  self.road[0].get_num_vehicles(self.intersectionID, 'S'), self.road[1].get_num_vehicles(self.intersectionID, 'S'),
                                  self.road[2].get_num_vehicles(self.intersectionID, 'S'), self.road[3].get_num_vehicles(self.intersectionID, 'S')])

        imbalance = max(vehicles_list) - min(vehicles_list)

        next_wait_time, time_gap = self.total_wait_time()

        if next_wait_time > self.max_wait_time:
            self.max_wait_time = next_wait_time

        if(self.prev_wait_time == 0) & (next_wait_time == 0):
            reward = 0
        elif self.neg_reward > 0:
            reward = -1
        else:
            reward = (1 - alpha)*((self.prev_wait_time - next_wait_time)/max(self.prev_wait_time,next_wait_time) - next_wait_time/self.max_wait_time) - alpha*(imbalance/maxVehicleGaP)
            #reward = (prev_wait_time - next_wait_time + (
            #            total_vehicles * TIME_STEP)) - alpha * next_wait_time - time_gap * beta
        #reward = pre_reward - reward

        if DEBUG:
            print("Reward given : ", reward)
            print("Time : ", next_wait_time, self.prev_wait_time)

        done = False
        if not self.disable_reset:

            if self.iter%100 == 0:
                self.reset()
                done = True
                reward = 0
            else:
                done = False


        if (self.iter <= LENGTH):
            for i in range(NUM_ROADS * 2):
                self.average_qLen_buffer[i].append(vehicles_list[i])
                temp = 0
                for j in range(self.iter):
                    temp += self.average_qLen_buffer[i][j]
                self.average_qLen[i] = temp / self.iter


            for i in range(NUM_ROADS):
                self.average_qLen_buffer[i+8].append(self.road[i].get_outgoing_vehicles(self.intersectionID))
                temp = 0
                for j in range(self.iter):
                    temp += self.average_qLen_buffer[i+8][j]
                self.average_qLen[i+8] = temp / self.iter
        else:
            for i in range(NUM_ROADS*2):
                out = self.average_qLen_buffer[i].popleft()
                self.average_qLen_buffer[i].append(vehicles_list[i])
                self.average_qLen[i] += (vehicles_list[i]-out)/LENGTH

            for i in range(NUM_ROADS):
                out = self.average_qLen_buffer[i+8].popleft()
                temp = self.road[i].get_outgoing_vehicles(self.intersectionID)
                self.average_qLen_buffer[i+8].append(temp)
                self.average_qLen[i+8] += (temp-out)/LENGTH


        plt_vector = []
        state_vector = []
        #state_vector.append(self.curr_phase)
        for i in range(0, 4):

            state_vector.append(min(50, int(vehicles_list[i+4]/self.road[i].get_in_lanes_num(self.intersectionID))))
            state_vector.append(min(50, int(vehicles_list[i]/self.road[i].get_in_lanes_num(self.intersectionID))))
            if ENABLE_DISPLAY:
                plt_vector.append(vehicles_list[i+4])
                plt_vector.append(vehicles_list[i])
            #state_vector.append(self.map_state(self.road[i].get_num_vehicles('S')))
            if self.laneChange:
                state_vector.append(int(self.road[i].get_outgoing_vehicles(self.intersectionID)/(self.road[i].num_of_lanes - self.road[i].get_in_lanes_num(self.intersectionID))))
        if self.enable_report:
            print("Intersection id", self.intersectionID, state_vector)
        for i in range(0,4):
            state_vector.append(self.road[i].get_in_lanes_num(self.intersectionID)-1)

        for i in range(0,4):
            state_vector.append(self.road[i].get_traffic_imbalance(self.intersectionID))


        #Figure FOUR: Current state
        if ENABLE_DISPLAY:
            self.env_sim.vehicle_lane(self.road[0].straight_v_list, 'S', 'W')
            self.env_sim.vehicle_lane(self.road[0].right_Turn_v_list, 'R', 'W')
            self.env_sim.vehicle_lane(self.road[1].straight_v_list, 'S', 'S')
            self.env_sim.vehicle_lane(self.road[1].right_Turn_v_list, 'R', 'S')
            self.env_sim.vehicle_lane(self.road[2].straight_v_list, 'S', 'E')
            self.env_sim.vehicle_lane(self.road[2].right_Turn_v_list, 'R', 'E')
            self.env_sim.vehicle_lane(self.road[3].straight_v_list, 'S', 'N')
            self.env_sim.vehicle_lane(self.road[3].right_Turn_v_list, 'R', 'N')
            #self.env_sim.update_iter(self.iter,avg_wait_time,self.vehicles, action_set[0])
            self.env_sim.lane_conf(state_vector[12:16])

        #  self.dp.plot(plt_vector)
        #  self.dp.multi_scatter_plot(plt_vector, self.iter, 4)


        return self.curr_phase, state_vector, reward, done

    # Update the environment with some vehicles in
    # a random fashion

    def update_env(self):  # add vehicles to environment
        if (self.iter < 34000 and self.iter > 25000) or (self.iter < 55000 and self.iter > 45000):
            random_vehicles = np.array([np.random.poisson(7, 1)[0], np.random.poisson(4, 1)[0], # 'S' | 'R'
                                    np.random.poisson(2, 1)[0], np.random.poisson(1, 1)[0],
                                    np.random.poisson(4, 1)[0], np.random.poisson(2, 1)[0],
                                    np.random.poisson(6, 1)[0], np.random.poisson(1, 1)[0]])
                          # poison distribution of vehicles in road network

        else:
            random_vehicles = np.array([np.random.poisson(2, 1)[0], np.random.poisson(2, 1)[0], # 'S' | 'R'
                                    np.random.poisson(3, 1)[0], np.random.poisson(2, 1)[0],
                                    np.random.poisson(6, 1)[0], np.random.poisson(1, 1)[0],
                                    np.random.poisson(4, 1)[0], np.random.poisson(2, 1)[0]])

            '''if self.iter > 20000:
                random_vehicles = np.array([np.random.poisson(3, 1)[0], np.random.poisson(4, 1)[0],  # 'S' | 'R'
                                        np.random.poisson(6, 1)[0], np.random.poisson(3, 1)[0],
                                        np.random.poisson(4, 1)[0], np.random.poisson(2, 1)[0],
                                        np.random.poisson(2, 1)[0], np.random.poisson(1, 1)[0]])'''

        random_vehicles = random_vehicles.round()
        # random_vehicles = [1,2,1,2]
        np.clip(random_vehicles, 0, 40)

        for road_index in range(0, 4):
            self.road[road_index].add_block(VehicleBlock(int(random_vehicles[2*road_index]),['S']), 'UP')
            self.road[road_index].add_block(VehicleBlock(int(random_vehicles[2*road_index+1]),['R']), 'UP')

    # Calculate total wait time of all roads at the intersection
    def total_wait_time(self):
        total_time = 0
        min_time = self.road[0].get_wait_time('S', self.intersectionID)
        max_time = 0
        for i in range(0, self.num_roads):
            total_time += self.road[i].get_wait_time('S', self.intersectionID)
            total_time += self.road[i].get_wait_time('R', self.intersectionID)

            if self.road[i].get_wait_time('S',self.intersectionID) < min_time:
                min_time = self.road[i].get_wait_time('S', self.intersectionID)
            if self.road[i].get_wait_time('S',self.intersectionID) > max_time:
                max_time = self.road[i].get_wait_time('S', self.intersectionID)
            if self.road[i].get_wait_time('R',self.intersectionID) < min_time:
                min_time = self.road[i].get_wait_time('R', self.intersectionID)
            if self.road[i].get_wait_time('R',self.intersectionID) > max_time:
                max_time = self.road[i].get_wait_time('R', self.intersectionID)

        return total_time, max_time - min_time


    def removeBlock(self, roadIndex, direction, is_first=True):
        assert (direction == 'R' or direction == 'S')
        total_wait = 0
        total_vehi = 0
        temp = 0

        left_block = False
        straight_block = False
        left_road = self.roadConnection[roadIndex][2]
        if direction == 'S':
            outRoadIndex = self.roadConnection[roadIndex][0]

            out_tf = self.road[outRoadIndex].get_outgoing_vehicles(self.intersectionID)
            out_tf_l = self.road[left_road].get_outgoing_vehicles(self.intersectionID)

            if out_tf >= 240 and out_tf_l >= 240:
                print("Both lanes are full ", self.road[outRoadIndex].get_id(), self.road[left_road].get_id())
                return 0,0

            if out_tf >= 240:
                straight_block = True
                print("Output straight lane is full", self.road[outRoadIndex].get_id())

            if out_tf_l >= 240:
                left_block = True
                print("Output left lane is full", self.road[left_road].get_id())

        else:
            outRoadIndex = self.roadConnection[roadIndex][1]
            out_tf_r = self.road[outRoadIndex].get_outgoing_vehicles(self.intersectionID)
            if out_tf_r >= 240:
                print("Output right lane is full",  self.road[outRoadIndex].get_id())
                return 0,0






        if self.enable_check:
            if (direction == 'S'):
                vehiclesToRemove = min(self.road[roadIndex].capacity('IN'), int(1.5 * self.road[outRoadIndex].capacity('OUT')))  # remain capacity
            else:
                vehiclesToRemove = min(self.road[roadIndex].capacity('IN'), self.road[outRoadIndex].capacity('OUT'))
        else:
            vehiclesToRemove = self.road[roadIndex].capacity(self.intersectionID, 'IN')

        # save the below data to add after the all step() methods are executed by the agents
        total_wait, total_vehi, vbs, vbs_to_left = self.road[roadIndex].remove_block(vehiclesToRemove, direction,
                                                            self.intersectionID, is_first, straight_block, left_block)

        self.road[outRoadIndex].set_outgoing_traffic(total_vehi)
        '''time, vehicles = self.road[outRoadIndex].set_outgoing_vb(vbs, self.intersectionID)


        if vbs_to_left.size != 0:
            temp_a, temp_b = self.road[left_road].set_outgoing_vb(vbs_to_left, self.intersectionID)
            #self.abs_time_que.append(time)
            #self.abs_vehicle_num_que.append(vehicles)
            time += temp_a
            vehicles += temp_b'''

        if vbs.size != 0:
            self.vehicles_to_transfer.append([vbs,outRoadIndex])
        if vbs_to_left.size != 0:
            self.vehicles_to_transfer.append([vbs_to_left,left_road])

        # save these data:
            # vbs, vbs_to_left



        return total_wait, total_vehi #, time, vehicles


    def sendLaneChangeRequest(self, action, rid):
        isPresent = False
        for i in range(len(self.request_queue)):
            if self.request_queue[i][1] == rid:
                isPresent = True
        if not isPresent:
            self.request_queue.append([self.intersectionID, rid,action])


    def enableChange(self, rid, action):
        self.request_buffer.append([rid, action])

    def holdChange(self, rid, action):
        id = self.get_local_road_id(rid)

        if self.road[id].upstream_id == self.intersectionID:
            if action == 'OUT':
                self.hold_buffer[id] = [2, self.counter]
            else:
                self.hold_buffer[id] = [1, self.counter]

        else:
            if action == 'OUT':
                self.hold_buffer[id] = [1, self.counter]
            else:
                self.hold_buffer[id] = [2, self.counter]

    def allow_change(self, action, rid):
        print("Allow additional intersection change")
        id = self.get_local_road_id(rid)
        if action == 'OUT':
            self.allow_buffer[id] = [2, self.allow_counter]
        else:
            self.allow_buffer[id] = [1, self.allow_counter]

    def getAllowChange(self, rid):
        if self.allow_buffer[rid][1] > 0:
            return self.allow_buffer[rid][0]
        else:
            return 0

    def getChangeAllow(self, rid):
        if self.hold_buffer[rid][1] > 0:
            return self.hold_buffer[rid][0]
        else:
            return 0



    def enable_reset(self):
        self.disable_reset = False

    def disable_reset(self):
        self.disable_reset = True

    def reset(self):
        for i in self.road:
            i.reset()

    # internal step function for each road with given time step
    def local_step(self, time_step):
        for i in range(0, self.num_roads):
            self.road[i].step(time_step)

    def get_local_road_id(self, rid):
        for i in range(len(self.road)):
            if (rid == self.road[i].id):
                return i
        print("ID not found")
        return -1


    def printArrayData(self):
        print(self.lane_conf)
        self.dp.single_arr(self.wait_time_buffer,self.intersectionID)
        self.dp.single_arr(self.throughput_buffer,self.intersectionID,'r')
        print(np.array(self.configMap))
        self.dp.show()


    def writeTofile(self, filename='default_file.csv'):
        df = pd.DataFrame({'wait_time': self.wait_time_buffer})
        df.to_csv(filename+'_wait_time.csv')
        df = pd.DataFrame({'throughput': self.throughput_buffer})
        df.to_csv(filename+'_throughput.csv')


    @staticmethod
    def deque_highest_val(dq):
        max_dq = 0
        for i in dq:
            if i > max_dq:
                max_dq = i
        return max_dq

    @staticmethod
    def moving_avg(dat, size, iter, moving_que, moving_avg):
        prev_avg = moving_avg
        if iter < size:
            moving_que.append(dat)
            moving_avg = 0
            for j in range(iter+1):
                moving_avg += moving_que[j]
            moving_avg = moving_avg / (iter + 1)
        else:
            out = moving_que.popleft()
            moving_que.append(dat)
            temp = 0
            for j in range(size):
                temp += moving_que[j]
            temp /= size
            moving_avg = temp

        return moving_avg

    @staticmethod
    def map_state(val):
        if val < 200:
            return 0
        elif val < 800:
            return 1
        elif val < 1600:
            return 2
        elif val < 3200:
            return 3
        else:
            return 4

    @staticmethod
    def decompose_action(action_val):
        action = action_val
        # Internal params
        action_set = np.array([4, 3, 3, 3, 3])
        decomposed_action = np.zeros(shape=len(action_set)).astype(int)
        for i in range(len(action_set)):
            decomposed_action[-i - 1] = action % action_set[-i - 1]
            action = int(action / action_set[-i - 1])
        return decomposed_action

    def getDequeSum(self, deque):
        sum = 0
        for elem in deque:
            sum += elem
        return sum




