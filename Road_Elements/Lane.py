''' Class Lane:
        Lane represent a one line in a road which capable of holding vehicles
'''
import numpy as np

DEBUG = 3

class Lane:

    '''
        constructor for Lane
        args:
            direction: Direction of the lane
            state: lane is changing the direction. Initial state is IDLE
            capacity: number of vehicles to arrive or leave lane at a unit TIME STEP
    '''
    def __init__(self,  cap=10, id=0, direc='IN', state='IDLE'):
        assert (direc == 'IN' or direc == 'OUT')
        self.direction = direc
        self.state = state
        self.capacity = cap
        self.lane_id = id
        self.num_of_vehicles = 0

    '''
        Change direction of the lane.
        It take more than one time step to change the lane
    '''

    def change_direction(self,direc):
        self.direction = direc

    def get_capacity(self):
        return self.capacity

    def get_direction(self):
        return self.direction

    def  add_vehicle(self, num_veh):
        self.num_of_vehicles += num_veh

    def get_num_vehicles(self):
        return self.num_of_vehicles
