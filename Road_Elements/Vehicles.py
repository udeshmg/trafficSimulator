import numpy as np

class VehicleBlock:

    '''
    Constructor of the Vehicle block

        args:
            num_vehicles : number of vehicles to add
            direc : 'R' or 'S' Right or Straight
        returns:
            none
        Dat:
            num_vehicles
            current_time
            turn_direction
    '''
    def __init__(self, num_vehicles, direc, id = 0, debugLevel=2):
        self.num_vehicles = num_vehicles
        self.current_time = 0
        self.abs_time = 0
        self.id = id

        self.turn_direction = np.array(direc)
        self.debugLevel = debugLevel

        if self.debugLevel > 2:
            print("Generated id", self.id, " Path ", self.turn_direction)

    '''
        advances the time of vehicles in the block
        args:
            time_step : time that vehicle block should advanced
        returns:
            current time of the vehicle block for advancing the time step
    '''

    def setDebugLevel(self, level):
        self.debugLevel = level

    def step(self, time_step):
        self.current_time += time_step
        self.abs_time += time_step
        return self.current_time

    '''
        Provides the total wait time of the vehicle block
        args:
            none
        returns:
            Returns the total time of all vehicles in the block    
    '''

    def waiting_time(self):
        if self.debugLevel  > 3:
            print("Vehicle block: ", self.num_vehicles, self.current_time)
        return self.num_vehicles * self.current_time

    '''
        Remove vehicles from the block
        args:
            num_vehicles : number of vehicles to remove from the block
        returns
            none
    '''

    def reduce_vehicles(self, num_vehicles):
        self.num_vehicles -= num_vehicles

    '''
        returns the number of vehicles in the vehicle block
        args:
            none
        returns:
            num_vehicles: number of vehicles
    '''

    def get_num_vehicles(self):
        return self.num_vehicles

    def get_direction(self):
        return self.turn_direction[0]

    def get_route(self):
        return self.turn_direction

    def update_direction(self):
        if self.turn_direction.size != 0:
            self.turn_direction = np.delete(self.turn_direction, 0)

    def resetTime(self):
        self.current_time = 0

    def get_current_time(self):
        return self.current_time