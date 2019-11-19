from Stat_Reporter.StatReporter import Reporter
import numpy as np
import copy
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
    def __init__(self, num_vehicles, direc, id = 0, debugLevel=2, path=None):
        self.num_vehicles = num_vehicles
        self.current_time = 0
        self.abs_time = 0
        self.id = id
        self.currentRoadId = 0
        self.currentDirInRoad = 'UP'
        self.freeFlowTime = 0
        self.turn_direction = np.array(direc)
        self.turn_direction = np.append(self.turn_direction, 'S')
        self.vertexList = np.array(path)
        self.debugLevel = debugLevel
        self.reporter = Reporter.getInstance()
        self.routeFinished = False

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

        if self.debugLevel > 2:
            print("Vehicle ID: ", self.id, " time:", self.abs_time, end = ''),
            print(" Road id: ", self.currentRoadId, " Dir:", self.currentDirInRoad, end = '')
            print(" Route: ", self.turn_direction)

        return self.current_time

    def absStep(self, time_step):
        self.abs_time += time_step

        if self.debugLevel > 2:
            print(" ABS: Vehicle ID: ", self.id, " time:", self.abs_time, end = ''),
            print(" ABS: Road id: ", self.currentRoadId, " Dir:", self.currentDirInRoad, end = '')
            print(" ABS: Route: ", self.turn_direction)
            print(" Free flow time: ", self.freeFlowTime)

        return self.current_time

    '''
        Provides the total wait time of the vehicle block
        args:
            none
        returns:
            Returns the total time of all vehicles in the block    
    '''

    def setCurrentRoadDetails(self,rid,dir='UP'):
        self.currentRoadId = rid
        self.currentDirInRoad = dir

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

    def finaliseRoute(self):
        self.routeFinished = True
        self.reporter.addVehicleData([[self.id, self.abs_time, self.freeFlowTime]])

    def get_num_vehicles(self):
        return self.num_vehicles

    def get_direction(self):
        return self.turn_direction[0]

    def get_route(self):
        return self.turn_direction

    def update_direction(self):
        if self.turn_direction.size != 0:
            self.turn_direction = np.delete(self.turn_direction, 0)

        if self.vertexList.size != 0:
            self.vertexList = np.delete(self.vertexList, 0)

    def resetTime(self):
        self.current_time = 0

    def get_current_time(self):
        return self.current_time