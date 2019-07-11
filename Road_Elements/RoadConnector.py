import numpy as np
from Road_Elements.Road import Road

class RoadConnector:

    def __init__(self,id, numRoads):
        self.road = np.empty(shape=0, dtype=Road)
        self.intersectionID = id
        self.roadMap = np.array([1,0])
        self.numRoads = numRoads

    def addRoad(self, road):
        self.road = np.append(self.road, road)

    def step(self):

        for roadIndex in range(len(self.road)):
            if self.numRoads > 1:
                outRoadIndex = self.roadMap[roadIndex]

                if self.road[outRoadIndex].get_outgoing_vehicles(self.intersectionID) < 240:
                    vehiclesToRemove = self.road[roadIndex].capacity(self.intersectionID, 'IN')
                    total_wait, total_vehi, vbs, vbs_to_left = self.road[roadIndex].remove_block(vehiclesToRemove, 'S',
                                                                                     self.intersectionID, False,
                                                                                     False, False)
                    total_wait, total_vehi, vbs, vbs_to_left = self.road[roadIndex].remove_block(vehiclesToRemove, 'R',
                                                                                     self.intersectionID, False,
                                                                                     False, False)
                    self.road[outRoadIndex].set_outgoing_traffic(total_vehi)
            else:
                vehiclesToRemove = self.road[roadIndex].capacity(self.intersectionID, 'IN')
                total_wait, total_vehi, vbs, vbs_to_left = self.road[roadIndex].remove_block(vehiclesToRemove, 'S',
                                                                                             self.intersectionID, False,
                                                                                             False, False)
                vehiclesToRemove = self.road[roadIndex].capacity(self.intersectionID, 'IN')
                total_wait, total_vehi, vbs, vbs_to_left = self.road[roadIndex].remove_block(vehiclesToRemove, 'R',
                                                                                             self.intersectionID, False,
                                                                                             False, False)
