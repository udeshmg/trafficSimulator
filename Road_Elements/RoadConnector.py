import numpy as np
from Road_Elements.Road import Road

class RoadConnector:

    def __init__(self,id, numRoads):
        self.road = np.empty(shape=0, dtype=Road)
        self.intersectionID = id
        self.roadMap = np.array([1,0])
        self.num_roads = numRoads

    def addRoad(self, road):
        self.road = np.append(self.road, road)

    def step(self,action):

        for roadIndex in range(len(self.road)):
            if self.num_roads > 1:
                outRoadIndex = self.roadMap[roadIndex]

                if self.road[outRoadIndex].get_outgoing_vehicles(self.intersectionID) < 240:
                    vehiclesToRemove = self.road[roadIndex].capacity(self.intersectionID, 'IN')*2
                    total_wait, total_vehi, vbs, vbs_to_left = self.road[roadIndex].remove_block(vehiclesToRemove, 'S',
                                                                                     self.intersectionID, False,
                                                                                     False, False)
                    if len(vbs) != 0:
                        self.road[outRoadIndex].set_outgoing_vb(vbs, self.intersectionID)
                    total_wait, total_vehi, vbs, vbs_to_left = self.road[roadIndex].remove_block(vehiclesToRemove, 'R',
                                                                                     self.intersectionID, False,
                                                                                     False, False)
                    if len(vbs) != 0:
                        self.road[outRoadIndex].set_outgoing_vb(vbs, self.intersectionID)

            else:
                vehiclesToRemove = self.road[roadIndex].capacity(self.intersectionID, 'IN')*2
                total_wait, total_vehi, vbs, vbs_to_left = self.road[roadIndex].remove_block(vehiclesToRemove, 'S',
                                                                                             self.intersectionID, False,
                                                                                             False, False)
                for i in vbs:
                    i.finaliseRoute()

                vehiclesToRemove = self.road[roadIndex].capacity(self.intersectionID, 'IN')*2
                total_wait, total_vehi, vbs, vbs_to_left = self.road[roadIndex].remove_block(vehiclesToRemove, 'R',
                                                                                             self.intersectionID, False,
                                                                                             False, False)
                for i in vbs:
                    i.finaliseRoute()

    def getStates(self):
        pass

    def setIntersectionData(self):
        pass
