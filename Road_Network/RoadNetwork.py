from OSM.OSMgraph import OsmGraph
from Road_Elements.Lane import Lane
from Road_Elements.Road import Road
from Road_Elements.Vehicles import VehicleBlock
from Road_Elements.Intersection import Intersection
from Road_Network.RoadElementGenerator import RoadElementGenerator
from Road_Network.VehicleGenerator import VehicleGenerator
from Traffic_loader.TrafficCollector import TrafficCollector
from Traffic_loader.TrafficGenerator import TrafficGenerator
from Stat_Reporter.StatReporter import Reporter
from Dependency_Graph.DependencyGraph import DependencyGraph
import numpy as np
import time
class RoadNetwork:

    def __init__(self):
        self.osmGraph = None
        self.roadElementGenerator = RoadElementGenerator()
        self.vehicleGenerator = VehicleGenerator()
        self.trafficLoader = TrafficCollector()
        self.trafficGenerator = TrafficGenerator()
        self.timeStep = 10
        self.dependencyG = DependencyGraph()
        self.depth = 2
        self.threshold = 12
        self.load_th = 0
        self.recentPaths = 30
        self.numOfVehiclesPerBlock = 1
        self.autoGenTrafficEnabled = False
        self.demandCalculationFreq = 10 #mins
        self.manualAllocate = False
        self.depFreq = 2

        self.demandBuffer = []
        self.rand = []
        for i in range(30):
            self.rand.append(np.random.randint(5))


    def buildGraph(self, long, lat, diameter, osm=True):
        print("Loading data from Open street maps...")
        self.osmGraph = OsmGraph(long=long, lat=lat, diameter=diameter, osm=osm)
        self.osmGraph.createAdjacenceEdgeAttribute()
        print("Map load complete")

    def init(self):
        self.roadElementGenerator.getGraph(self.osmGraph)
        self.vehicleGenerator.getGraph(self.osmGraph)
        if not self.autoGenTrafficEnabled:
            self.trafficLoader.loadUserData()


    def createNetwork(self,agentString=""):
        print("Generating Roads Elements...")
        self.roadElementGenerator.connectElements()
        print("Roads Elements generated.")

    def generateVehicles(self, source, path, numOfVehicles=1, debugLvl=3, indexAtfile=0, time=0):
        for i in range(numOfVehicles):
            vb, startEdge = self.vehicleGenerator.generateVehicleWithId(path, debugLvl, indexAtfile)
            if vb != None or startEdge != None:
                rid = self.osmGraph.nxGraph[startEdge[0]][startEdge[1]]['edgeId']

                if self.roadElementGenerator.roadList[rid - 1].upstream_id == source:
                    self.roadElementGenerator.roadList[rid - 1].add_block(vb, 'DOWN')
                else:
                    self.roadElementGenerator.roadList[rid - 1].add_block(vb, 'UP')

                if self.manualAllocate:
                    self.demandBuffer.append(vb.vertexList)




        #print(self.roadElementGenerator.roadList[rid - 1].get_num_vehicles(self.roadElementGenerator.roadList[rid - 1].downstream_id,'T'))

    def addTrafficFromData(self,step):


        if not self.autoGenTrafficEnabled:
            vehicleList = self.trafficLoader.getData(step)
            nodeList = self.osmGraph.getNearestNode(vehicleList)
            for n in nodeList:
                self.generateVehicles(n[0],n[2],self.numOfVehiclesPerBlock, 2, n[3], step)
        else:


            for index, i in enumerate(self.trafficGenerator.getTrafficData()):
                n = self.osmGraph.getPathFromNodes(i[0],i[1],i[2])
                print("Traffic Data Added: ", n[0],n[2],self.rand[index])
                self.generateVehicles(n[0],n[2],i[2], 2, step)


 # source and destination

    def simulateOneStep(self, stepNumber):

        if stepNumber % 30:
            self.rand.clear()
            for i in range(30):
                self.rand.append(np.random.randint(5))

        if (stepNumber%self.demandCalculationFreq == 0 ) and self.manualAllocate: #mins
            self.setRoadconfigPath()
            self.demandBuffer.clear()

        for i in range(len(self.roadElementGenerator.agentList)):
            self.roadElementGenerator.agentList[i].actOnIntersection()

        for i in range(len(self.roadElementGenerator.roadList)):
            self.roadElementGenerator.roadList[i].step(10)

        if self.roadElementGenerator.isGuided:
            self.dependencyCheck(stepNumber)

        if self.roadElementGenerator.selfLaneChange and self.roadElementGenerator.enableDependencyCheck:
            self.dependencyCheckRoad(stepNumber)

        for i in range(len(self.roadElementGenerator.agentList)):
            self.roadElementGenerator.agentList[i].getOutputData()

    def setRoadconfig(self, val):
        roadChanges = self.osmGraph.allocateLaneBasedOnLoad()
        for i in range(len(roadChanges)):
            if roadChanges[i][1] == 1:
                self.roadElementGenerator.roadList[roadChanges[i][0]-1].change_direction('IN',
                                                                                      self.roadElementGenerator.roadList[roadChanges[i][0]-1].upstream_id,
                                                                                      val)
            else:
                self.roadElementGenerator.roadList[roadChanges[i][0] - 1].change_direction('OUT',
                                                                                            self.roadElementGenerator.roadList[
                                                                                                roadChanges[i][
                                                                                                    0] - 1].upstream_id,
                                                                                            val)

    def setRoadconfigPath(self):
        #roadChanges = self.osmGraph.allocateLaneBasedOnPaths()
        roadChanges = self.osmGraph.allocateLaneBasedOnLoadTime(self.demandBuffer)

        for i in range(len(roadChanges)):
            if roadChanges[i][1] == 1:
                if self.roadElementGenerator.roadList[roadChanges[i][0]-1].get_in_lanes_num(
                        self.roadElementGenerator.roadList[roadChanges[i][0]-1].upstream_id) == 2:
                    self.roadElementGenerator.roadList[roadChanges[i][0]-1].change_direction('IN',
                                                                                      self.roadElementGenerator.roadList[
                                                                                          roadChanges[i][0]-1].upstream_id,
                                                                                      1)

                else:
                    self.roadElementGenerator.roadList[roadChanges[i][0]-1].change_direction('IN',
                                                                                      self.roadElementGenerator.roadList[
                                                                                          roadChanges[i][0]-1].upstream_id,
                                                                                      0)
            else:
                if self.roadElementGenerator.roadList[roadChanges[i][0] - 1].get_in_lanes_num(
                        self.roadElementGenerator.roadList[roadChanges[i][0] - 1].upstream_id) == 4:
                    self.roadElementGenerator.roadList[roadChanges[i][0] - 1].change_direction('OUT',
                                                                                               self.roadElementGenerator.roadList[
                                                                                                   roadChanges[i][
                                                                                                       0] - 1].upstream_id,
                                                                                               1)

                else:
                    self.roadElementGenerator.roadList[roadChanges[i][0] - 1].change_direction('OUT',
                                                                                               self.roadElementGenerator.roadList[
                                                                                                   roadChanges[i][
                                                                                                       0] - 1].upstream_id,
                                                                                               0)

    def dependencyCheck(self,stepNumber):
        if stepNumber % self.depFreq == 0:
            #self.osmGraph.SDpaths = self.osmGraph.SDpaths[-20:len(self.osmGraph.SDpaths)]
            #self.dependencyG.createVariableDAG(self.osmGraph.nxGraph, self.osmGraph.SDpaths)

            #paths = list(self.osmGraph.recentPaths)
            paths  = []
            for i in self.roadElementGenerator.roadList:
                if len(i.getAllvehiclesPaths()) > 0:
                    for j in i.getAllvehiclesPaths():
                        if len(j) > 0:
                            #print("Path: ", i.id, j)
                            paths.append(j)

            #print(paths)
            self.dependencyG.createVariableDAG(self.osmGraph.nxGraph, paths)

        if stepNumber%self.depFreq == 0 and stepNumber > 32:
            self.dependencyG.assignLoadToNodes(self.roadElementGenerator.roadList)

            laneChangeRidList = []
            laneChangeactionList = []
            laneChangeConvertedactionList = []
            requestList = []
            for j in self.roadElementGenerator.intersectionList:
                if j.num_roads >= 3:
                    while (j.request_queue):
                        req = j.request_queue.pop(0)
                        print("Request", req)
                        if not (req[1]) in laneChangeRidList:
                            if self.dependencyG.diG.has_node(req[1]):
                                requestList.append(req[0])
                                laneChangeRidList.append(req[1])
                                laneChangeactionList.append(req[2])
                                if req[2] == 'IN':
                                    laneChangeConvertedactionList.append(1)
                                elif req[2] == 'OUT':
                                    laneChangeConvertedactionList.append(-1)
                        else:
                            print("found illegal road: ",req[0], req[1])

            conflicts, road_changes = self.dependencyG.find_dependency(laneChangeRidList, laneChangeConvertedactionList,
                                                                       self.depth, self.threshold, self.load_th)

            print("Start Evaluate: ")
            for j, k, l in zip(conflicts, requestList, laneChangeactionList):
                if conflicts[j] < self.threshold:  # number of conflicts
                    print("Allow Change: intersection ", k, " Rid: ", j)
                    #self.roadElementGenerator.intersectionList[k - 1].enableChange(j, l)

                    self.roadElementGenerator.roadList[j-1].change_direction(l,
                        self.roadElementGenerator.roadList[j - 1].upstream_id)
                else:
                    print("Disable Change: intersection ", k, " Rid: ", j)
                    self.roadElementGenerator.intersectionList[k - 1].holdChange(j, l)

            for j in range(len(road_changes)):
                if road_changes[j][1] == 1:
                    action = 'IN'
                else:
                    action = 'OUT'
                print("Road changed: ", road_changes[j][0], " Action: ", action)
                out = self.roadElementGenerator.roadList[road_changes[j][0]-1]\
                    .change_direction(action, self.roadElementGenerator.roadList[road_changes[j][0]-1].upstream_id)


        '''if stepNumber%20 == 0:
            #self.osmGraph.SDpaths = self.osmGraph.SDpaths[list(self.recentPaths)]
            paths = list(self.osmGraph.recentPaths)
            self.dependencyG.createVariableDAG(self.osmGraph.nxGraph,paths)'''

    def dependencyCheckRoad(self, stepNumber):
        if stepNumber % 2 == 0:
            #self.osmGraph.SDpaths = self.osmGraph.SDpaths[-20:len(self.osmGraph.SDpaths)]
            #self.dependencyG.createVariableDAG(self.osmGraph.nxGraph, self.osmGraph.SDpaths)

            #paths = list(self.osmGraph.recentPaths)
            paths  = []
            for i in self.roadElementGenerator.roadList:
                if len(i.getAllvehiclesPaths()) > 0:
                    for j in i.getAllvehiclesPaths():
                        if len(j) > 0:
                            #print("Path: ", i.id, j)
                            paths.append(j)

            #print(paths)
            self.dependencyG.createVariableDAG(self.osmGraph.nxGraph, paths)


        if stepNumber % 2 == 0 and stepNumber > 10:
            self.dependencyG.assignLoadToNodes(self.roadElementGenerator.roadList)

            laneChangeRidList = []
            laneChangeactionList = []
            laneChangeConvertedactionList = []
            requestList = []
            for j in self.roadElementGenerator.roadList:
                if j.laneChangeRequest:
                    req = j.laneChangeRequest.pop()
                    laneChangeRidList.append(req[0])
                    laneChangeactionList.append(req[1])
                    if req[1] == 'IN':
                        laneChangeConvertedactionList.append(1)
                    elif req[1] == 'OUT':
                        laneChangeConvertedactionList.append(-1)

            conflicts, road_changes = self.dependencyG.find_dependency(laneChangeRidList,
                                                                       laneChangeConvertedactionList,
                                                                       self.depth, self.threshold, self.load_th)

            print("Dependency output: ", conflicts, road_changes)

            print("Start Evaluate: ")
            for j, k, l in zip(conflicts, laneChangeRidList, laneChangeactionList):
                if conflicts[j] < self.threshold:  # number of conflicts
                    print("Allow Change: Road ", k)
                    self.roadElementGenerator.roadList[k-1].change_direction(l,
                                                                             self.roadElementGenerator.roadList[k-1].upstream_id,
                                                                             )
                    self.roadElementGenerator.roadList[k-1].laneChangeRequest.clear()
                else:
                    print("Disable Change: intersection ", k)
                    self.roadElementGenerator.roadList[k-1].laneChangeRequest.clear()
                    #self.roadElementGenerator.intersectionList[k - 1].holdChange(j, l)

            for j in range(len(road_changes)):
                if road_changes[j][1] == 1:
                    action = 'IN'
                else:
                    action = 'OUT'
                print("Road changed: ", road_changes[j][0], " Action: ", action)
                out = self.roadElementGenerator.roadList[road_changes[j][0] - 1] \
                    .change_direction(action,
                                      self.roadElementGenerator.roadList[road_changes[j][0] - 1].upstream_id)



#rn.generateVehicles(3,2,3)
#rn.generateVehicles(12,3,3)
#rn.generateVehicles(18,17,3)
