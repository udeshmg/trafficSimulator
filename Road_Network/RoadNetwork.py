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
        self.recentPaths = 50
        self.numOfVehiclesPerBlock = 1
        self.autoGenTrafficEnabled = False

    def buildGraph(self, long, lat, diameter):
        print("Loading data from Open street maps...")
        self.osmGraph = OsmGraph(long=long, lat=lat, diameter=diameter)
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

    def generateVehicles(self, source, path, numOfVehicles=1, debugLvl=3):
        for i in range(numOfVehicles):
            vb, startEdge = self.vehicleGenerator.generateVehicleWithId(path, debugLvl)
            if vb != None or startEdge != None:
                rid = self.osmGraph.nxGraph[startEdge[0]][startEdge[1]]['edgeId']

                if self.roadElementGenerator.roadList[rid - 1].upstream_id == source:
                    self.roadElementGenerator.roadList[rid - 1].add_block(vb, 'DOWN')
                else:
                    self.roadElementGenerator.roadList[rid - 1].add_block(vb, 'UP')


        #print(self.roadElementGenerator.roadList[rid - 1].get_num_vehicles(self.roadElementGenerator.roadList[rid - 1].downstream_id,'T'))

    def addTrafficFromData(self,step):

        if not self.autoGenTrafficEnabled:
            vehicleList = self.trafficLoader.getData(step)
            nodeList = self.osmGraph.getNearestNode(vehicleList)
            for n in nodeList:
                self.generateVehicles(n[0],n[2],self.numOfVehiclesPerBlock, 2)
        else:
            for i in self.trafficGenerator.getTrafficData():
                n = self.osmGraph.getPathFromNodes(i[0],i[1],i[2])
                self.generateVehicles(n[0],n[2],i[2], 2)      


 # source and destination

    def simulateOneStep(self, stepNumber):

        for i in range(len(self.roadElementGenerator.agentList)):
            self.roadElementGenerator.agentList[i].actOnIntersection()

        for i in range(len(self.roadElementGenerator.roadList)):
            self.roadElementGenerator.roadList[i].step(10)

        if self.roadElementGenerator.isGuided:
            self.dependencyCheck(stepNumber)

        for i in range(len(self.roadElementGenerator.agentList)):
            self.roadElementGenerator.agentList[i].getOutputData()

    def dependencyCheck(self,stepNumber):

        if stepNumber%6 == 0 and stepNumber > 50:
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
                    self.roadElementGenerator.intersectionList[k - 1].enableChange(j, l)
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


        if stepNumber%30 == 0:
            self.osmGraph.SDpaths = self.osmGraph.SDpaths[-self.recentPaths:len(self.osmGraph.SDpaths)]
            self.dependencyG.createVariableDAG(self.osmGraph.nxGraph,self.osmGraph.SDpaths)

#rn.generateVehicles(3,2,3)
#rn.generateVehicles(12,3,3)
#rn.generateVehicles(18,17,3)
