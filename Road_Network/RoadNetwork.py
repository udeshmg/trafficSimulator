from OSM.OSMgraph import OsmGraph
from Road_Elements.Lane import Lane
from Road_Elements.Road import Road
from Road_Elements.Vehicles import VehicleBlock
from Road_Elements.Intersection import Intersection
from Road_Network.RoadElementGenerator import RoadElementGenerator
from Road_Network.VehicleGenerator import VehicleGenerator

class RoadNetwork:

    def __init__(self):
        self.osmGraph = None
        self.roadElementGenerator = RoadElementGenerator()
        self.vehicleGenerator = VehicleGenerator()
        self.timeStep = 10

    def buildGraph(self, long, lat, diameter):
        self.osmGraph = OsmGraph(long=long, lat=lat, diameter=diameter)
        self.osmGraph.createAdjacenceEdgeAttribute()

    def init(self):
        self.roadElementGenerator.getGraph(self.osmGraph)
        self.vehicleGenerator.getGraph(self.osmGraph)

    def createNetwork(self):
        self.roadElementGenerator.connectElements()

    def generateVehicles(self, source, destination):
        vb, startEdge = self.vehicleGenerator.generateVehicleWithId(source, destination)
        rid = self.osmGraph.nxGraph[startEdge[0]][startEdge[1]]['edgeId']

        if self.roadElementGenerator.roadList[rid - 1].upstream_id == source:
            self.roadElementGenerator.roadList[rid - 1].add_block(vb, 'DOWN')
        else:
            self.roadElementGenerator.roadList[rid - 1].add_block(vb, 'UP')

        print(self.roadElementGenerator.roadList[rid - 1].get_num_vehicles(
            self.roadElementGenerator.roadList[rid - 1].downstream_id,'T'))

    def addTrafficFromData(self):
        print("access database")

    def simulateOneStep(self, stepNumber):

        for i in range(len(self.roadElementGenerator.agentList)):
            self.roadElementGenerator.agentList[i].actOnIntersection()

        for i in range(len(self.roadElementGenerator.roadList)):
            self.roadElementGenerator.roadList[i].step(10)

        self.generateVehicles(10,5)

        for i in range(len(self.roadElementGenerator.agentList)):
            self.roadElementGenerator.agentList[i].getOutputData()





rn = RoadNetwork()
rn.buildGraph(-73.92, 40.75, 600)
rn.init()
rn.createNetwork()

rn.generateVehicles(10,2)
rn.generateVehicles(10,40)
rn.generateVehicles(10,61)

for i in range(1000):
    print("Step: ",i)
    rn.simulateOneStep(i)










