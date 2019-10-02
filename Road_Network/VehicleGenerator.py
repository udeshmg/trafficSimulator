from OSM.OSMgraph import OsmGraph
from Road_Elements.Lane import Lane
from Road_Elements.Road import Road
from Road_Elements.Vehicles import VehicleBlock
from Road_Elements.Intersection import Intersection
import math

import networkx as nx

class VehicleGenerator():

    def __init__(self):
        self.vehicleData = None
        self.osmG = None # OsmGraph(0,0,0)
        self.vehicleId = 0
        self.numVehiclePerBlock = 1
        self.vehicleList = []

    def clear(self):
        self.vehicleId = 0
        self.vehicleList = []

    def getGraph(self, osmG):
        self.osmG = osmG

    def generateVehicleWithId(self, path, debugLvl=3, indexAtFile=0):
        #path = nx.shortest_path(self.osmG.nxGraph,sourceId,destinationId,weight='length')
        if len(path) > 1:
            if self.vehicleId == 91 or self.vehicleId == 92:
                print("### ", path, self.vehicleId)
                debugLvl = 2
            else:
                debugLvl = 2
            decodedPath, freeFlowTime = self.decodePath(path)
            vb = VehicleBlock(1, decodedPath, self.vehicleId, debugLvl, path=path)
            vb.freeFlowTime = freeFlowTime
            self.vehicleId += 1
            self.vehicleList.append(vb)
            return vb, (path[0], path[1])
        return None, None

    def decodePath(self, path):
        turnDirection = []
        freeFlowTime = 0
        for i in range(len(path)-2):
            edge1 = self.osmG.nxGraph[path[i]][path[i+1]]
            freeFlowTime += math.ceil(edge1['time to travel']/10)*10
            for j in edge1['angle map']:
                if j[0] == (path[i+1],path[i+2]): # next Edge
                    turnDirection.append(j[1])
                    break

        return turnDirection, freeFlowTime

'''osmGraph = OsmGraph(-73.92, 40.75, 600)
osmGraph.createAdjacenceEdgeAttribute()
vg = VehicleGenerator()
vg.getGraph(osmGraph)
vg.generatorVehicleWithId(10,73)'''
