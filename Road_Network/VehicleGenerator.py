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

    def getGraph(self, osmG):
        self.osmG = osmG

    def generateVehicleWithId(self, path, debugLvl=3):
        #path = nx.shortest_path(self.osmG.nxGraph,sourceId,destinationId,weight='length')
        if len(path) > 1:
            decodedPath, freeFlowTime = self.decodePath(path)
            vb = VehicleBlock(1, decodedPath, self.vehicleId, debugLvl)
            vb.freeFlowTime = freeFlowTime
            self.vehicleId += 1
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
