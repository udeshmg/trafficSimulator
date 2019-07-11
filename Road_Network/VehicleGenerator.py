from OSM.OSMgraph import OsmGraph
from Road_Elements.Lane import Lane
from Road_Elements.Road import Road
from Road_Elements.Vehicles import VehicleBlock
from Road_Elements.Intersection import Intersection

import networkx as nx

class VehicleGenerator():

    def __init__(self):
        self.vehicleData = None
        self.osmG = None # OsmGraph(0,0,0)
        self.vehicleId = 0

    def getGraph(self, osmG):
        self.osmG = osmG

    def generateVehicleWithId(self, sourceId, destinationId):
        path = nx.shortest_path(self.osmG.nxGraph,sourceId,destinationId,weight='length')
        decodedPath = self.decodePath(path)
        vb = VehicleBlock(1, decodedPath, self.vehicleId, 3)
        self.vehicleId += 1
        return vb, (path[0], path[1])

    def decodePath(self, path):
        turnDirection = []
        for i in range(len(path)-2):
            edge1 = self.osmG.nxGraph[path[i]][path[i+1]]

            for j in edge1['angle map']:
                if j[0] == (path[i+1],path[i+2]): # next Edge
                    turnDirection.append(j[1])
                    break

        return turnDirection

'''osmGraph = OsmGraph(-73.92, 40.75, 600)
osmGraph.createAdjacenceEdgeAttribute()
vg = VehicleGenerator()
vg.getGraph(osmGraph)
vg.generatorVehicleWithId(10,73)'''
