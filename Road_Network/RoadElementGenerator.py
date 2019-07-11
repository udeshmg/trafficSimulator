from OSM.OSMgraph import OsmGraph
from Road_Elements.Lane import Lane
from Road_Elements.Road import Road
from Road_Elements.Vehicles import VehicleBlock
from Road_Elements.Intersection3d import Intersection
from Agent.ddqn import DQNAgent
from Agent.dummyAgent import dummyAgent
from Road_Elements.RoadConnector import RoadConnector

import warnings
import networkx as nx

class RoadElementGenerator:

    def __init__(self, laneDirChange=True, isGuided=True):
        self.intersectionList = []
        self.agentList = []
        self.roadList = []
        self.osmGraph = None
        self.laneDirChange = laneDirChange
        self.isGuided = isGuided

    def getGraph(self,osmG):
        self.osmGraph = osmG

    def getAgentStateActionSpace(self,numRoads):
        if self.laneDirChange:
            action = numRoads*(3**numRoads) # traffic phase x (3 actions for lane change for each road)
            state = 1+5*numRoads
        else:
            action = numRoads
            state = 1+2*numRoads

        return state,action

    def connectElements(self):

        for e in range(len(self.osmGraph.edgeIdMap)):
            numLanes = 6
            print("ID", self.osmGraph.edgeIdMap[e])
            self.roadList.append(Road(e+1, numLanes))

            if self.osmGraph.edgeIdMap[e][0] < self.osmGraph.edgeIdMap[e][1]:
                self.roadList[-1].upstream_id = self.osmGraph.edgeIdMap[e][0]
                self.roadList[-1].downstream_id = self.osmGraph.edgeIdMap[e][1]

            elif self.osmGraph.edgeIdMap[e][1] < self.osmGraph.edgeIdMap[e][0]:
                self.roadList[-1].upstream_id = self.osmGraph.edgeIdMap[e][1]
                self.roadList[-1].downstream_id = self.osmGraph.edgeIdMap[e][0]


        if self.osmGraph == None:
            warnings.warn("graph is not build. Call 'buildGraph' to create")
            return -1

        for n in self.osmGraph.nxGraph.nodes():
            print("Node", n)

            #if self.osmGraph.nxGraph.degree[n] == 4:
            edgeList, numRoads = self.osmGraph.getRoadsForIntersection(n)
            if numRoads > 2:
                self.intersectionList.append(Intersection(self.osmGraph.nxGraph.nodes[n]['id'], numRoads))



                stateSize, actionSize = self.getAgentStateActionSpace(numRoads)
                self.agentList.append(DQNAgent(stateSize, actionSize, self.intersectionList[-1],
                                               numRoads, self.osmGraph.nxGraph.nodes[n]['id'],
                                               self.laneDirChange, self.isGuided))
            else:
                self.intersectionList.append(RoadConnector(self.osmGraph.nxGraph.nodes[n]['id'], numRoads))
                self.agentList.append(dummyAgent(self.osmGraph.nxGraph.nodes[n]['id'], self.intersectionList[-1]))

            for i in range(len(edgeList)):
                a = self.osmGraph.nxGraph[edgeList[i][0]][edgeList[i][1]]['edgeId']
                print(edgeList[i][0], edgeList[i][1], a)
                self.intersectionList[-1].addRoad(self.roadList[a - 1])


        for n in self.osmGraph.nxGraph.nodes():
            if self.osmGraph.nxGraph.degree[n] == 4:
                print(self.intersectionList[n-1].intersectionID,n)
                print(self.intersectionList[n-1].road[0].id,self.intersectionList[n-1].road[1].id,
                  self.intersectionList[n-1].road[2].id,self.intersectionList[n-1].road[3].id)
            if self.osmGraph.nxGraph.degree[n] == 3:
                print(self.intersectionList[n-1].intersectionID,n)
                print(self.intersectionList[n-1].road[0].id,self.intersectionList[n-1].road[1].id,
                  self.intersectionList[n-1].road[2].id)

            if self.osmGraph.nxGraph.degree[n] == 2:
                print(self.intersectionList[n-1].intersectionID,n)
                print(self.intersectionList[n-1].road[0].id,self.intersectionList[n-1].road[1].id)

            if self.osmGraph.nxGraph.degree[n] == 1:
                print(self.intersectionList[n-1].intersectionID,n)
                print(self.intersectionList[n-1].road[0].id)

        for rd in self.roadList:
            print("Road", rd.id, rd.upstream_id, rd.downstream_id)
            print(self.osmGraph.edgeIdMap[rd.id - 1])



'''gen = RoadElementGenerator()

osmGraph = OsmGraph(-73.92, 40.75, 600)
osmGraph.createAdjacenceEdgeAttribute()

gen.getGraph(osmGraph)
gen.connectElements()'''