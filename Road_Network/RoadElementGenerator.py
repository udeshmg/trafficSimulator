from OSM.OSMgraph import OsmGraph
from Road_Elements.Road import Road
from Road_Elements.Vehicles import VehicleBlock
from Road_Elements.Intersection3d import Intersection
from Agent.ddqn_parallel import DQNAgentNoBound
from Agent.ddqn_lane import DQNAgentLane
from Agent.ddqn import DQNAgent
from Agent.dummyAgent import dummyAgent
from Road_Elements.RoadConnector import RoadConnector
from pathlib import Path
import warnings
import networkx as nx

class RoadElementGenerator:

    def __init__(self, laneDirChange=True, isGuided=True):
        self.intersectionList = []
        self.agentList = []
        self.roadList = []
        self.osmGraph = None
        self.laneDirChange = laneDirChange
        if self.laneDirChange:
            self.isGuided = isGuided
        else:
            self.isGuided = False
        self.preLearned = True
        self.noAgent = False
        self.isNoBound = False
        self.noLaneChange = False
        self.timeToLaneShift = 12
        self.imbalance = 30
        self.limitVehicles = False

        self.selfLaneChange = False
        self.enableDependencyCheck = False

    def getGraph(self,osmG):
        self.osmGraph = osmG

    def getAgentStateActionSpace(self,numRoads):
        if self.laneDirChange:
            if self.isNoBound:
                action = 3 ** numRoads  # traffic phase x (3 actions for lane change for each road)
                state = 3 * numRoads
            else:
                action = numRoads*(3**numRoads) # traffic phase x (3 actions for lane change for each road)
                state = 5*numRoads
        else:
            action = numRoads
            state = 1+2*numRoads

        return state,action

    def configureAgents(self,string):
        if self.preLearned and not self.noAgent:
            if self.laneDirChange:
                for i in range(len(self.agentList)):
                    if self.agentList[i].num_roads == 3:
                        if self.isNoBound:
                            self.agentList[i].load(Path(string+"/DDQN_lane_3test"))
                        else:
                            self.agentList[i].load(Path(string+"/DDQN_lane_3_1"))
                    if self.agentList[i].num_roads == 4:
                        if self.isNoBound:
                            self.agentList[i].load(Path(string + "/DDQN_lane_4test"))
                        else:
                            self.agentList[i].load(Path(string+"/DDQN_lane_4_1"))
            else:
                for i in range(len(self.agentList)):
                    if self.agentList[i].num_roads == 3:
                        self.agentList[i].load(Path(string+"/DDQN_sig_3ver2"))
                    if self.agentList[i].num_roads == 4:
                        self.agentList[i].load(Path(string + "/DDQN_sig_4ver2"))




    def connectElements(self):

        for e in range(len(self.osmGraph.edgeIdMap)):
            numLanes = 6
            #print("ID", self.osmGraph.edgeIdMap[e])
            time = self.osmGraph.nxGraph[self.osmGraph.edgeIdMap[e][0]][self.osmGraph.edgeIdMap[e][1]]['time to travel']
            self.roadList.append(Road(e+1, numLanes, time))


            if self.osmGraph.edgeIdMap[e][0] < self.osmGraph.edgeIdMap[e][1]:
                self.roadList[-1].upstream_id = self.osmGraph.edgeIdMap[e][0]
                self.roadList[-1].downstream_id = self.osmGraph.edgeIdMap[e][1]

            elif self.osmGraph.edgeIdMap[e][1] < self.osmGraph.edgeIdMap[e][0]:
                self.roadList[-1].upstream_id = self.osmGraph.edgeIdMap[e][1]
                self.roadList[-1].downstream_id = self.osmGraph.edgeIdMap[e][0]

            self.roadList[-1].imbalanceCounter = self.imbalance
            self.roadList[-1].checkConsistent = self.timeToLaneShift
            self.roadList[-1].selfLaneChange = self.selfLaneChange
            self.roadList[-1].enableDependencyCheck = self.enableDependencyCheck
            self.roadList[-1].limitVehicles = self.limitVehicles

            #print("Road: UP: ",self.roadList[-1].id, self.roadList[-1].upstream_id, self.roadList[-1].downstream_id)


        if self.osmGraph == None:
            warnings.warn("graph is not build. Call 'buildGraph' to create")
            return -1

        for n in self.osmGraph.nxGraph.nodes():

            edgeList, numRoads = self.osmGraph.getRoadsForIntersection(n)
            if edgeList != 0 and numRoads != 0:
                if numRoads > 2:
                    self.intersectionList.append(Intersection(self.osmGraph.nxGraph.nodes[n]['id'], numRoads))
                    self.intersectionList[-1].local_view = not self.isGuided


                    stateSize, actionSize = self.getAgentStateActionSpace(numRoads)
                    if not self.noAgent:
                        if self.isNoBound:
                            self.agentList.append(DQNAgentLane(stateSize, actionSize, self.intersectionList[-1],
                                                   numRoads, self.osmGraph.nxGraph.nodes[n]['id'],
                                                   self.laneDirChange, self.isGuided))
                            self.agentList[-1].restrictLaneChange = self.noLaneChange
                        else:
                            self.agentList.append(DQNAgent(stateSize, actionSize, self.intersectionList[-1],
                                                   numRoads, self.osmGraph.nxGraph.nodes[n]['id'],
                                                   self.laneDirChange, self.isGuided))
                    else:
                        self.agentList.append(dummyAgent(self.osmGraph.nxGraph.nodes[n]['id'], self.intersectionList[-1]))
                else:
                    self.intersectionList.append(RoadConnector(self.osmGraph.nxGraph.nodes[n]['id'], numRoads))
                    self.agentList.append(dummyAgent(self.osmGraph.nxGraph.nodes[n]['id'], self.intersectionList[-1]))
                
                for i in range(len(edgeList)):
                    a = self.osmGraph.nxGraph[edgeList[i][0]][edgeList[i][1]]['edgeId']
                    #print("Road Map: ", self.intersectionList[-1].intersectionID, edgeList[i][0], edgeList[i][1], a)
                    self.intersectionList[-1].addRoad(self.roadList[a - 1])


        '''for n in self.osmGraph.nxGraph.nodes():
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
            print(self.osmGraph.edgeIdMap[rd.id - 1])'''



'''gen = RoadElementGenerator()

osmGraph = OsmGraph(-73.92, 40.75, 600)
osmGraph.createAdjacenceEdgeAttribute()

gen.getGraph(osmGraph)
gen.connectElements()'''