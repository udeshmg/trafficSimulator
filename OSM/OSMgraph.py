import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

class OsmGraph():

    def __init__(self, long, lat, diameter):
        self.long = long
        self.lat = lat
        self.diameter = diameter
        self.edgeIdMap = []
        self.nxGraph = self.buildGraph()


    def buildGraph(self):
        weightedG = nx.Graph()
        spatialG = ox.graph_from_point((self.lat, self.long), distance=self.diameter, network_type='drive')

        nodeIdMap = {}
        nodeCounter = 1
        for u in spatialG.nodes():
            nodeIdMap[u] = nodeCounter
            weightedG.add_node(nodeCounter)
            weightedG.nodes[nodeCounter]['id'] = nodeCounter
            weightedG.nodes[nodeCounter]['x'] = spatialG.nodes[u]['x']
            weightedG.nodes[nodeCounter]['y'] = spatialG.nodes[u]['y']
            weightedG.nodes[nodeCounter]['osmid'] = spatialG.nodes[u]['osmid']
            nodeCounter += 1

        edgeCounter = 1
        for x, y in spatialG.edges():
            u = nodeIdMap[x]
            v = nodeIdMap[y]
            self.edgeIdMap.append((u,v))

            weightedG.add_edge(u,v)
            weightedG[u][v]['length'] = spatialG[x][y][0]['length']
            weightedG[u][v]['angle map'] = []
            weightedG[u][v]['osmid'] = spatialG[x][y][0]['osmid']
            weightedG[u][v]['lanes'] = 6
            weightedG[u][v]['edgeId'] = edgeCounter
            edgeCounter += 1

        #print(self.edgeIdMap)

        return weightedG

    #def getAngle(self, edge1, edge2):

        # get cordinates

    def drawGraph(self):
        pos = dict((u, (self.nxGraph.nodes[u]['x'], self.nxGraph.nodes[u]['y'])) for u in self.nxGraph.nodes())
        value = [self.nxGraph[u][v]['length']/100 for u, v in self.nxGraph.edges()]
        nx.draw(self.nxGraph, pos, with_labels=True, width=value, node_size=10)
        nx.draw_networkx_nodes(self.nxGraph, pos)
        nx.draw_networkx_edge_labels(self.nxGraph,pos,
                                     edge_labels=dict([((u, v,), d['edgeId']) for u, v, d in self.nxGraph.edges(data=True)]))
        plt.show()

    def getEdgeCoordinates(self, edge):
        x1 = self.nxGraph.nodes[edge[0]]['x']
        y1 = self.nxGraph.nodes[edge[0]]['y']
        x2 = self.nxGraph.nodes[edge[1]]['x']
        y2 = self.nxGraph.nodes[edge[1]]['y']

        return x1,y1,x2,y2

    def getEdgeVector(self, edge):
        x1 = self.nxGraph.nodes[edge[0]]['x']
        y1 = self.nxGraph.nodes[edge[0]]['y']
        x2 = self.nxGraph.nodes[edge[1]]['x']
        y2 = self.nxGraph.nodes[edge[1]]['y']

        return x2-x1, y2-y1

    def getAngleBetweenEdges(self,edgeVector1, edgeVector2):
        #cosine_angle = np.dot(edgeVector1, edgeVector2) / (np.linalg.norm(edgeVector1) * np.linalg.norm(edgeVector2))
        angle = math.atan2(edgeVector2[0],edgeVector2[1]) - math.atan2(edgeVector1[0],edgeVector1[1])
        if angle < 0:
            angle = 2*math.pi - abs(angle)
        return angle

    def createAdjacenceEdgeAttribute(self):
        for u,v in self.nxGraph.edges():

            if self.nxGraph.degree[u] == 4:
                edgeList = list(self.nxGraph.edges(u))
                angleEdgeMap = []
                for i in range(len(edgeList)):
                    if edgeList[i] != (u,v):
                        angle = self.getAngleBetweenEdges(self.getEdgeVector((v, u)) ,
                                                          self.getEdgeVector((edgeList[i][1], u)))
                        angleEdgeMap.append([angle,edgeList[i]])

                angleEdgeMap.sort()
                print(u,v, angleEdgeMap)
                self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'L'])
                self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'S'])
                self.nxGraph[u][v]['angle map'].append([angleEdgeMap[2][1], 'R'])


            if self.nxGraph.degree[u] == 3:
                edgeList = list(self.nxGraph.edges(u))
                angleEdgeMap = []
                for i in range(len(edgeList)):
                    if edgeList[i] != (u,v):
                        angle = self.getAngleBetweenEdges(self.getEdgeVector((v, u)) ,
                                                          self.getEdgeVector((edgeList[i][1], u)))
                        angleEdgeMap.append([angle, edgeList[i]])

                angleEdgeMap.sort()
                print(u,v, angleEdgeMap)
                remainAngle = angleEdgeMap[1][0] - angleEdgeMap[0][0]

                if  abs(math.pi - remainAngle) < abs(math.pi - angleEdgeMap[0][0]) and \
                    abs(math.pi - remainAngle) < abs(math.pi - angleEdgeMap[1][0]):
                    self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'L'])
                    self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'R'])
                else:
                    if abs(math.pi - angleEdgeMap[0][0]) <= abs(math.pi - angleEdgeMap[1][0]):
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'S'])
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'R'])

                    if abs(math.pi - angleEdgeMap[0][0]) > abs(math.pi - angleEdgeMap[1][0]):
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'L'])
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'S'])

            if self.nxGraph.degree[u] == 2:
                edgeList = list(self.nxGraph.edges(u))
                for i in range(len(edgeList)):
                    if edgeList[i] != (u, v):
                        self.nxGraph[u][v]['angle map'].append([edgeList[i], 'S'])

            temp = v
            v = u
            u = temp
            if self.nxGraph.degree[u] == 4:
                edgeList = list(self.nxGraph.edges(u))
                angleEdgeMap = []
                for i in range(len(edgeList)):
                    if edgeList[i] != (u,v):
                        angle = self.getAngleBetweenEdges(self.getEdgeVector((v, u)) ,
                                                          self.getEdgeVector((edgeList[i][1], u)))
                        angleEdgeMap.append([angle,edgeList[i]])

                angleEdgeMap.sort()
                print(u,v, angleEdgeMap)
                self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'L'])
                self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'S'])
                self.nxGraph[u][v]['angle map'].append([angleEdgeMap[2][1], 'R'])


            if self.nxGraph.degree[u] == 3:
                edgeList = list(self.nxGraph.edges(u))
                angleEdgeMap = []
                for i in range(len(edgeList)):
                    if edgeList[i] != (u,v):
                        angle = self.getAngleBetweenEdges(self.getEdgeVector((v, u)) ,
                                                          self.getEdgeVector((edgeList[i][1], u)))
                        angleEdgeMap.append([angle, edgeList[i]])

                angleEdgeMap.sort()
                print(u,v, angleEdgeMap)
                remainAngle = angleEdgeMap[1][0] - angleEdgeMap[0][0]

                if  abs(math.pi - remainAngle) < abs(math.pi - angleEdgeMap[0][0]) and \
                    abs(math.pi - remainAngle) < abs(math.pi - angleEdgeMap[1][0]):
                    self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'L'])
                    self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'R'])
                else:
                    if abs(math.pi - angleEdgeMap[0][0]) <= abs(math.pi - angleEdgeMap[1][0]):
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'S'])
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'R'])

                    if abs(math.pi - angleEdgeMap[0][0]) > abs(math.pi - angleEdgeMap[1][0]):
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[0][1], 'L'])
                        self.nxGraph[u][v]['angle map'].append([angleEdgeMap[1][1], 'S'])

            if self.nxGraph.degree[u] == 2:
                edgeList = list(self.nxGraph.edges(u))
                for i in range(len(edgeList)):
                    if edgeList[i] != (u, v):
                        self.nxGraph[u][v]['angle map'].append([edgeList[i], 'S'])

    def getRoadsForIntersection(self,n):
        angleEdgeMap = []
        if self.nxGraph.degree[n] == 4:
            edgeList = list(self.nxGraph.edges(n))
            firstEdge = edgeList[0]
            for i in range(len(edgeList)):
                if edgeList[i] != firstEdge:
                    angle = self.getAngleBetweenEdges(self.getEdgeVector((firstEdge[1], n)) ,
                                                      self.getEdgeVector((edgeList[i][1], n)))
                    angleEdgeMap.append([angle,edgeList[i]])

            angleEdgeMap.sort(reverse=True)
            angleEdgeMap.insert(0, [0, firstEdge])
            print([row[1] for row in angleEdgeMap] )
            return [row[1] for row in angleEdgeMap], self.nxGraph.degree[n]

        elif self.nxGraph.degree[n] == 3:
            output = [None]*3
            edgeList = list(self.nxGraph.edges(n))
            firstEdge = edgeList[0]

            newList = [u for u in self.nxGraph[firstEdge[0]][firstEdge[1]]['angle map']
                       if u[0] == edgeList[1] or u[0] == edgeList[2]]
            print("3: list", newList)

            if 'R' in [row[1] for row in newList] and 'L' in [row[1] for row in newList]:
                output[0] = firstEdge
                output[1] = [row[0] for row in newList if row[1] == 'R'][0]
                output[2] = [row[0] for row in newList if row[1] == 'L'][0]

            elif 'R' in [row[1] for row in newList] and 'S' in [row[1] for row in newList]:
                output[0] = [row[0] for row in newList if row[1] == 'R'][0]
                output[1] = [row[0] for row in newList if row[1] == 'S'][0]
                output[2] = firstEdge

            elif 'L' in [row[1] for row in newList] and 'S' in [row[1] for row in newList]:
                output[0] = [row[0] for row in newList if row[1] == 'L'][0]
                output[1] = firstEdge
                output[2] = [row[0] for row in newList if row[1] == 'S'][0]

            print(output)
            return output, self.nxGraph.degree[n]

        elif self.nxGraph.degree[n] == 2 or self.nxGraph.degree[n] == 1 :
            return list(self.nxGraph.edges(n)), self.nxGraph.degree[n]













####### Testing
#nxGraph = OsmGraph(-73.92, 40.75, 750)
'''
self.nxGraph.createAdjacenceEdgeAttribute()
for u,v in self.nxGraph.nxGraph.edges():
    print(u,v)
    print(self.nxGraph.nxGraph[u][v]['angle map'])

                        '''
#nxGraph.drawGraph()
