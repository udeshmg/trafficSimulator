import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import deque

class OsmGraph():

    def __init__(self, long, lat, diameter, osm=True):
        self.long = long
        self.lat = lat
        self.diameter = diameter
        self.edgeIdMap = []
        if osm:
            self.nxGraph = self.buildGraph()
        else:
            self.nxGraph = self.manualGraph()
        self.SDpairs = []
        self.SDpaths = []
        self.filteredSDpairs = []
        self.recentPaths = deque(maxlen=20)

    def clear(self):
        self.SDpairs = []
        self.SDpaths = []
        self.filteredSDpairs = []


    def manualGraph(self):
        weightedG = nx.Graph()
        nodeIdMap = {}
        nodeCounter =  1

        '''coordMap = [[1,0],
                    [2,0],
                    [0,1],
                    [1,1],
                    [2,1],
                    [3,1],
                    [0,2],
                    [1,2],
                    [2,2],
                    [3,2],
                    [1,3],
                    [2,3]]'''

        coordMap = [[1,0],[2,0],[3,0],[4,0],[5,0],
                    [0,1], [1,1], [2,1], [3,1], [4,1], [5,1], [6,1],
                    [0,2], [1,2], [2,2], [3,2], [4,2], [5,2], [6,2],
                    [0,3], [1,3], [2,3], [3,3], [4,3], [5,3], [6,3],
                    [0,4], [1,4], [2,4], [3,4], [4,4], [5,4], [6,4],
                    [0,5], [1,5], [2,5], [3,5], [4,5], [5,5], [6,5],
                    [1,6],[2,6],[3,6],[4,6],[5,6]]

        for i in range(len(coordMap)):
            weightedG.add_node(i+1)
            nodeIdMap[i] = nodeCounter
            weightedG.add_node(nodeCounter)
            weightedG.nodes[nodeCounter]['id'] = nodeCounter
            weightedG.nodes[nodeCounter]['x'] = coordMap[i][0]
            weightedG.nodes[nodeCounter]['y'] = coordMap[i][1]
            weightedG.nodes[nodeCounter]['osmid'] = 0
            weightedG.nodes[nodeCounter]['source'] = 0
            weightedG.nodes[nodeCounter]['destination'] = 0
            nodeCounter += 1

        edgeCounter = 1

        '''roadMap = [[1,4],
                   [2,5],
                   [3,4],
                   [4,5],
                   [5,6],
                   [4,8],
                   [5,9],
                   [7,8],
                   [8,9],
                   [9,10],
                   [8,11],
                   [9,12]
                  ]'''

        roadMap = [[1,7],[2,8],[3,9],[4,10],[5,11],
                   [6,7],[7,8],[8,9],[9,10],[10,11],[11,12],
                   [7,14],[8,15],[9,16],[10,17],[11,18],
                   [13,14],[14,15],[15,16],[16,17],[17,18],[18,19],
                   [14,21],[15,22],[16,23],[17,24],[18,25],
                   [20,21],[21,22],[22,23],[23,24],[24,25],[25,26],
                   [21,28],[22,29],[23,30],[24,31],[25,32],
                   [27,28],[28,29],[29,30],[30,31],[31,32],[32,33],
                   [28,35],[29,36],[30,37],[31,38],[32,39],
                   [34,35],[35,36],[36,37],[37,38],[38,39],[39,40],
                   [35,41],[36,42],[37,43],[38,44],[39,45]]

        for u,v in roadMap:
            self.edgeIdMap.append((u, v))
            weightedG.add_edge(u, v)
            weightedG[u][v]['length'] = 180
            weightedG[u][v]['angle map'] = []
            weightedG[u][v]['osmid'] = 0 #spatialG[x][y][0]['osmid']
            weightedG[u][v]['lanes'] = 6
            weightedG[u][v]['edgeId'] = edgeCounter
            weightedG[u][v]['path'] = 0
            weightedG[u][v]['time to travel'] = 21

            edgeCounter += 1

        '''weightedG[37][38]['time to travel'] = 30
        weightedG[37][36]['time to travel'] = 30
        weightedG[36][35]['time to travel'] = 30
        weightedG[7][8]['time to travel'] = 30
        weightedG[8][9]['time to travel'] = 30
        weightedG[9][10]['time to travel'] = 30

        weightedG[35][28]['time to travel'] = 30
        weightedG[28][21]['time to travel'] = 30
        weightedG[21][14]['time to travel'] = 30
        weightedG[25][32]['time to travel'] = 30
        weightedG[32][39]['time to travel'] = 30
        weightedG[18][25]['time to travel'] = 30

        weightedG[30][31]['time to travel'] = 10
        weightedG[30][29]['time to travel'] = 10'''

        return weightedG


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
            weightedG.nodes[nodeCounter]['source'] = 0
            weightedG.nodes[nodeCounter]['destination'] = 0
            nodeCounter += 1

        edgeCounter = 1
        for x, y in spatialG.edges():
            u = nodeIdMap[x]
            v = nodeIdMap[y]
            if weightedG.degree[u] < 4 and weightedG.degree[v] < 4:
                self.edgeIdMap.append((u,v))
                weightedG.add_edge(u,v)
                weightedG[u][v]['length'] = spatialG[x][y][0]['length']
                weightedG[u][v]['angle map'] = []
                weightedG[u][v]['osmid'] = spatialG[x][y][0]['osmid']
                weightedG[u][v]['lanes'] = 6
                weightedG[u][v]['edgeId'] = edgeCounter
                weightedG[u][v]['path'] = 0

                if 'maxspeed' in spatialG[x][y]:
                    weightedG[u][v]['time to travel'] = weightedG[u][v]['length'] / (spatialG[x][y][0]['maxpeed']*0.447)
                else:
                    weightedG[u][v]['time to travel'] = weightedG[u][v]['length'] / (20*0.447)

                edgeCounter += 1

        #print(self.edgeIdMap)

        for u,v in weightedG.edges:
            print(weightedG[u][v]['edgeId'], weightedG[u][v]['length'], weightedG[u][v]['time to travel'])

        return weightedG

    #def getAngle(self, edge1, edge2):

        # get cordinates

    def drawGraph(self,block=True, attr='length'):
        pos = dict((u, (self.nxGraph.nodes[u]['x'], self.nxGraph.nodes[u]['y'])) for u in self.nxGraph.nodes())
        value = [self.nxGraph[u][v][attr]/20 for u, v in self.nxGraph.edges()]
        nx.draw(self.nxGraph, pos, with_labels=True, width=value, node_size=10)
        nx.draw_networkx_nodes(self.nxGraph, pos)
        nx.draw_networkx_edge_labels(self.nxGraph,pos,
                                     edge_labels=dict([((u, v,), int(d[attr])) for u, v, d in self.nxGraph.edges(data=True)]))
        plt.show(block)

    def filterSDpairs(self):
        for i in range(len(self.SDpairs)):
            if self.SDpairs[i][0] != self.SDpairs[i][1]:
                self.filteredSDpairs.append(self.SDpairs[i])

    def drawGraphWithUserTraffic(self, block=True , figName="Time selected"):
        pos = dict((u, (self.nxGraph.nodes[u]['x'], self.nxGraph.nodes[u]['y'])) for u in self.nxGraph.nodes())
        value = [self.nxGraph[u][v]['path']/10 for u, v in self.nxGraph.edges()]
        #plt.figure(figName)
        colorMap = []
        '''for node in self.nxGraph.nodes():
            if (node in [i[0] for i in self.filteredSDpairs]) and (node in [i[1] for i in self.filteredSDpairs]):
                colorMap.append('b')
            elif (node in [i[0] for i in self.filteredSDpairs]):
                colorMap.append('r')
            elif (node in [i[1] for i in self.filteredSDpairs]):
                colorMap.append('g')
            else:
                colorMap.append('gray')'''

        for node in self.nxGraph.nodes():
            if self.nxGraph.nodes[node]['source'] >  self.nxGraph.nodes[node]['destination'] :
                colorMap.append('r')
            elif self.nxGraph.nodes[node]['source'] < self.nxGraph.nodes[node]['destination']:
                colorMap.append('g')
            elif self.nxGraph.nodes[node]['source'] == 0 and  self.nxGraph.nodes[node]['destination'] == 0:
                colorMap.append('gray')
            else:
                colorMap.append('b')



        nx.draw(self.nxGraph, pos, with_labels=True, width=value, node_size=3)
        nx.draw_networkx_nodes(self.nxGraph, pos, node_color=colorMap)
        nx.draw_networkx_edge_labels(self.nxGraph,pos,
                                     edge_labels=dict([((u, v,), d['edgeId']) for u, v, d in self.nxGraph.edges(data=True)]))
        #plt.show()
        plt.pause(1)

    def drawPathOnMap(self,imbalance=False, type=False, dir='both'):
        for u,v in self.nxGraph.edges():
            self.nxGraph[u][v]['path'] = 0

        for i in range(len(self.SDpaths)):
            for j in range(len(self.SDpaths[i])-1):
                if not imbalance:
                    if type:
                        if dir == 'UP':
                            if self.SDpaths[i][j] > self.SDpaths[i][j+1]:
                                self.nxGraph[self.SDpaths[i][j]][self.SDpaths[i][j+1]]['path'] += self.SDpairs[i][2]
                        elif dir == 'DOWN':
                            if self.SDpaths[i][j] <= self.SDpaths[i][j+1]:
                                self.nxGraph[self.SDpaths[i][j]][self.SDpaths[i][j+1]]['path'] += self.SDpairs[i][2]
                        else:
                            self.nxGraph[self.SDpaths[i][j]][self.SDpaths[i][j + 1]]['path'] += self.SDpairs[i][2]
                    else:
                        self.nxGraph[self.SDpaths[i][j]][self.SDpaths[i][j+1]]['path'] += 1
                else:
                    if self.nxGraph.nodes[self.SDpaths[i][j]]['id'] > self.nxGraph.nodes[self.SDpaths[i][j+1]]['id']:
                        self.nxGraph[self.SDpaths[i][j]][self.SDpaths[i][j + 1]]['path'] += self.SDpairs[i][2]

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
            #print("3: list", newList)

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

            #print(output)
            return output, self.nxGraph.degree[n]

        elif self.nxGraph.degree[n] == 2 or self.nxGraph.degree[n] == 1 :
            return list(self.nxGraph.edges(n)), self.nxGraph.degree[n]

        else:
            return 0, 0



    def getNearestNode(self,locationList):
        SDList =[[-1,-1,-1.0,-1.0] for i in range(len(locationList))]
        for n in self.nxGraph.nodes():
            for l in range(len(locationList)):
                nodeLoc = [self.nxGraph.nodes[n]['x'], self.nxGraph.nodes[n]['y']]
                source = [locationList[l][0],locationList[l][1]]
                destination = [locationList[l][2], locationList[l][3]]

                distance1 = self.calculateDistance2p(source, nodeLoc)
                distance2 = self.calculateDistance2p(destination, nodeLoc)

                #print("Distance: ", distance1, distance2)

                if SDList[l][2] > distance1 or SDList[l][2] == -1:
                    SDList[l][0] = self.nxGraph.nodes[n]['id']
                    SDList[l][2] = distance1

                if SDList[l][3] > distance2 or SDList[l][3] == -1:
                    SDList[l][1] = self.nxGraph.nodes[n]['id']
                    SDList[l][3] = distance2

        for l in range(len(SDList)):
            self.nxGraph.nodes[SDList[l][0]]['source'] += 1
            self.nxGraph.nodes[SDList[l][1]]['destination'] += 1

            path = nx.shortest_path(self.nxGraph, SDList[l][0], SDList[l][1])
            SDList[l].append(path)
            self.recentPaths.append(path)

            if [SDList[l][0],SDList[l][1]] not in self.SDpairs:
                self.SDpairs.append([SDList[l][0],SDList[l][1]])
                self.SDpaths.append(path)

        return [[row[0],row[1],row[4]] for row in SDList]

    def getPathFromNodes(self, source, destination, load):
        path = nx.shortest_path(self.nxGraph, source, destination, weight='time to travel')

        self.nxGraph.nodes[source]['source'] += load
        self.nxGraph.nodes[destination]['destination'] += load
        self.recentPaths.append(path)
        if [source, destination] not in [[row[0],row[1]] for row in self.SDpairs]:
            self.SDpairs.append([source, destination, load])
            self.SDpaths.append(path)

        return source, destination, path



    @staticmethod
    def calculateDistance2p(point1, point2):
        return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)


####### Testing
#nxGraph = OsmGraph(-73.9189, 40.7468, 300)
#a = [[-73.9191, 40.7479, -73.9168, 40.7455]]
#nxGraph.drawGraph()
#print(nxGraph.getNearestNode(a))


'''
self.nxGraph.createAdjacenceEdgeAttribute()
for u,v in self.nxGraph.nxGraph.edges():
    print(u,v)
    print(self.nxGraph.nxGraph[u][v]['angle map'])

                        '''
#spatialG = ox.graph_from_point(( 40.744612, -73.995830), distance=600, network_type='drive')
#ox.plot_graph(spatialG, edge_linewidth=4, edge_color='b')
