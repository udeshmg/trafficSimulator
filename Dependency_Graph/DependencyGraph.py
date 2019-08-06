import networkx as nx
from collections import OrderedDict

class DependencyGraph():

    def __init__(self):
        self.diG = nx.DiGraph()

    def createVariableDAG(self,G,OD_list):
        self.diG = nx.DiGraph()
        for i in OD_list:
            #print(i)
            for j in range(len(i) - 1):
                u = G[i[j]][i[j + 1]]['edgeId']
                u1 = (i[j], i[j + 1])
                for k in range(j + 1, min(len(i) - 1, j + 3)):
                    # v = getIndex(i[k],i[k+1], gridSize)
                    v = G[i[k]][i[k + 1]]['edgeId']
                    v1 = (i[k], i[k + 1])
                    if self.getDirectionRelationship(u1,v1):
                        if self.diG.has_edge(u, v) == True:
                            if self.diG[u][v]['direction'] == 0 or self.diG[u][v]['direction'] == -1:
                                self.diG[u][v]['direction'] = 0
                            else:
                                self.diG[u][v]['direction'] = 1
                        else:
                            self.diG.add_weighted_edges_from([(u, v, 1)], weight='direction')
                    else:
                        if self.diG.has_edge(u, v) == True:
                            if self.diG[u][v]['direction'] == 0 or self.diG[u][v]['direction'] == 1:
                                self.diG[u][v]['direction'] = 0
                            else:
                                self.diG[u][v]['direction'] = -1
                        self.diG.add_weighted_edges_from([(u, v, -1)], weight='direction')

    def getDirectionRelationship(self,edge1, edge2):
        direc1 = self.find_direction(edge1[0], edge1[1])
        direc2 = self.find_direction(edge2[0], edge2[1])

        if direc1 == direc2:
            return 1
        else:
            return -1

    def find_direction(self,u, v):
        if (u > v):
            return 'UP'
        else:
            return 'DOWN'

    def assignLoadToNodes(self, roadList):

        for n in self.diG.nodes():
            self.diG.nodes[n]['imbalance'] = roadList[n-1].get_traffic_imbalance(roadList[n-1].upstream_id)  # Current imbalance
            self.diG.nodes[n]['configuration'] = roadList[n-1].get_in_lanes_num(roadList[n-1].upstream_id)  # Current config
            self.diG.nodes[n]['load'] = roadList[n-1].get_num_vehicles(roadList[n-1].upstream_id, 'T') + roadList[n-1].get_num_vehicles(roadList[n-1].downstream_id, 'T')
        return self.diG

    def find_dependency(self, startNodelist, actionList, depth, threshold, load_th=20):

        for node in self.diG.nodes():
            self.diG.nodes[node]['list'] = []
            self.diG.nodes[node]['change'] = 0
            self.diG.nodes[node]['visited'] = False
            self.diG.nodes[node]['action'] = 0
            self.diG.nodes[node]['depth'] = 0

        additional_changes = []
        conflict_counter = OrderedDict()
        queue_lvl1 = []
        queue_lvl2 = []
        for i in range(len(startNodelist)):
            conflict = 0
            print(startNodelist)
            self.diG.nodes[startNodelist[i]]['list'] = [[startNodelist[i], depth, actionList[i], conflict]]
            self.diG.nodes[startNodelist[i]]['change'] = actionList[i]
            queue_lvl2.append([])
            queue_lvl1.append([startNodelist[i]])
            conflict_counter[startNodelist[i]] = 0

        while depth != 0:
            for k in range(len(startNodelist)):
                print("### start iteration of ", k)
                while queue_lvl1[k]:
                    print(queue_lvl1)
                    currentNode = queue_lvl1[k].pop(0)

                    if not self.diG.nodes[currentNode]['visited']:
                        print("Current node [unvisited]", currentNode)
                        self.diG.nodes[currentNode]['visited'] = True
                        self.diG.nodes[currentNode]['depth'] = depth

                        increase_action = 0
                        decrease_action = 0
                        for i in range(len(self.diG.nodes[currentNode]['list'])):
                            if self.diG.nodes[currentNode]['list'][i][2] == 1:
                                increase_action += 1
                            elif self.diG.nodes[currentNode]['list'][i][2] == -1:
                                decrease_action += 1
                        # print(" Data ", self.diG.nodes[currentNode]['imbalance'], self.diG.nodes[currentNode]['configuration'], increase_action, decrease_action)
                        inc, dec, action = self.road_decision(self.diG.nodes[currentNode]['imbalance'],
                                                         self.diG.nodes[currentNode]['configuration'],
                                                         increase_action, decrease_action,
                                                         self.diG.nodes[currentNode]['load'], load_th)

                        self.diG.nodes[currentNode]['action'] = action
                        self.diG.nodes[currentNode]['increase'] = inc
                        self.diG.nodes[currentNode]['decrease'] = dec

                        for i in range(len(self.diG.nodes[currentNode]['list'])):
                            if self.diG.nodes[currentNode]['list'][i][2] == 1:
                                if inc == False:
                                    self.diG.nodes[currentNode]['list'][i][3] += 1
                                    conflict_counter[self.diG.nodes[currentNode]['list'][i][0]] += 1
                            elif self.diG.nodes[currentNode]['list'][i][2] == -1:
                                if dec == False:
                                    self.diG.nodes[currentNode]['list'][i][3] += 1
                                    conflict_counter[self.diG.nodes[currentNode]['list'][i][0]] += 1
                            # compare

                    id = self.findStartNodeIndex(self.diG.nodes[currentNode]['list'], startNodelist[k])

                    for i in self.diG.successors(currentNode):
                        outcome = self.diG[currentNode][i]['direction'] * self.diG.nodes[currentNode]['list'][id][2]
                        # print("Child added ", i)
                        index = self.findStartNodeIndex(self.diG.nodes[i]['list'], startNodelist[k])
                        if self.diG.nodes[i]['visited']:
                            if (outcome == 1 and self.diG.nodes[i]['increase'] == False) or (
                                    outcome == -1 and self.diG.nodes[i]['decrease'] == False):
                                if index == -1:
                                    self.diG.nodes[i]['list'].append([startNodelist[k], depth - 1, outcome, 1])
                                    conflict_counter[startNodelist[k]] += 1
                                    queue_lvl2[k].append(i)
                            else:
                                if index == -1:
                                    self.diG.nodes[i]['list'].append([startNodelist[k], depth - 1, outcome, 0])
                                    queue_lvl2[k].append(i)
                        else:
                            if index == -1:
                                self.diG.nodes[i]['list'].append([startNodelist[k], depth - 1, outcome, 0])
                                queue_lvl2[k].append(i)

                queue_lvl1[k] = queue_lvl2[k].copy()
                queue_lvl2[k].clear()
                print("### end iteration of ", k)
            depth -= 1

        for node in self.diG.nodes():
            increase_action = 0
            decrease_action = 0
            for i in range(len(self.diG.nodes[node]['list'])):
                if (conflict_counter[self.diG.nodes[node]['list'][i][0]] < threshold) and self.diG.nodes[node]['list'][i][1] == \
                        self.diG.nodes[node]['depth']:
                    if self.diG.nodes[node]['list'][i][2] == 1:
                        increase_action += 1
                    elif self.diG.nodes[node]['list'][i][2] == -1:
                        decrease_action += 1

            if not (increase_action == 0 and decrease_action == 0):
                inc, dec, action = self.road_decision(self.diG.nodes[node]['imbalance'],
                                                 self.diG.nodes[node]['configuration'],
                                                 increase_action, decrease_action, self.diG.nodes[node]['load'], load_th)

                self.diG.nodes[node]['action'] = action

                if action != 0:
                    if not node in startNodelist:
                        print("Additional change", node, action)
                        additional_changes.append([node, action])
        print("Conflict Counter", conflict_counter)
        return conflict_counter, additional_changes

    @staticmethod
    def road_decision(imbalance, configuration, increase_action, decrease_action, load=0, thresh=20):
        # imbalance 1 means output high
        # conf input lanes
        increase_result = False
        decrease_result = False

        action = 0
        if (imbalance == 1 and configuration == 2):
            increase_result = True
            decrease_result = False

        elif imbalance == 1:
            if decrease_action > 0 and increase_action == 0:  # when only output increases
                decrease_result = True
                if load > thresh:
                    action = -1

            elif decrease_action >= 0 and increase_action > 0:
                decrease_result = False
                increase_result = True


        elif (imbalance == 2 and configuration == 4):
            increase_result = False
            decrease_result = True

        elif imbalance == 2:
            if increase_action > 0 and decrease_action == 0:
                increase_result = True
                if load > thresh:
                    action = 1

            elif increase_action >= 0 and decrease_action > 0:
                decrease_result = True
                increase_result = False


        elif imbalance == 0:
            if configuration == 2:
                if increase_action == 0:
                    decrease_result = True
                    increase_result = True

                else:
                    decrease_result = False
                    increase_result = True
                    if load < 360:
                        action = 1
                    else:
                        print("Almost full: ")

            elif configuration == 4:
                if decrease_action == 0:
                    decrease_result = True
                    increase_result = True

                else:
                    decrease_result = True
                    increase_result = False
                    if load < 360:
                        action = -1
                    else:
                        print("Almost full: ")

            else:  # action change incident
                '''if increase_action > decrease_action and load > thresh:
                    action = 1
                elif increase_action < decrease_action and load > thresh:
                    action = -1'''
                decrease_result = True
                increase_result = True
        return increase_result, decrease_result, action

    @staticmethod
    def findStartNodeIndex(l, startNode, column=0):
        for i in range(len(l)):
            if l[i][column] == startNode:
                return i
        return -1