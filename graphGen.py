from Road_Network.RoadNetwork import RoadNetwork
from Stat_Reporter.StatReporter import Reporter

iterations = 360
rn = RoadNetwork()
reporter = Reporter()
#rn.buildGraph(-73.9224, 40.746, 250)
rn.buildGraph(-73.92, 40.75, 600)
#rn.buildGraph(-73.996575, 40.747477, 600)
#rn.buildGraph(-73.995830, 40.744612, 1200)
#rn.buildGraph(-73.933, 40.758,600)
#rn.buildGraph(-73.9189, 40.7468, 500)
#rn.osmGraph.drawGraph(False)
rn.init()
rn.createNetwork()
print(rn.osmGraph.nxGraph.number_of_edges())


## agent data
rn.roadElementGenerator.preLearned = False
rn.roadElementGenerator.configureAgents("Tests")
rn.roadElementGenerator.isGuided = True
rn.roadElementGenerator.laneDirChange = True
rn.autoGenTrafficEnabled = False
rn.trafficGenerator.trafficPattern = 3
#rn.osmGraph.drawGraph()


rn.trafficLoader.simulateStartTime = 0 #Time in minutes

reporter.configure(len(rn.osmGraph.nxGraph),rn.osmGraph.nxGraph.number_of_edges(), iterations)
#rn.addTrafficFromData(10) #Time in minutes
print("Last Index Reached: ", rn.trafficLoader.lastIndex)
print("Last Vehicle ID", rn.vehicleGenerator.vehicleId)

print(rn.osmGraph.SDpairs)
print("OD pairs: ", len(rn.osmGraph.SDpairs))
print("OD pairs: ", len(rn.osmGraph.SDpaths))
print("Original pairs", rn.osmGraph.SDpairs)
rn.osmGraph.filterSDpairs()
#rn.osmGraph.drawPathOnMap(False, rn.autoGenTrafficEnabled)
print("Filtered pairs",rn.osmGraph.filteredSDpairs)
#rn.osmGraph.drawPathOnMap()
#rn.osmGraph.drawGraphWithUserTraffic()
rn.osmGraph.drawGraph()

for i in range(20):
    rn.vehicleGenerator.clear()
    rn.osmGraph.clear()
    rn.trafficLoader.clear()

    start = i*5
    window = 5
    rn.trafficLoader.simulateStartTime = start
    rn.addTrafficFromData(window)
    rn.osmGraph.drawPathOnMap(False, rn.autoGenTrafficEnabled)
    rn.osmGraph.drawGraphWithUserTraffic(block=False, figName=str(start)+":"+str(window))
    print("Last Index Reached: ", rn.trafficLoader.lastIndex)
    print("Last Vehicle ID", rn.vehicleGenerator.vehicleId)
    #a = input()

'''for i in range(iterations):
    print("Step: ",i)
    rn.simulateOneStep(i)
    reporter.currentTime = (i+1)*10
    if i%6:
        rn.addTrafficFromData(i/6)

for i in range(len(rn.roadElementGenerator.intersectionList)):
    rn.roadElementGenerator.intersectionList[i].setIntersectionData()


reporter.saveData("Simulate_Data")
reporter.allIntersectionWaitingTime(10,1)
reporter.allIntersectionWaitingTime(10,3,color='b')'''

'''reporter.getIntersectionData(9)
reporter.getIntersectionData(17)
print(reporter.intersectionData[17][1:10])
print(reporter.intersectionData[9][1:10])
print(reporter.intersectionData[21][1:10])
print(reporter.intersectionData[7][1:10])
print(reporter.intersectionData[3][1:10])'''

#reporter.getSimulateData()
#reporter.show()