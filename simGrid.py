from Road_Network.RoadNetwork import RoadNetwork
from Stat_Reporter.StatReporter import Reporter
import os

iterations = 800
dummyCycles = 0
rn = RoadNetwork()
reporter = Reporter()

#coordx, coordy, radius = -73.9224, 40.746, 350
coordx, coordy, radius = -73.92, 40.75, 600 # type3
#coordx, coordy, radius =  -73.995830, 40.744612, 600
#coordx, coordy, radius = -73.996575, 40.747477, 600
#coordx, coordy, radius = -73.92, 40.749, 500
rn.buildGraph(coordx, coordy, radius, False)
#rn.buildGraph(-73.93, 40.755, 200)
#rn.buildGraph(-73.9189, 40.7468, 400)


#rn.osmGraph.drawGraph(True, 'time to travel')

#### ---- configuration phase --------###
rn.roadElementGenerator.isGuided = False
rn.roadElementGenerator.laneDirChange = True
rn.roadElementGenerator.preLearned = True
rn.roadElementGenerator.noAgent = False
rn.autoGenTrafficEnabled = True
rn.roadElementGenerator.isNoBound = True
rn.roadElementGenerator.selfLaneChange = False
rn.roadElementGenerator.enaleDependencyCheck = False
rn.roadElementGenerator.noLaneChange = True
manaulchange = True

rn.trafficGenerator.trafficPattern = 8

rn.numOfVehiclesPerBlock = 7
saveMode = True
#Time in minutes
rn.trafficLoader.simulateStartTime = 0
#### ---- configuration phase --------###

imb = 30
cost = 12
location = "Simulate_Data_small grid/temp/"
#location = "Results/cost/"+str(cost)+"/"
#location = "temp/"
if rn.roadElementGenerator.noAgent:
    if rn.roadElementGenerator.selfLaneChange:
        if rn.roadElementGenerator.enaleDependencyCheck:
            file = "noAgent3"
        else:
            file = "noAgent2"
    else:
        file = "noAgent1"
    string = "No agent"
elif rn.roadElementGenerator.laneDirChange:
    if rn.roadElementGenerator.isGuided:
        file = "laneChange2"+str(rn.numOfVehiclesPerBlock)
        string = "Lane change: Guided"

    elif rn.roadElementGenerator.noLaneChange:
        if manaulchange:
            file = "manual1" + str(rn.numOfVehiclesPerBlock)
            string = "manual"
        else:
            file = "signalOnly1" + str(rn.numOfVehiclesPerBlock)
            string = "Signal only"
    else:

        file = "noGuide1"+str(rn.numOfVehiclesPerBlock)
        string = "Lane change: no Guide"
else:
    file = "signalOnly1"+str(rn.numOfVehiclesPerBlock)
    string = "Signal only"

rn.init()
rn.roadElementGenerator.timeToLaneShift = cost
rn.roadElementGenerator.imbalance = imb
rn.createNetwork()

filename = location+file+"/config.txt"
os.makedirs(os.path.dirname(filename), exist_ok=True)
pid = os.getpid()
with open(filename, "w") as f:
    f.write("Map: x:{},y:{},z:{}\n".format(coordx, coordy, radius))
    f.write("Process ID: {}\n".format(pid))
    f.write("Type: {}, Iterations{}\n".format(string,iterations))
    if rn.roadElementGenerator.preLearned:
        f.write("Agent configured\n")
    if rn.roadElementGenerator.isNoBound:
        f.write("Agent not bounded\n")
    f.write("Agent Type: {}\n".format(rn.roadElementGenerator.isNoBound))
    f.write("Sim start: {}\n".format(rn.trafficLoader.simulateStartTime))
    f.write("Number of vehicles per block: {}\n".format(rn.numOfVehiclesPerBlock))
    f.write("Path dependency: depth:{}, threshold:{}, load:{}\n".format(rn.depth,rn.threshold,rn.load_th))
    f.write("Cost of lane Change: {}".format(rn.roadElementGenerator.timeToLaneShift))
    f.write("Aggressiveness: {}".format(rn.roadElementGenerator.imbalance))

    f.write("Running...\n")
#network

#Agents
print("Edges: ", len(rn.roadElementGenerator.roadList))
rn.roadElementGenerator.configureAgents("Tests")
reporter.configure(len(rn.osmGraph.nxGraph), len(rn.roadElementGenerator.roadList), iterations + dummyCycles)

#rn.addTrafficFromData(2) #Time in minutes
#print("Last Index Reached: ", rn.trafficLoader.lastIndex)

#print(rn.osmGraph.SDpairs)
#print("OD pairs: ", len(rn.osmGraph.SDpairs))
#print("OD pairs: ", len(rn.osmGraph.SDpaths))
#print("Original pairs", rn.osmGraph.SDpairs)
#rn.osmGraph.filterSDpairs()
#print("Filtered pairs",rn.osmGraph.filteredSDpairs)
#rn.osmGraph.drawGraphWithUserTraffic()
gen = 3
for i in range(iterations):
    print("Step: ",i, " PID:",pid)
    rn.simulateOneStep(i)
    reporter.currentTime = (i+1)*10
    if (i%gen) == 0 and (i < 390 or i > 410):
        rn.addTrafficFromData((i)/6)
        '''rn.osmGraph.drawPathOnMap(False, rn.autoGenTrafficEnabled, dir='both')
        rn.osmGraph.drawGraphWithUserTraffic(figName='both')

        rn.dependencyG.createVariableDAG(rn.osmGraph.nxGraph, rn.osmGraph.SDpaths)
        rn.dependencyG.drawGraph()'''

    if (i == 2 or i == 430) and manaulchange:
        rn.setRoadconfig(0 if i == 2 else 1)
    '''if i == 50 or i == 450:
        rn.osmGraph.drawPathOnMap(False, rn.autoGenTrafficEnabled)
        rn.osmGraph.drawGraphWithUserTraffic()
        rn.dependencyG.createVariableDAG(rn.osmGraph.nxGraph, rn.osmGraph.SDpaths[-4:len(rn.osmGraph.SDpaths)])
        rn.dependencyG.drawGraph()'''

    if i == 400:
        rn.trafficGenerator.trafficPattern = 7
        gen = 3


    #rn.osmGraph.drawPathOnMap(False, rn.autoGenTrafficEnabled)
    #rn.osmGraph.drawGraphWithUserTraffic()
    #rn.dependencyG.createVariableDAG(rn.osmGraph.nxGraph, rn.osmGraph.SDpaths)
    #rn.dependencyG.drawGraph()



for i in range(dummyCycles):
    print("Step: ", i, " PID:", pid)
    rn.simulateOneStep(i+iterations)
    reporter.currentTime = (i+1+iterations)*10


    #if i == 360:
        #rn.trafficLoader.simulateStartTime = 1080
        #rn.trafficGenerator.trafficPattern = 2

print("Number of vehicles: ", rn.vehicleGenerator.vehicleId, " Index: ", rn.trafficLoader.lastIndex, " Added v: " , rn.trafficLoader.numOfVehicles)
for i in range(len(rn.roadElementGenerator.intersectionList)):
    rn.roadElementGenerator.intersectionList[i].setIntersectionData()

for i in range(len(rn.roadElementGenerator.roadList)):
    rn.roadElementGenerator.roadList[i].setRoadConfigurationData()
    print(" Road id: ", i+1,
          rn.roadElementGenerator.roadList[i].get_num_vehicles(rn.roadElementGenerator.roadList[i].upstream_id, 'T')+
          rn.roadElementGenerator.roadList[i].get_num_vehicles(rn.roadElementGenerator.roadList[i].downstream_id, 'T'),
          len(rn.roadElementGenerator.roadList[i].vehiclesFromDownstream)+len(rn.roadElementGenerator.roadList[i].vehiclesFromUpstream))


'''for i in range(len(rn.vehicleGenerator.vehicleList)):
    if not rn.vehicleGenerator.vehicleList[i].routeFinished:
        rn.vehicleGenerator.vehicleList[i].debugLevel = 3
        rn.vehicleGenerator.vehicleList[i].step(10)'''

if saveMode:
    reporter.saveData(location+file)



with open(filename, "a") as f:
    f.write("Completed:\n")

'''reporter.getIntersectionData(9)
reporter.getIntersectionData(17)
print(reporter.intersectionData[17][1:10])
print(reporter.intersectionData[9][1:10])
print(reporter.intersectionData[21][1:10])
print(reporter.intersectionData[7][1:10])
print(reporter.intersectionData[3][1:10])'''

reporter.getSimulateData()
reporter.show()