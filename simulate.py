from Road_Network.RoadNetwork import RoadNetwork
from Stat_Reporter.StatReporter import Reporter
import os

iterations = 360
dummyCycles = 120
rn = RoadNetwork()
reporter = Reporter()

#coordx, coordy, radius = -73.9224, 40.746, 250
coordx, coordy, radius = -73.92, 40.75, 600 # type3
#coordx, coordy, radius = -73.996575, 40.747477, 600
#coordx, coordy, radius = -73.92, 40.749, 500
rn.buildGraph(coordx, coordy, radius)
#rn.buildGraph(-73.93, 40.755, 200)
#rn.buildGraph(-73.9189, 40.7468, 400)


#rn.osmGraph.drawGraph(True, 'time to travel')

#### ---- configuration phase --------###
rn.roadElementGenerator.isGuided = True
rn.roadElementGenerator.laneDirChange = True
rn.roadElementGenerator.preLearned = True
rn.roadElementGenerator.noAgent = False
rn.autoGenTrafficEnabled = True

rn.trafficGenerator.trafficPattern = 1

rn.numOfVehiclesPerBlock = 1
saveMode = True
#Time in minutes
rn.trafficLoader.simulateStartTime = 0
#### ---- configuration phase --------###


if rn.roadElementGenerator.noAgent:
    file = "Simulate_Data/noAgent"
    string = "No agent"
elif rn.roadElementGenerator.laneDirChange:
    if rn.roadElementGenerator.isGuided:
        file = "Simulate_Data/laneChange"+str(rn.numOfVehiclesPerBlock)
        string = "Lane change: Guided"
    else:

        file = "Simulate_Data/noGuide"+str(rn.numOfVehiclesPerBlock)
        string = "Lane change: no Guide"
else:
    file = "Simulate_Data/signalOnly"+str(rn.numOfVehiclesPerBlock)
    string = "Signal only"


filename = file+"/config.txt"
os.makedirs(os.path.dirname(filename), exist_ok=True)
pid = os.getpid()
with open(filename, "w") as f:
    f.write("Map: x:{},y:{},z:{}\n".format(coordx, coordy, radius))
    f.write("Process ID: {}\n".format(pid))
    f.write("Type: {}, Iterations{}\n".format(string,iterations))
    if rn.roadElementGenerator.preLearned:
        f.write("Agent configured\n")
    f.write("Sim start: {}\n".format(rn.trafficLoader.simulateStartTime))
    f.write("Number of vehicles per block: {}\n".format(rn.numOfVehiclesPerBlock))
    f.write("Path dependency: depth:{}, threshold:{}, load:{}\n".format(rn.depth,rn.threshold,rn.load_th))
    f.write("Running...\n")


#network
rn.init()
rn.createNetwork()
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

for i in range(iterations):
    print("Step: ",i, " PID:",pid)
    rn.simulateOneStep(i)
    reporter.currentTime = (i+1)*10
    if i%2 == 0 :
        rn.addTrafficFromData((i)/6)

    if rn.roadElementGenerator.roadList[11].straight_v_list.size != 0:
        for j in rn.roadElementGenerator.roadList[11].straight_v_list_downstream:
            print(j.id)


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


for i in range(len(rn.vehicleGenerator.vehicleList)):
    if not rn.vehicleGenerator.vehicleList[i].routeFinished:
        rn.vehicleGenerator.vehicleList[i].debugLevel = 3
        rn.vehicleGenerator.vehicleList[i].step(10)

if saveMode:
    reporter.saveData(file)



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