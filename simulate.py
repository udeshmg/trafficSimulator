from Road_Network.RoadNetwork import RoadNetwork
from Stat_Reporter.StatReporter import Reporter
import os

#import sys
#old_stdout = sys.stdout

iterations = 361
dummyCycles = 240
rn = RoadNetwork()
reporter = Reporter()

#coordx, coordy, radius = -73.9224, 40.746, 350

coordx, coordy, radius = -73.92, 40.75, 600 # type3
#coordx, coordy, radius =  -73.995830, 40.744612, 600
#coordx, coordy, radius = -73.996575, 40.747477, 600
#coordx, coordy, radius = -73.92, 40.749, 500
rn.buildGraph(coordx, coordy, radius)
#rn.buildGraph(-73.93, 40.755, 200)
#rn.buildGraph(-73.9189, 40.7468, 400)


#rn.osmGraph.drawGraph(True, 'time to travel')

#### ---- configuration phase --------###
rn.roadElementGenerator.isGuided = False
rn.roadElementGenerator.laneDirChange = True
rn.roadElementGenerator.preLearned = True
rn.roadElementGenerator.noAgent = True
rn.autoGenTrafficEnabled = False # Remove when using real data
rn.roadElementGenerator.isNoBound = True
rn.roadElementGenerator.noLaneChange = False

rn.roadElementGenerator.selfLaneChange = False
rn.roadElementGenerator.enableDependencyCheck = False

manualChange = True
rn.manualAllocate = manualChange
rn.trafficGenerator.trafficPattern = 1
rn.demandCalculationFreq = 15
rn.numOfVehiclesPerBlock = 5
rn.osmGraph.minLoad = 100
saveMode = True
#Time in minutes
rn.trafficLoader.simulateStartTime = 0
#### ---- configuration phase --------###

freq = 2
imb = 30
cost = 12
d = 3
length = 5
#location = "Complete_Data_new/network5/Imbalance/"+str(imb)+"/"
location = "Complete_Data/network5/results/"
#location = "Complete_Data/cost/"+str(cost)+"/"
#location = "Complete_Data/minLoad/"+str(rn.osmGraph.minLoad)+"/"
#location = "Complete_Data/depth/"+str(d)+"/"
#location = "temp/"
rn.depFreq = freq
rn.depth = d
rn.dependencyG.len = length
if rn.roadElementGenerator.noAgent:
    if manualChange:
        file = "manualnoAgent"+ str(rn.demandCalculationFreq) + str(rn.numOfVehiclesPerBlock)
        string = "manualnoAgent"
    else:
        file = "SIG" + str(rn.numOfVehiclesPerBlock)
        string = "No agent"
    '''else:
        if rn.roadElementGenerator.enableDependencyCheck:
            file = "HLA1"+str(rn.numOfVehiclesPerBlock)
            string = "No agent"
        elif rn.roadElementGenerator.selfLaneChange:
            file = "LD" + str(rn.numOfVehiclesPerBlock)
            string = "No agent"
        else:
            file = "SIG" + str(rn.numOfVehiclesPerBlock)
            string = "No agent"'''

elif rn.roadElementGenerator.laneDirChange:
    if rn.roadElementGenerator.isGuided:
        file = "laneChange"+str(rn.numOfVehiclesPerBlock)
        string = "Lane change: Guided"
    elif rn.roadElementGenerator.noLaneChange:
        if manualChange:
            file = "manual2" + str(rn.numOfVehiclesPerBlock)
            string = "manual"
        else:
            file = "signalOnly" + str(rn.numOfVehiclesPerBlock)
            string = "Signal only"
    else:
        file = "noGuide3"+str(rn.numOfVehiclesPerBlock)
        string = "Lane change: no Guide"
else:
    file = "signalOnly1"+str(rn.numOfVehiclesPerBlock)
    string = "Signal only"

print("Location: ", location+file)

os.makedirs(os.path.dirname(location+file+"/message.log"), exist_ok=True)
log_file = open(location+file+"/message.log","w")
#sys.stdout = log_file




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

for i in range(iterations):
    print("Step: ",i, " PID:",pid)
    rn.simulateOneStep(i)
    reporter.currentTime = (i+1)*10

    if rn.autoGenTrafficEnabled:
        if (i%2) == 0 and (i > 250 or i < 230):
            rn.addTrafficFromData((i)/6)
    else:
        if (i%3) == 0:
            rn.addTrafficFromData((i) / 6)


    #if i%40== 0 and manualChange:
    #    rn.setRoadconfigPath()

    if i%30 == 0:
        if rn.trafficGenerator.trafficPattern == 2:
            rn.trafficGenerator.trafficPattern = 1
        if rn.trafficGenerator.trafficPattern == 1:
            rn.trafficGenerator.trafficPattern = 2

    #rn.dependencyG.createVariableDAG(rn.osmGraph.nxGraph, rn.osmGraph.SDpaths)
    #rn.dependencyG.drawGraph()



for i in range(dummyCycles):
    print("Step: ", i, " PID:", pid)
    rn.simulateOneStep(i+iterations)
    reporter.currentTime = (i+1+iterations)*10

    if i%40 and manualChange:
        rn.setRoadconfigPath()

    #if i == 360:
        #rn.trafficLoader.simulateStartTime = 1080
        #rn.trafficGenerator.trafficPattern = 2



print("Number of vehicles: ", rn.vehicleGenerator.vehicleId, " Index: ", rn.trafficLoader.lastIndex, " Added v: " , rn.trafficLoader.numOfVehicles)
for i in range(len(rn.roadElementGenerator.intersectionList)):
    rn.roadElementGenerator.intersectionList[i].setIntersectionData()

for i in range(len(rn.roadElementGenerator.roadList)):
    rn.roadElementGenerator.roadList[i].setRoadConfigurationData()
    '''print(" Road id: ", i+1,
          rn.roadElementGenerator.roadList[i].get_num_vehicles(rn.roadElementGenerator.roadList[i].upstream_id, 'T')+
          rn.roadElementGenerator.roadList[i].get_num_vehicles(rn.roadElementGenerator.roadList[i].downstream_id, 'T'),
          len(rn.roadElementGenerator.roadList[i].vehiclesFromDownstream)+len(rn.roadElementGenerator.roadList[i].vehiclesFromUpstream))'''


'''for i in range(len(rn.vehicleGenerator.vehicleList)):
    if not rn.vehicleGenerator.vehicleList[i].routeFinished:
        rn.vehicleGenerator.vehicleList[i].debugLevel = 3
        rn.vehicleGenerator.vehicleList[i].step(10)'''

if saveMode:
    reporter.saveData(location+file)

#sys.stdout = old_stdout
#log_file.close()

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