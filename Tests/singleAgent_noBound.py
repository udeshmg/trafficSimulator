from Agent.ddqn_parallel import DQNAgentNoBound
from Agent.ddqn_lane import DQNAgentLane
from Road_Elements.Intersection3d import Intersection
from Road_Elements.Road import Road
from Road_Elements.Vehicles import VehicleBlock
from Road_Network.RoadElementGenerator import RoadElementGenerator
import  numpy as np
from GUI.Display import Display
from Stat_Reporter.StatReporter import Reporter



def VehicleGenerator(i,numRoads):
    if  i > 6000 and i < 7000:
        random_vehicles = np.array(
            [4, 5,    # 'S' | 'R'
             7, 1,
             2, 4,
             7, 2] )
    elif i > 1000 and i < 8000:
        random_vehicles = np.array(
            [1, 4,   # 'S' | 'R'
             5, 4,
             6, 3,
             3, 4])
    elif i > 500:
        random_vehicles = np.array(
            [1,3,   # 'S' | 'R'
             1,2,
             4,3,
             3,3])

    else:
        random_vehicles = np.array(
            [6, 3,   # 'S' | 'R'
             1, 3,
             1, 1,
             2, 2])

    random_vehicles = random_vehicles.round()

    #np.clip(random_vehicles, 0, 40)
    return random_vehicles[0:2*numRoads]



def randomVehicleGenerator(i,numRoads):
    if  i > 6000 and i < 7000:
        random_vehicles = np.array(
            [np.random.poisson(4, 1)[0], np.random.poisson(5, 1)[0],   # 'S' | 'R'
             np.random.poisson(7, 1)[0], np.random.poisson(1, 1)[0],
             np.random.poisson(2, 1)[0], np.random.poisson(4, 1)[0],
             np.random.poisson(7, 1)[0], np.random.poisson(2, 1)[0]])
    elif i > 4000 and i < 8000:
        random_vehicles = np.array(
            [np.random.poisson(1, 1)[0], np.random.poisson(4, 1)[0],   # 'S' | 'R'
             np.random.poisson(5, 1)[0], np.random.poisson(4, 1)[0],
             np.random.poisson(6, 1)[0], np.random.poisson(3, 1)[0],
             np.random.poisson(3, 1)[0], np.random.poisson(4, 1)[0]])
    elif i > 3000:
        random_vehicles = np.array(
            [np.random.poisson(1, 1)[0], np.random.poisson(3, 1)[0],   # 'S' | 'R'
             np.random.poisson(1, 1)[0], np.random.poisson(2, 1)[0],
             np.random.poisson(4, 1)[0], np.random.poisson(3, 1)[0],
             np.random.poisson(3, 1)[0], np.random.poisson(3, 1)[0]])

    else:
        random_vehicles = np.array(
            [np.random.poisson(6, 1)[0], np.random.poisson(3, 1)[0],   # 'S' | 'R'
             np.random.poisson(1, 1)[0], np.random.poisson(3, 1)[0],
             np.random.poisson(1, 1)[0], np.random.poisson(1, 1)[0],
             np.random.poisson(2, 1)[0], np.random.poisson(2, 1)[0]])

    random_vehicles = random_vehicles.round()

    np.clip(random_vehicles, 0, 40)

    return random_vehicles[0:2*numRoads]
dp = Display()
reporter = Reporter()
#create Roads
numRoads = 4
numIterations = 18000
stepSize = 10
laneDirChange = True
isGuided = False
save = False
load = False

reporter.configure(1,1,numIterations)

roadList = []
rdGenerator = RoadElementGenerator(laneDirChange,isGuided)
intersection = Intersection(1,numRoads)
for i in range(numRoads):
    roadList.append(Road(i+1,6))
    roadList[i].upstream_id = 1
    roadList[i].downstream_id = 0
    intersection.addRoad(roadList[i])

#get number of states
#stateSize, actionSize = rdGenerator.getAgentStateActionSpace(numRoads)

stateSize, actionSize = numRoads*3, 3**numRoads
agent = DQNAgentLane(stateSize,actionSize,intersection,numRoads,1,laneDirChange,isGuided)
agent.debugLvl = 3

if load:
    if laneDirChange:
        agent.load("DDQN_lane_"+str(numRoads)+"noBound")
    else:
        agent.load("DDQN_sig_"+str(numRoads)+"test")


intersection.debugLvl = 3
k = 0
for i in range(numIterations):
    print("Step: ",i)
    agent.actOnIntersection()

    for j in range(len(roadList)):
        roadList[j].step(10)

    vehicleVector = VehicleGenerator(1,numRoads)

    if i % 9000 == 0:
        k += 2


    if numRoads == 3 and i%2 == 0 :
        roadList[0].add_block(VehicleBlock(int(vehicleVector[(k+0)%6]),['L']),'UP')
        roadList[0].add_block(VehicleBlock(int(vehicleVector[(k+1)%6]),['R']),'UP')
        roadList[1].add_block(VehicleBlock(int(vehicleVector[(k+2)%6]),['S']),'UP')
        roadList[1].add_block(VehicleBlock(int(vehicleVector[(k+3)%6]),['L']),'UP')
        roadList[2].add_block(VehicleBlock(int(vehicleVector[(k+4)%6]),['S']),'UP')
        roadList[2].add_block(VehicleBlock(int(vehicleVector[(k+5)%6]),['R']),'UP')
    if numRoads == 4 and i%2 == 0 :
        roadList[0].add_block(VehicleBlock(int(vehicleVector[k%8]), ['S']), 'UP')
        roadList[0].add_block(VehicleBlock(int(vehicleVector[(k+1)%8]), ['R']), 'UP')

        # Road 2
        roadList[1].add_block(VehicleBlock(int(vehicleVector[(k+2)%8]), ['S']), 'UP')
        roadList[1].add_block(VehicleBlock(int(vehicleVector[(k+3)%8]), ['R']), 'UP')

        # Road 3
        roadList[2].add_block(VehicleBlock(int(vehicleVector[(k+4)%8]), ['S']), 'UP')
        roadList[2].add_block(VehicleBlock(int(vehicleVector[(k+5)%8]), ['R']), 'UP')

        # Road 4
        roadList[3].add_block(VehicleBlock(int(vehicleVector[(k+6)%8]), ['S']), 'UP')
        roadList[3].add_block(VehicleBlock(int(vehicleVector[(k+7)%8]), ['R']), 'UP')

    agent.getOutputData()

    if i % 1 == 0:
        for j in range(len(roadList)):
            if roadList[j].upstream_id == 0 or roadList[j].downstream_id == 0:
                roadList[j].remove_block(int(roadList[j].capacity(0, 'IN')), 'S', 0, True)
                roadList[j].remove_block(int(roadList[j].capacity(0, 'IN')), 'R', 0, True)


if save:
    if laneDirChange:
        agent.save("DDQN_lane_"+str(numRoads)+"test")
    else:
        agent.save("DDQN_sig_"+str(numRoads)+"test")


reporter.setIntersectionData(1,intersection.reporter_queue)

reporter.intersectionWaitingTime(1,50,1,'r')
reporter.intersectionWaitingTime(1,50,3,'b')
reporter.show()