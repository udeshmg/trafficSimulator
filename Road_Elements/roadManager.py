from Road_Elements.Road import Road
from Road_Elements.Vehicles import VehicleBlock
from Agent.laneMangingAgent import DQNA_laneManager


def randomTraffic(iter):
    if iter < 4000:
        vec = [10,10]
    elif iter < 6000:
        vec = [3,14]
    elif iter < 8000:
        vec = [15,3]
    else:
        vec = [7,7]

    return vec

class roadManager():

    def __init__(self):
        self.road = Road(1, 6, 10)
        self.state_size = 3
        self.actio_size = 3
        self.step_size = 10
        self.iter = 0
        self.output_cycle_time = 2
        self.agent = DQNA_laneManager(self.state_size, self.actio_size, self.road, 1)

        self.road.upstream_id = 1
        self.road.downstream_id = 2
        self.road.limitVehicles = True

    def simulateStep(self):

        self.agent.actOnRoad()
        # simulate road step

        for i in range(self.output_cycle_time):
            vec = randomTraffic(self.iter)

            if self.iter%4:
                self.road.add_block(VehicleBlock(vec[0], 'S'), 'UP')
                self.road.add_block(VehicleBlock(vec[1], 'S'), 'DOWN')

            self.road.step(self.step_size)

            if self.iter%4:
                self.road.remove_block(int(self.road.capacity(1, 'IN')), 'S', 1, True)
                print("Vehicles from upstream: ", int(self.road.capacity(1, 'IN')), self.road.get_in_lanes_num(1))
                self.road.remove_block(int(self.road.capacity(2, 'IN')), 'S', 2, True)
                print("Vehicles from downstream ", int(self.road.capacity(2, 'IN')))


        self.iter += 1

        print("Road data", self.road.upTraffic, self.road.downTraffic, self.road.get_in_lanes_num(self.road.upstream_id))
        self.agent.getOutputData()

    def save(self, name):
        self.agent.save(name)

    def load(self, name):
        self.agent.load(name)

rd = roadManager()


rd.load("weights_lane_adj")

for i in range(10000):
    print("Step top: ", i)
    rd.simulateStep()

rd.save("weights_lane_adj")
rd.agent.plotReward()
