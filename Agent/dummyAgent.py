
class dummyAgent():

    def __init__(self, id, intersection):
        self.intersection = intersection
        self.num_roads = 2
        self.actionSpace = self.intersection.num_roads
        self.action = 0
        self.iter = 0

    def actOnIntersection(self):
        self.iter += 1

        if self.iter % 2 == 0:
            if self.action == self.actionSpace:
                self.action = 0
            else:
                self.action += 1

        self.intersection.step(self.action)


    def getOutputData(self):
        pass