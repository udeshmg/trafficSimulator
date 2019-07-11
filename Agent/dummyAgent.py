
class dummyAgent():

    def __init__(self, id, intersection):
        self.intersection = intersection

    def actOnIntersection(self):
        self.intersection.step()

    def getOutputData(self):
        print("No data to get")