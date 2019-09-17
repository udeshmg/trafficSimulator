import pandas as pd
from pathlib import Path

class TrafficCollector:

    def __init__(self):
        self.vehicleFileLocation = Path("NycData/green_tripdata_2015-01.csv")
        #self.vehicleFileLocation = Path("Z:/Downloads/green_tripdata_2015-01.csv")
        self.lastTimeRead = 0
        self.lastIndex = 0
        self.numOfVehicles = 0
        self.simulateStartTime = 0

    def clear(self):
        self.lastTimeRead = 0
        self.lastIndex = 0
        self.numOfVehicles = 0
        self.simulateStartTime = 0

    def printData(self):
        print(self.vehicleData.iloc[[1], [0]])

        print(self.vehicleData.iat[1, 0])
        #print(int(self.vehicleData.iat[1, 3]))
        print(int(self.vehicleData.iat[1, 1]))

    def loadUserData(self):
        print("Loading Data...")
        self.vehicleData = pd.read_csv(self.vehicleFileLocation)
        print("Data acquisition complete.")

    def getData(self,currTime):
        vehicleList = []
        temp = 0
        while currTime+self.simulateStartTime  > self.lastTimeRead:
            #print("Index: ", self.lastIndex)
            time = self.decodeTime(self.vehicleData.iat[self.lastIndex, 1])
            if time >= self.simulateStartTime:
                sLong = self.vehicleData.iat[self.lastIndex, 5]
                sLat = self.vehicleData.iat[self.lastIndex, 6]
                dLong = self.vehicleData.iat[self.lastIndex, 7]
                dLat = self.vehicleData.iat[self.lastIndex, 8]
                temp += 1
                self.numOfVehicles += 1
                vehicleList.append([sLong,sLat,dLong, dLat])
            self.lastTimeRead = time
            self.lastIndex += 1
        #print("Vehicle List", vehicleList)
        print(" Index: ", self.lastIndex, " at time: ", currTime, " Total Vehicles ", temp)
        return vehicleList

    def decodeTime(self, timeString):
        hour = 0
        minute = 0
        hourIndexReached = False
        minuteIndexReached = False
        multiplier = 1
        for index, iter in enumerate(timeString):
            if minuteIndexReached:
                minute = minute*multiplier+int(iter)
                multiplier *= 10

            if iter == ':':
                minuteIndexReached = True
                hourIndexReached = False
                multiplier = 1

            if hourIndexReached:
                hour = hour*multiplier + int(iter)
                multiplier *= 10

            if iter == ' ':
                hourIndexReached = True
        return hour*60+minute
