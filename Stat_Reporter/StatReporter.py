import numpy as np
from GUI.Display import Display
from pathlib import Path
import pandas as pd
import os

class Reporter():
    __instance = None

    @staticmethod
    def getInstance():
        if Reporter.__instance == None:
            Reporter()
        return Reporter.__instance

    def __init__(self):
        if Reporter.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            self.vehicleData = np.empty(shape=(1,4))
            Reporter.__instance = self
            self.dp = Display()
            self.currentTime = 0

    def loadFromFile(self,string):
        for i in range(self.nodes):
            data = pd.read_csv(Path(string+'/intersection'+str(i+1)+'.csv'))
            self.intersectionData[i] = data

        data = pd.read_csv(Path(string + '/vehicle_data.csv'))

        road = pd.read_csv(Path(string + '/road.csv'))
        self.roadData1 = np.zeros(shape=road.shape)
        self.roadData1 = np.array(road)
        #print(data)
        self.vehicleData1 = np.zeros(shape=data.shape)
        #print(len(self.vehicleData1))
        self.vehicleData1 = np.array(data)
        #print(self.vehicleData1)


    def configure(self, nodes,edges,iterations):
        self.nodes = nodes
        self.iterations = iterations
        self.roadData = np.zeros(shape=(edges+1, iterations),dtype=int)
        self.intersectionData = np.zeros(shape=(nodes, iterations+1, 5),dtype=int)


    def addVehicleData(self, data):
        data[-1].insert(0,self.currentTime)
        self.vehicleData = np.append(self.vehicleData, data, axis=0)

    def setIntersectionData(self,id,data):
        self.intersectionData[id-1] = data

    def setRoadData(self,id,data):
        self.roadData[id-1] = data

    def getSimulateData(self):
        print(self.vehicleData)

    def getIntersectionData(self,id):
        print(self.intersectionData[id-1])

    def saveData(self,string):
        try:
            os.mkdir(string)
            print("Directory ", string, " Created ")
        except FileExistsError:
            print("Directory ", string, " already exists")

        for i in range(self.nodes):
            np.savetxt(Path(string+'/intersection'+str(i+1)+'.csv'), self.intersectionData[i], delimiter=",", fmt='%10.2f')

        np.savetxt(Path(string + '/vehicle_data.csv'), self.vehicleData, delimiter=",",
                   fmt='%10.2f')

        np.savetxt(Path(string + '/road.csv'), self.roadData, delimiter=",",
                   fmt='%10.2f')


    def intersectionWaitingTime(self,id,window,column=1,color='r',name='title name'):
        mvgAvg1 = 0
        mvgAvg2 = 0
        waitTime = np.empty(shape=0)
        for i in range(len(self.intersectionData[id-1][:,column])):
            mvgAvg1 = 0
            mvgAvg2 = 0
            for j in range(max(0,i-window),i):
                mvgAvg1 += self.intersectionData[id-1][j][column]
                mvgAvg2 += self.intersectionData[id-1][j][column+1]

            waitTime = np.append(waitTime, mvgAvg1/max(1,mvgAvg2))
        self.dp.single_arr(waitTime,id,edgeclr=color,name=name)

    def allIntersectionWaitingTime(self,window,column=1,color='r',name='title name'):
        waitTime = np.empty(shape=0)
        for i in range(1,len(self.intersectionData[0][:,column])):
            mvgAvg1 = [0 for node in range(self.nodes)]
            mvgAvg2 = [0 for node in range(self.nodes)]
            for k in range(self.nodes):
                for j in range(max(1,i-window),i):
                    mvgAvg1[k] += self.intersectionData[k][j][column]
                    mvgAvg2[k] += self.intersectionData[k][j][column+1]
            avgTime = sum(mvgAvg1)/max(1,sum(mvgAvg2))
            waitTime = np.append(waitTime, avgTime)
        self.dp.single_arr(waitTime,100,edgeclr=color,name=name)


    def roadConfiguration(self,figNum,name='config'):
        self.dp.colorMap(self.roadData1,figNum,name)


    def allVehicleTiming(self,window,color='g',name='Vehicle completion data'):
        vehicleTime = np.empty(shape=0)
        timeStep = 0

        k = 0
        for i in range(1,self.iterations+1):
            startIndex = max(1,i-window)
            lastIndex = i

            while k < len(self.vehicleData1) and self.vehicleData1[k][0] < startIndex*10:
                k += 1
            j = k
            sum =0
            count = 0
            while j < len(self.vehicleData1) and self.vehicleData1[j][0] < lastIndex*10 :
                sum += self.vehicleData1[j][2]
                count += 1
                j += 1

            vehicleTime = np.append( vehicleTime, sum/max( 1,count))

        #print(vehicleTime)
        self.dp.single_arr(vehicleTime, 50, edgeclr=color, name=name)

    def show(self):
        self.dp.show()



'''from Stat_Reporter.StatReporter import Reporter
reporter = Reporter()
reporter.configure(51,719)'''


import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))