from Stat_Reporter.StatReporter import Reporter
import pathlib as Path
import pandas as pd
import numpy as np

meanTime = np.empty(shape=0)
reporter = Reporter()
reporter.configure(75,125,480)
#reporter.configure(12,60,799)
window = 480
j = 1
#data = pd.read_csv("Simulate_Data_new/backup/networktype1/laneChange17/vehicle_data.csv")
#path  = "Simulate_Data_new/backup/networktype2/"
#path = "temp/"
path  = "Complete_Data/network5/results/"
#path = "Simulate_Data_small grid/temp/backup/"
#path = "Simulate_cost_of_lane_change/"
#path = "Complete_Data/cost/"+str(4)+"/"
i = 0
#path = "Results/length/backup/"+str(i)+"/"
#path = "Results/Imbalance Factor/backup/"+str(i)+"/"
#config = "signalOnly"
#config = "laneChange"
config = ["manualnoAgent",
          "manualnoAgent",
          "manualnoAgent",
          "manualnoAgent",
          "manualnoAgent", "laneChange", "laneChange", "manualnoAgent"]
#config = "noAgent"
size = "5"

name = ["15",
        "30",
        "60",
        "90",
        "120",
        #"3",
        #"",
        #"90",
        #"120",
        #"30"
        ]

colorMap = ["g", "b", "r", "gray", "m", "c", "brown", "peru", "purple", "crimson"]
folder = [1,2,3,4,5,6]
'''type = name[0]
file = path+config+type+size
reporter.loadFromFile(file)
## Plot Script
reporter.allIntersectionWaitingTime(window, i, 'r', name='Total Wait time: '+config+str(i))
a = reporter.allVehicleTiming(window,'r', name='Travel time: '+config+str(i))
meanTime = np.append(meanTime,a)
print(reporter.vehicleData1.shape)
reporter.roadConfiguration(1,config)
reporter.plotTravelTimeShift('System of lane Change')
#reporter.plotVehicleTime()

type = name[1]
file = path+config+type+size
reporter.loadFromFile(file)
## Plot Script
reporter.allIntersectionWaitingTime(window, i, 'b',  name='Total Wait time: '+config+str(i))
a = reporter.allVehicleTiming(window,'b', name='Travel time: '+config)
meanTime = np.append(meanTime,a)
reporter.roadConfiguration(2,config)
print(reporter.vehicleData1.shape)

type = name[2]
file = path+config+type+size
reporter.loadFromFile(file)
## Plot Script
reporter.allIntersectionWaitingTime(window, i, 'r',  name='Total Wait time: '+config+str(i))
a = reporter.allVehicleTiming(window,'r', name='Travel time: '+config+str(i))
meanTime = np.append(meanTime,a)
reporter.roadConfiguration(3,config)
print(reporter.vehicleData1.shape)'''

for i in range(len(name)):
    #path = "Complete_Data/depth/" + str(folder[i]) + "/"
    file = path+config[i]+name[i]+size
    reporter.loadFromFile(file)
    reporter.allIntersectionWaitingTime(window, j, colorMap[i], name='Total Wait time: '+config[i]+str(i))
    a = reporter.allVehicleTiming(window, colorMap[i], name='Travel time: ' + config[i] + str(size))
    meanTime = np.append(meanTime, a)
    #reporter.plotTravelTimeShift(i+14)
    reporter.roadConfiguration(i, config)
    print(reporter.vehicleData1.shape)
    #

'''#type = "/laneChange"
file = path+type+"1"
reporter.loadFromFile(file)
## Plot Script
reporter.allIntersectionWaitingTime(window, i, 'gray',  name='Total Wait time: '+type+str(i))
reporter.allVehicleTiming(window,'gray', name='Travel time: '+type+str(i))
#reporter.roadConfiguration(4,type)
print(reporter.vehicleData1.shape)'''
'''type = "/laneChange"
file = path+type+"5"
reporter.loadFromFile(file)
## Plot Script
reporter.allIntersectionWaitingTime(window, i, 'y',  name=type+'Total Wait time'+type+str(i))
reporter.allVehicleTiming(window,'y',name=type+'Travel time')
print(reporter.vehicleData1.shape)
reporter.roadConfiguration(5,type)'''

'''file = path+"/noAgent2"
reporter.loadFromFile(file)
## Plot Script
reporter.allIntersectionWaitingTime(window, i, 'gray',  name='Total Wait time')
reporter.allVehicleTiming(window,'r')
print(reporter.vehicleData1.shape)'''

print(meanTime)
print("Mean time: ", np.mean(meanTime), " Std time: ", np.std(meanTime))

reporter.show()

