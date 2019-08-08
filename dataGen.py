from Stat_Reporter.StatReporter import Reporter
import pathlib as Path

reporter = Reporter()
reporter.configure(75,125,479)
#reporter.configure(17,24,439)
window = 300
i = 3

#path  = "Simulate_Data/backup/networktype4/7"
#path = "Simulate_Data/backup/networktype5/5"
path  = "Simulate_Data"
size = "1"
type = "/laneChange"
file = path+type+size
reporter.loadFromFile(file)
## Plot Script
reporter.allIntersectionWaitingTime(window, i, 'g', name='Total Wait time: '+type+str(i))
reporter.allVehicleTiming(window,'g', name='Travel time: '+type+str(i))
print(reporter.vehicleData1.shape)
reporter.roadConfiguration(1,type)

type = "/noGuide"
file = path+type+size
reporter.loadFromFile(file)
## Plot Script
reporter.allIntersectionWaitingTime(window, i, 'b',  name='Total Wait time: '+type+str(i))
reporter.allVehicleTiming(window,'b', name='Travel time: '+type)
reporter.roadConfiguration(2,type)
print(reporter.vehicleData1.shape)

'''type = "/signalOnly"
file = path+type+size
reporter.loadFromFile(file)
## Plot Script
reporter.allIntersectionWaitingTime(window, i, 'r',  name='Total Wait time: '+type+str(i))
reporter.allVehicleTiming(window,'r', name='Travel time: '+type+str(i))
reporter.roadConfiguration(3,type)
print(reporter.vehicleData1.shape)'''
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

reporter.show()

