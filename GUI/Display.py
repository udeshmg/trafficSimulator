import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import colors
DEBUG_LEVEL = 1



class Display:

    def __init__(self):
        pass
        print("Setting up display")
        # plt.axis([0, 10, -1000, 1000])

    @staticmethod
    def plot(vehicle_data):
        plt.figure(2)
        plt.axis([0, 7, 0, 140])
        obj = ['1', '2', '3', '4', '5', '6', '7', '8']
        bar_list = plt.bar(obj, height=vehicle_data, align='center', width=0.8)
        bar_list[0].set_color('r')
        bar_list[1].set_color('r')
        bar_list[2].set_color('g')
        bar_list[3].set_color('g')
        bar_list[4].set_color('b')
        bar_list[5].set_color('b')
        bar_list[6].set_color('y')
        bar_list[7].set_color('y')

        plt.pause(0.0005)
        plt.clf()

    def show(self):
        plt.show()

    @staticmethod
    def debug_info(string, level):
        if DEBUG_LEVEL < level:
            print(string)

    @staticmethod
    def multi_scatter_plot(average_val, i, figure_num=6):
        plt.figure(figure_num)
        plt.scatter(i, average_val[0], edgecolors='g', c=None, s=1)
        plt.scatter(i, average_val[1], edgecolors='r', c=None, s=1)
        plt.scatter(i, average_val[2], edgecolors='y', c=None, s=1)
        plt.scatter(i, average_val[3], edgecolors='b', c=None, s=1)

    @staticmethod
    def single_arr(arr, figure_num=6, edgeclr='b', name='Add the title name', legendS=('LD','HLA','DLA','TS','Temp')):
        fig = plt.figure(figure_num)

        #plt.title(name)
        plt.xlim(390,720)
        plt.ylim(3,8)
        #plt.legend('TS')
        plt.xlabel('time steps',fontsize=16)
        plt.ylabel('Average travel time \n (min)',fontsize=16)
        plt.tick_params(labelsize=16)
        #plt.legend('Guided lane change')
        i = [i for i in range(len(arr))]
        plt.scatter(i, arr/60, edgecolors=edgeclr, c=None, s=1)
        ax = fig.axes
        #ax[0].legend(legendS)
        plt.rcParams["legend.markerscale"] = -0.5
        ax[0].legend(legendS, prop={'size': 12})
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.gcf().subplots_adjust(left=0.2)



    @staticmethod
    def heatMap(arr, figure_num):
        plt.figure(figure_num)
        cmap = colors.ListedColormap(['green', 'red','gray'])
        bounds = [2,3,4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(arr, cmap=cmap, interpolation='nearest',norm=norm)

    @staticmethod
    def figure(average_val, i, figure_num=5):
        plt.figure(figure_num)
        plt.scatter(i, average_val, edgecolors='r', c=None, s=1)

    @staticmethod
    def figure_time(average_val, i, figure_num=3):
        plt.figure(figure_num)
        #plt.axis([60000,12000,0,140])
        plt.scatter(i, average_val, edgecolors='g', c=None, s=1)

    @staticmethod
    def plot3d(array2d):
        x = range(array2d.shape[1])
        y = range(array2d.shape[0])
        print('szie', x, y)
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x, y)

        # scaled = np.divide(array2d, np.amax(array2d))
        color_bar = ax.scatter(X, Y, array2d)
        ax.set_title('Q table')
        #cbar = plt.colorbar(color_bar)
        #cbar.set_label("Values (units)")
        plt.show()

    @staticmethod
    def colorMap(array,figNum,name='config'):
        plt.figure(figNum)
        plt.ylabel('Road ID')
        plt.xlabel('Time step')
        plt.imshow(array, interpolation="nearest", origin="upper",aspect='auto')
        plt.colorbar()

    @staticmethod
    def histogram(arr,title=5):
        plt.figure(title)
        #plt.ylim(0,max(arr))
        plt.xlim(0.8,10)
        #plt.title(title)
        plt.hist(arr, color='blue', edgecolor='black', bins=[i/10 for i in range(1,150)])
        plt.tick_params( labelsize=36)
        plt.xlabel('DFFT',fontsize=36)
        plt.ylabel('number of vehicles',fontsize=36)
        plt.gcf().subplots_adjust(bottom=0.12, top=0.99)
        plt.gcf().subplots_adjust(left=0.3)
        plt.gcf().set_size_inches(7,7)

    @staticmethod
    def perVehiclePlot(arr,index,title):
        plt.figure(14)
        plt.ylim(0,500)
        plt.title(title)
        plt.hist(arr, color='blue', edgecolor='black', bins=index)
        plt.xlabel('vehicle ID')
        plt.ylabel('Travel time')

    @staticmethod
    def plotwithVar(x, y, var, title):
        plt.figure(15)
        plt.ylim(300, max(y) + max(var) + 5)
        plt.errorbar(x, y, var, marker='^')
        plt.ylabel('Average travel time')
        plt.xlabel('Time steps to trigger traffic imbalance')
        plt.show()




