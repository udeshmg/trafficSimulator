import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


fig = plt.figure()
ax = fig.add_subplot(111)

## the data
N = 4


description = 'Average travel time'
traffic_signals = [248.72, 261.3, 424.04, 566.30]
traffic_signalsStd =   [2, 4, 3, 4]

lane_change_noGuide = [251.27, 242.25, 434.33, 436]
lane_change_noGuideStd =   [2, 4, 3, 4]

lane_change_withGuide = [241.91, 242.25, 274.75, 358.58]
lane_change_withGuideStd =   [2, 4, 3, 4]

file = pd.read_csv("Simulate_Data/networktype4.csv")

col1 = file['Traffic signals only']
col2 = file['Lane change system with no guidance']
col3 = file['Lane change system with guidance']

## necessary variables
ind = np.arange(N)                # the x locations for the groups
width = 0.2                   # the width of the bars

## the bars
rects1 = ax.bar(ind, col1, width,
                color='grey',
                yerr=traffic_signalsStd[0:N],
                error_kw=dict(elinewidth=2,ecolor='black'))


rects2 = ax.bar(ind+width, col2, width,
                color='green',
                yerr=lane_change_noGuideStd[0:N],
                error_kw=dict(elinewidth=2,ecolor='black'))

rects3 = ax.bar(ind+2*width, col3, width,
                    color='blue',
                    yerr=lane_change_withGuideStd[0:N],
                    error_kw=dict(elinewidth=2,ecolor='black'))

# axes and labels
ax.set_xlim(-width,len(ind)+width)
#ax.set_ylim(0,400)
ax.set_ylabel('time')
ax.set_title('Parameter:'+description)
xTickMarks = ['traffic '+str(i) for i in range(1,N+1)]
ax.set_xticks(ind+2*width)
xtickNames = ax.set_xticklabels(xTickMarks)[0:4]
plt.setp(xtickNames, rotation=45, fontsize=8)

## add a legend
ax.legend( (rects1[0], rects2[0], rects3[0]), ('Traffic signals',
                                                'Lane change system with no guidance',
                                                'Lane change system with guidance') )

plt.show()