import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



fig = plt.figure()
ax = fig.add_subplot(111)

## the data
N = 3


description = 'Average travel time'
'''traffic_signals = [248.72, 261.3, 424.04, 566.30]
traffic_signalsStd =   [2, 4, 3, 4]

lane_change_noGuide = [251.27, 242.25, 434.33, 436]
lane_change_noGuideStd =   [2, 4, 3, 4]

lane_change_withGuide = [241.91, 242.25, 274.75, 358.58]
lane_change_withGuideStd =   [2, 4, 3, 4]'''

traffic_signals = [248.72, 261.3, 424.04, 566.30]
traffic_signalsStd =   [0, 0, 0, 0]

lane_change_noGuide = [251.27, 242.25, 434.33, 436]
lane_change_noGuideStd =   [0, 0, 0, 0]

lane_change_withGuide = [241.91, 242.25, 274.75, 358.58]
lane_change_withGuideStd =   [0, 0, 0, 0]


file = pd.read_csv("Complete_Data/network6/backup/networktype4.csv")

col1 = file['Traffic signals only']
col2 = file['Lane change system with no guidance']
col3 = file['Lane change system with guidance']
col4 = file['demand']



## necessary variables
ind = np.arange(N)                # the x locations for the groups
width = 0.2                   # the width of the bars


a = [{"hatch":'x'}, {"hatch":'+'}, {"hatch":'.'}, {"hatch": "/"}]

kwargs = a[0]
## the bars
rects1 = ax.bar(ind, col1, width,
                color='gray',
                yerr=traffic_signalsStd[0:N],
                error_kw=dict(elinewidth=2,ecolor='black'), **kwargs)

kwargs = a[1]
rects2 = ax.bar(ind+2*width, col2, width,
                color='green',
                yerr=lane_change_noGuideStd[0:N],
                error_kw=dict(elinewidth=2,ecolor='black'), **kwargs)

kwargs = a[2]
rects3 = ax.bar(ind+3*width, col3, width,
                    color='blue',
                    yerr=lane_change_withGuideStd[0:N],
                    error_kw=dict(elinewidth=2,ecolor='black'), **kwargs)

kwargs = a[3]
rects4 = ax.bar(ind+width, col4, width,
                    color='red',
                    yerr=lane_change_withGuideStd[0:N],
                    error_kw=dict(elinewidth=2,ecolor='black'), **kwargs)

# axes and labels
ax.set_xlim(-width,len(ind)+width)
ax.set_ylim(0,8.6)
ax.set_ylabel('Average travel time (min)',fontsize=24)
ax.set_xlabel('Up sampled factor',fontsize=24)
#ax.set_title('Parameter:'+description)
#xTickMarks = [5,6,7]
xTickMarks = [6,7,8]
ax.set_xticks(ind+2*width)
xtickNames = ax.set_xticklabels(xTickMarks)[0:4]

ax.tick_params(axis="y", labelsize=24)
plt.setp(xtickNames, fontsize=24)

## add a legend
ax.legend( (rects4[0], rects1[0], rects2[0], rects3[0]), ('DLA','TS',
                                                'LD',
                                                'HLA'), fontsize=24 )
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().set_size_inches(7,7)
plt.show()