import pandas
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

%matplotlib widget
#%matplotlib notebook

#header:  timestamp, prediction_timestamp, x_center, y_center

#********************************   OPTIONS   ********************************
START_W_PRED_TIME_MIN = 1; #1: Show no predicted path as you slide base time; 0: Show full pred path as you slide

#df_by_timestamp = pandas.read_csv('28_58_minis_4shark.csv',index_col='timestamp')
#df_by_timestamp = pandas.read_csv('148_MINIS_without140_fixedLenWidth.csv',index_col='timestamp')
df_by_timestamp = pandas.read_csv('/home/diaz/nuplan-devkit/tutorials/diaz_trajectories_simOuts.csv',index_col='timestamp')
#df_by_timestamp = pandas.read_csv('diaz_trajectories_MLplanner.csv',index_col='timestamp')

#*****************************************************************************
selected_timestamp = df_by_timestamp.index[0] #get initial timestamp axis base timestamp
plot_relevant_data = df_by_timestamp[df_by_timestamp.index == selected_timestamp]
if START_W_PRED_TIME_MIN == 1:
    selected_predtime = plot_relevant_data['prediction_timestamp'].min() #get initial prediction_timestamp
else:
    selected_predtime = plot_relevant_data['prediction_timestamp'].max() #get initial prediction_timestamp

plot_relevant_data2 = plot_relevant_data[plot_relevant_data["prediction_timestamp"] <= selected_predtime]

axes_perma_min_basetime = np.min(df_by_timestamp.index)
axes_perma_max_basetime = np.max(df_by_timestamp.index)



#Get selected frame's x, y extents:
plot_lim_xmin = df_by_timestamp['x_center'].min()
plot_lim_xmax = df_by_timestamp['x_center'].max()
plot_lim_ymin = df_by_timestamp['y_center'].min()
plot_lim_ymax = df_by_timestamp['y_center'].max()


fig, ax = plt.subplots()
plt.xlim([plot_lim_xmin,plot_lim_xmax])
plt.ylim([plot_lim_ymin,plot_lim_ymax])
ax.autoscale(False)


# ?? do i need this still:
plt.subplots_adjust(left=0.25, bottom=0.25)

plt.xlabel('X space')
plt.ylabel('Y space')
#plt.title(['timestamp=',selected_timestamp,' Pred. time=',selected_predtime])
scat=ax.scatter(plot_relevant_data2[['x_center']], plot_relevant_data2[['y_center']],marker='.')
ax.set_title(['Timestamp=',selected_timestamp,' Pred. time=',selected_predtime])
    

#timestamp slider:
unique_set_ts=set(df_by_timestamp.index)
unique_list_ts=(list(unique_set_ts))
unique_list_ts_sorted_arr = np.asarray(np.sort(unique_list_ts))
axts = plt.axes([0.25, 0.1, 0.65, 0.03])
ts_slider = Slider(
    ax=axts,
    label='Timestamp',
    valinit=unique_list_ts_sorted_arr[0],
    valmin=unique_list_ts_sorted_arr[0],
    valmax=unique_list_ts_sorted_arr[-1],
    valstep=unique_list_ts_sorted_arr,
    orientation="horizontal"
)
#predicted time slider:
unique_set_pt=set(df_by_timestamp.loc[[selected_timestamp]]['prediction_timestamp'])
unique_list_pt=(list(unique_set_pt))
unique_list_pt_sorted_arr = np.asarray(np.sort(unique_list_pt))
axpt = plt.axes([0.1, 0.25, 0.0225, 0.63])
pt_slider = Slider(
    ax=axpt,
    label='Prediction time',
    valinit=unique_list_pt_sorted_arr[0],
    valmin=unique_list_pt_sorted_arr[0],
    valmax=unique_list_pt_sorted_arr[-1],
    valstep=unique_list_pt_sorted_arr,
    orientation="vertical"
)

#print(plot_relevant_data[['y_center']])
#initial plot
#plot_relevant_data = df_by_basetime.loc[[bt_x]][df_by_basetime.loc[[bt_x]]['overest_risk__base_time']==bt_y]
#line1, = plt.plot(plot_relevant_data[['x_center']], plot_relevant_data[['y_center']], lw=2)


#line1.set_xdata(plot_relevant_data[['x_center']])
#line1.set_ydata(plot_relevant_data[['y_center']])
#line2.set_xdata(plot_relevant_data[['abs_time']])
#line2.set_ydata(plot_relevant_data[['Retrospective_Risk']])
#plt.title(['timestamp=',selected_timestamp,' Pred. time=',selected_predtime])

#fig.canvas.draw_idle()
xx = np.column_stack((plot_relevant_data2[['x_center']].to_numpy(),plot_relevant_data2[['y_center']].to_numpy()))
scat.set_offsets(xx)


# The function to be called anytime a slider's value changes
def update_x(val):
    selected_timestamp = ts_slider.val
    unique_set_pt=set(df_by_timestamp.loc[[selected_timestamp]]['prediction_timestamp'])
    unique_list_pt=(list(unique_set_pt))
    unique_list_pt_sorted_arr = np.asarray(np.sort(unique_list_pt))
    
    plot_relevant_data = df_by_timestamp[df_by_timestamp.index == selected_timestamp]
    if START_W_PRED_TIME_MIN == 1:
        selected_predtime = plot_relevant_data['prediction_timestamp'].min() #get initial prediction_timestamp
    else:
        selected_predtime = plot_relevant_data['prediction_timestamp'].max() #get initial prediction_timestamp
        
    pt_slider.valinit=selected_predtime
    pt_slider.valmin=unique_list_pt_sorted_arr[0]
    pt_slider.valmax=unique_list_pt_sorted_arr[-1]
    pt_slider.valstep=unique_list_pt_sorted_arr
    
    pt_slider.set_val(selected_predtime)
    plot_relevant_data2 = plot_relevant_data[plot_relevant_data["prediction_timestamp"] <= selected_predtime]
    #ax.cla()
    ax.set_title(['Timestamp=',selected_timestamp,' Pred. time=',selected_predtime])
    #ax.set_title(f'Timestamp={selected_timestamp:.2f}, Pred. time={selected_predtime:.2f}')
    
    xx = np.column_stack((plot_relevant_data2[['x_center']].to_numpy(),plot_relevant_data2[['y_center']].to_numpy()))
    scat.set_offsets(xx)
    scat.set_alpha([.5,.5,.5])
    #ax.set_array(plot_relevant_data[['y_center']])
    

    #fig.canvas.draw_idle()

def update_y(val):
    print(1.1)
    selected_predtime = pt_slider.val
    plot_relevant_data = df_by_timestamp[df_by_timestamp.index == ts_slider.val]
    plot_relevant_data2 = plot_relevant_data[plot_relevant_data["prediction_timestamp"] <= selected_predtime]
    #ax.cla()
    ax.set_title(['Timestamp=',selected_timestamp,' Pred. time=',selected_predtime])
    
    xx = np.column_stack((plot_relevant_data2[['x_center']].to_numpy(),plot_relevant_data2[['y_center']].to_numpy()))
    scat.set_offsets(xx)
    

# register the update function with each slider
#update_bty_slider()
ts_slider.on_changed(update_x)
pt_slider.on_changed(update_y)
plt.show()

