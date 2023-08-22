import os
import pandas as pd
from datetime import datetime

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\vehicle_reidentification")

# read monthly data
df = pd.read_csv("ignore/data_train/2023_01_ISR_19Ave.txt", sep = '\t')
df.TimeStamp = pd.to_datetime(df.TimeStamp, format = '%m-%d-%Y %H:%M:%S.%f')

# check intersection ID
print(df.DeviceID.unique())
df.drop('DeviceID', axis = 1, inplace = True)

# filter month
print(df.TimeStamp.dt.month.unique())
df = df[df.TimeStamp.dt.month == 1]
print(df.TimeStamp.dt.day.unique())

days = list(df.TimeStamp.dt.day.unique())

# =============================================================================
# write events to html file
# =============================================================================

# signal & detector configuration
phase, on = 2, 82
sig = [1, 8, 10]
det = [9, 27, 10, 28, 11, 29, 5, 6]
det_order = {'Parameter': det}

# filter df for phase events
pdf = df.copy(deep = True)[(df.EventID.isin(sig) & (df.Parameter == phase))]
pdf.Parameter = pdf.Parameter.astype(str)

# filter df for detection events
adf = df.copy(deep = True)[((df.EventID == on) & (df.Parameter.isin(det)))]
adf.Parameter = adf.Parameter.astype(str)

# write events to html file for each day
for day in days:
    # filter signal & detection dfs for day
    sig_df = pdf.copy(deep = True)[pdf.TimeStamp.dt.day == day]
    det_df = adf.copy(deep = True)[adf.TimeStamp.dt.day == day]
    
    # plot data continuity
    fig_sig = px.scatter(sig_df, x = 'TimeStamp', y = 'EventID')
    fig_det = px.scatter(det_df, x = 'TimeStamp', y = 'Parameter', category_orders = det_order)
    
    file = '202301' + str(day).zfill(2)
    output_sig = os.path.join("ignore/data_train/plot_data_continuity", file + "_sig.html")
    output_det = os.path.join("ignore/data_train/plot_data_continuity", file + "_det.html")
    
    fig_sig.write_html(output_sig)
    fig_det.write_html(output_det)
        
# =============================================================================
# preprocess training dataset
# =============================================================================

for day in days:
    ddf = df.copy(deep = True)
    ddf = ddf[ddf.TimeStamp.dt.day == day]
    hours = list(ddf.TimeStamp.dt.hour.unique())
    
    for hour in hours:
        if (hour == 2) | ((day == 1) & (hour == 13)): # skip these hours
            pass
        else:
            hdf = ddf.copy(deep = True)
            hdf = hdf[hdf.TimeStamp.dt.hour == hour]
            file = '202301' + str(day).zfill(2) + '_' + str(hour).zfill(2)
            hdf.to_csv(os.path.join("ignore/data_train/raw", file + '.txt'), sep = '\t', index = False)
        