import os
import pandas as pd

os.chdir("/Users/prameshpudasaini/Documents/vehicle_reidentification/ignore/data_ground_truth_additional/original")

# read data files
df1 = pd.read_csv("20221206_raw.txt", sep = '\t')
df1 = df1[df1.DeviceID == 46]
df1.drop('DeviceID', axis = 1, inplace = True)
df1.TimeStamp = pd.to_datetime(df1.TimeStamp, format = '%m-%d-%Y %H:%M:%S.%f')

df2 = pd.read_csv("20221214_raw.txt", sep = '\t')
df2 = df2[df2.DeviceID == 46]
df2.drop('DeviceID', axis = 1, inplace = True)
df2.TimeStamp = pd.to_datetime(df2.TimeStamp, format = '%m-%d-%Y %H:%M:%S.%f')

df3 = pd.read_csv("1405-46-2023-03-28-02-45.csv", header = None)
df3.columns = list(df1.columns)
df3.TimeStamp = pd.to_datetime(df3.TimeStamp, format = '%m-%d-%Y %H:%M:%S.%f')

# filter for selected time periods
data1 = df1[df1.TimeStamp.between('2022-12-06 06:45:00', '2022-12-06 07:15:00')]
data2 = df1[df1.TimeStamp.between('2022-12-06 09:45:00', '2022-12-06 12:00:00')]
data3 = df2[df2.TimeStamp.between('2022-12-14 06:45:00', '2022-12-14 07:15:00')]
data4 = df3[df3.TimeStamp.between('2023-03-27 07:00:00', '2023-03-27 14:00:00')]

os.chdir("/Users/prameshpudasaini/Documents/vehicle_reidentification/ignore/data_ground_truth_additional/raw")

# save files
data1.to_csv("20221206_0645_0715.txt", sep = '\t', index = False)
data2.to_csv("20221206_0945_1200.txt", sep = '\t', index = False)
data3.to_csv("20221214_0645_0715.txt", sep = '\t', index = False)
data4.to_csv("20230327_0700_1400.txt", sep = '\t', index = False)
