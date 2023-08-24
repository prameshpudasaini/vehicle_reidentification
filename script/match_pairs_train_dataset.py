import os
import pandas as pd
import numpy as np

os.chdir(r"D:\GitHub\vehicle_reidentification")

path_events_new_IDs = "ignore/data_train/processed"
file_list_new_IDs = os.listdir(path_events_new_IDs)

# =============================================================================
# pre-process match pairs with old IDs
# =============================================================================

# old match pairs: subsets
path_match_old_IDs = "ignore/data_train/candidate_match_pairs_old_IDs"
df1 = pd.read_csv(os.path.join(path_match_old_IDs, "cristina.csv"))
df2 = pd.read_csv(os.path.join(path_match_old_IDs, "pramesh.csv"))

# fill forward day and hour columns
fill_cols = ['day', 'hour']
df1[fill_cols] = df1[fill_cols].fillna(method = 'ffill')
df2[fill_cols] = df2[fill_cols].fillna(method = 'ffill')

# combine df1 and df2
df3 = pd.concat([df1, df2])

# adv-stop match pairs in sub set
tdf3 = df3.drop('rear', axis = 1)
tdf3.dropna(subset = ['stop'], axis = 0, inplace = True)

# adv-rear match pairs in full set
rdf3 = df3.drop('stop', axis = 1)
rdf3.dropna(subset = ['rear'], axis = 0, inplace = True)

# old match pairs: full set
df0 = pd.read_csv(os.path.join(path_match_old_IDs, "full.csv"))
df0.drop('remark', axis = 1, inplace = True)

# adv-stop match pairs in full set
tdf0 = df0.drop('rear', axis = 1)
tdf0.dropna(subset = ['stop'], axis = 0, inplace = True)

# adv-rear match pairs in full set
rdf0 = df0.drop('stop', axis = 1)
rdf0.dropna(subset = ['rear'], axis = 0, inplace = True)

# merge match pairs between full and sub sets
cols_adv_stop = ['day', 'hour', 'adv', 'stop']
cols_adv_rear = ['day', 'hour', 'adv', 'rear']
tmdf = pd.merge(tdf0, tdf3, on = cols_adv_stop, how = 'inner')
rmdf = pd.merge(rdf0, rdf3, on = cols_adv_rear, how = 'inner')

# update match between full and sub sets
df0['match_stop'] = df0.apply(
    lambda row: 1 if (row[cols_adv_stop] == tmdf[cols_adv_stop]).all(axis = 1).any() else 0, axis = 1
)
df0['match_rear'] = df0.apply(
    lambda row: 1 if (row[cols_adv_rear] == rmdf[cols_adv_rear]).all(axis = 1).any() else 0, axis = 1
)

# filter df0 for adv-stop and adv-rear matches
fdf0 = df0.copy(deep = True)[((df0.match_stop == 1) | (df0.match_rear == 1) | (df0.match == 1))]
fdf0.drop(['match', 'match_stop', 'match_rear'], axis = 1, inplace = True)

days = fdf0.day.unique()

# =============================================================================
# process match pairs with old IDs for timestamps
# =============================================================================

path_events_old_IDs = "ignore/data_train/processed_old_IDs"
file_list_old_IDs = os.listdir(path_events_old_IDs)

temp = []
for file in file_list_old_IDs:
    temp.append(pd.read_csv(os.path.join(path_events_old_IDs, file), sep = '\t'))
    
# concat df for old IDs
ocdf = pd.concat(temp)
ocdf.TimeStamp = pd.to_datetime(ocdf.TimeStamp, format = '%Y-%m-%d %H:%M:%S.%f')

# add day, hour variables
ocdf['day'] = ocdf.TimeStamp.dt.day
ocdf['hour'] = ocdf.TimeStamp.dt.hour

# select required columns from omdf and filter days
ocdf = ocdf[['day', 'hour', 'ID', 'Lane', 'TimeStamp']]
ocdf = ocdf[ocdf.day.isin(days)]

# merge omdf with fdf0 for adv timestamps; rename merged columns
omdf = pd.merge(
    fdf0, ocdf, 
    left_on = ['day', 'hour', 'adv'], 
    right_on = ['day', 'hour', 'ID'], 
    how = 'outer'
)
omdf.rename(columns = {'ID': 'ID_adv', 'Lane': 'Lane_adv', 'TimeStamp': 'TimeStamp_adv'}, inplace = True)

# merge omdf with fdf0 for stop timestamps; rename merged columns
omdf = pd.merge(
    omdf, ocdf, 
    left_on = ['day', 'hour', 'stop'], 
    right_on = ['day', 'hour', 'ID'], 
    how = 'outer'
)
omdf.rename(columns = {'ID': 'ID_stop', 'Lane': 'Lane_stop', 'TimeStamp': 'TimeStamp_stop'}, inplace = True)

# merge omdf with fdf0 for rear timestamps; rename merged columns
omdf = pd.merge(
    omdf, ocdf, 
    left_on = ['day', 'hour', 'rear'], 
    right_on = ['day', 'hour', 'ID'], 
    how = 'outer'
)
omdf.rename(columns = {'ID': 'ID_rear', 'Lane': 'Lane_rear', 'TimeStamp': 'TimeStamp_rear'}, inplace = True)

# drop redundant rows with Nan for adv IDs
omdf.dropna(subset = ['adv'], axis = 0, inplace = True)

# check if adv-stop lanes are same
omdf.Lane_stop.fillna('NaN_placeholder', inplace = True)
omdf['lane_check'] = (omdf.Lane_adv == omdf.Lane_stop) | (omdf.Lane_rear == 'LT') # all true

# all adv-stop lanes match; drop lane parameters
omdf.drop(['Lane_adv', 'Lane_stop', 'Lane_rear', 'lane_check'], axis = 1, inplace = True)

# drop columns starting with ID
omdf.drop(['ID_adv', 'ID_stop', 'ID_rear'], axis = 1, inplace = True)

# adv-stop pairs: check whether travel time and difference of timestamps match
tomdf = omdf.dropna(subset = ['stop'], axis = 0)
tomdf['check'] = tomdf.travel_time == (tomdf.TimeStamp_stop - tomdf.TimeStamp_adv).dt.total_seconds()

# adv-rear pairs: check whether travel time and difference of timestamps match
romdf = omdf.dropna(subset = ['rear'], axis = 0)
romdf['check'] = romdf.travel_time == (romdf.TimeStamp_rear - romdf.TimeStamp_adv).dt.total_seconds()

# =============================================================================
# match pairs between old IDs and new IDs based on timestamps
# =============================================================================

ndf = pd.read_csv("ignore/data_train/candidate_match_pairs.csv")

# convert timestamps to datetime
date_cols = ['TimeStamp_adv', 'TimeStamp_stop', 'TimeStamp_rear']
ndf[date_cols] = ndf[date_cols].apply(pd.to_datetime)
ndf = ndf[ndf.TimeStamp_adv.dt.day.isin(days)]

# adv-stop and adv-rear pairs
tndf = ndf.dropna(subset = ['stop'], axis = 0)
rndf = ndf.dropna(subset = ['rear'], axis = 0)

cols_adv_stop = ['TimeStamp_adv', 'TimeStamp_stop']
cols_adv_rear = ['TimeStamp_adv', 'TimeStamp_rear']

# update match between tomdf & tndf, romdf & rndf
tndf['match_stop'] = tndf.apply(
    lambda row: 1 if (row[cols_adv_stop] == tomdf[cols_adv_stop]).all(axis = 1).any() else 0, axis = 1
)
rndf['match_rear'] = rndf.apply(
    lambda row: 1 if (row[cols_adv_rear] == romdf[cols_adv_rear]).all(axis = 1).any() else 0, axis = 1
)

print("Total adv-stop matches:", sum(tndf.match_stop))
print("Total adv-rear matches:", sum(rndf.match_rear))

# concat tndf and rndf
df = pd.concat([tndf, rndf])

# add match variable as boolean
df['match'] = ((df.match_stop == 1) | (df.match_rear == 1)).astype(int)

# sort by timestamp and remove redundant columns
df.sort_values(by = 'TimeStamp_adv', inplace = True)
df.drop(['TimeStamp_adv', 'TimeStamp_stop', 'TimeStamp_rear', 'match_stop', 'match_rear'], axis = 1, inplace = True)

df.to_csv("ignore/data_train/candidate_match_pairs_updated_from_old_IDs.csv", index = False)
