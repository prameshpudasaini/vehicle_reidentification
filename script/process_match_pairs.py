import os
import pandas as pd

os.chdir(r"D:\GitHub\vehicle_reidentification")
input_path = "ignore/data_ground_truth/processed"

# =============================================================================
# process match pairs
# =============================================================================

def processMatchPairs(file):
    # read validated ground-truth match pairs dataset
    vdf = pd.read_csv("data/candidate_match_pairs_validated.csv")
    vdf = vdf[(vdf.file == file[:-4]) & (vdf.match == 1)] # discard ".txt" in file
    vdf.drop('match', axis = 1, inplace = True)
    
    # select thru match pairs
    vdf = vdf.drop('rear', axis = 1)
    vdf.dropna(subset = ['stop'], axis = 0, inplace = True)
    
    # sets of adv and stop-bar IDs
    id_adv = set(vdf.adv.astype(int))
    id_stop = set(vdf.stop.astype(int))
    
    # read processed events dataset
    df = pd.read_csv(os.path.join(input_path, file), sep = '\t')
    df.TimeStamp = pd.to_datetime(df.TimeStamp, format = '%Y-%m-%d %H:%M:%S.%f')
    df.drop(['Parameter', 'Det', 'CycleNum', 'CycleLength', 'YellowTime', 'GreenTime'], axis = 1, inplace = True)
    
    # create df for adv and stop-bar det
    adf = df.copy()[df.ID.isin(id_adv)]
    sdf = df.copy()[df.ID.isin(id_stop)]
    
    # add suffix to column names
    adf.columns += '_adv'
    sdf.columns += '_stop'
    
    # merge vdf with adf and sdf
    mdf = pd.merge(vdf, adf, left_on = 'adv', right_on = 'ID_adv')
    mdf = pd.merge(mdf, sdf, left_on = 'stop', right_on = 'ID_stop')
        
    # drop redundant columns
    drop_cols = ['file', 'adv', 'stop', 'travel_time', 'remark', 'ID_adv', 'volume_stop', 'Lane_stop', 'ID_stop']
    mdf.drop(drop_cols, axis = 1, inplace = True)
    
    # add travel time (includes travel time for IDs with "lane change")
    mdf['travel_time'] = (mdf.TimeStamp_stop - mdf.TimeStamp_adv).dt.total_seconds().round(1)
    mdf.drop(['TimeStamp_adv', 'TimeStamp_stop'], axis = 1, inplace = True)
    
    return mdf

# match pairs for each file and append
file_list = os.listdir(input_path)
result = []

for file in file_list:
    print("Processing match pairs for file: ", file)
    result.append(processMatchPairs(file))
    
fdf = pd.concat(result)

# drop rows with Nan values
fdf.dropna(axis = 0, inplace = True)
fdf.reset_index(drop = True, inplace = True)
fdf.to_csv("data/processed_match_pairs_ground_truth.txt", sep = '\t', index = False)
