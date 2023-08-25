import os
import pandas as pd

os.chdir(r"D:\GitHub\vehicle_reidentification")

# =============================================================================
# process match pairs
# =============================================================================

def processMatchPairs(file):
    # read validated ground-truth match pairs dataset
    vdf = pd.read_csv(file_path)
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
    
    # create arrival time based on cycle length, AIY, TUY
    df['arrival_after'] = ((df.AIY < df.TUY) | (df.TUY == 0)).astype(int)
    df.loc[df.arrival_after == 1, 'arrival_time'] = df[['AIY', 'TUY']].min(axis = 1) # arrival after yellow
    df.loc[df.arrival_after == 0, 'arrival_time'] = -df[['AIY', 'TUY']].min(axis = 1) # arrival before yellow
    
    # drop redundant columns
    df.drop(['Parameter', 'Det', 'CycleNum', 'CycleLength', 'YellowTime', 'GreenTime', 'arrival_after'], axis = 1, inplace = True)
    
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

# specify path, file, file name for test/train datasets
data_type = 'test'
# data_type = 'train'

if data_type == 'test':
    input_path = "ignore/data_ground_truth/processed"
    file_path = "data/candidate_match_pairs_ground_truth.csv"
    output_file = "data/processed_match_pairs_ground_truth.txt"
else:
    input_path = "ignore/data_train/processed_sub"
    file_path = "data/candidate_match_pairs_train.csv"
    output_file = "data/processed_match_pairs_train.txt"

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
fdf.to_csv(output_file, sep = '\t', index = False)
