import os
import pandas as pd

os.chdir(r"D:\GitHub\vehicle_reidentification")

# compute arrival volume at advance det by cycle
def arrivalVolumeCycle(file, raw_path, cycle_path):
    # raw df
    rdf = pd.read_csv(os.path.join(raw_path, file), sep = '\t')
    rdf.TimeStamp = pd.to_datetime(rdf.TimeStamp, format = '%Y-%m-%d %H:%M:%S.%f')
    
    # cycle df
    cdf = pd.read_csv(os.path.join(cycle_path, file), sep = '\t')
    cdf[['YST', 'YST_NC']] = cdf[['YST', 'YST_NC']].apply(pd.to_datetime)
    
    # min, max yellow start times
    yst_min = cdf.YST.iloc[0]
    yst_max = cdf.YST_NC.iloc[-1]
    
    # filter actuations over advance det and between min, max yellow start times
    rdf = rdf[(rdf.EventID == 82) & (rdf.Parameter.isin([27, 28, 29]))]
    rdf = rdf[(rdf.TimeStamp > yst_min) & (rdf.TimeStamp < yst_max)]
    
    # rename YST to timestamp
    cdf.rename(columns = {'YST': 'TimeStamp'}, inplace = True)
    cdf = cdf[['CycleNum', 'TimeStamp']]
    
    # append rdf and cdf
    mdf = pd.concat([rdf, cdf]).sort_values(by = 'TimeStamp')
    mdf.CycleNum.ffill(inplace = True)
    
    # compute cycle volume
    cycle_vol = mdf.groupby('CycleNum').agg(volume_adv_cycle = ('TimeStamp', 'size')).reset_index()
    
    return cycle_vol

# # compute arrival volume on yellowa at advance det  
# def arrivalVolumeYellow(xdf):  
#     # convert yellow arrival as before and after
#     xdf['arrival_after_yellow'] = xdf['arrival_after_yellow'].replace({1: 'after', 0: 'before'})
    
#     # filter on det over advance detector
#     xdf = xdf[xdf.Det == 'adv']
    
#     # number of vehicle arrivals by before/after yellow in each cycle
#     cycle_vol = xdf.groupby(['CycleNum', 'arrival_after_yellow']).agg(volume_yellow = ('TimeStamp', 'size')).reset_index()
#     cycle_vol = cycle_vol.pivot(index = 'CycleNum', columns = 'arrival_after_yellow', values = 'volume_yellow').reset_index()
    
#     # create dummy df for number of cycles in an hour
#     max_cycle = int(max(cycle_vol.CycleNum))
#     ydf = pd.DataFrame({'CycleNum': range(1, max_cycle + 1)})
    
#     # merge ydf and cycle vol
#     ydf = ydf.merge(cycle_vol, how = 'outer')
    
#     # shift arrival volume after yellow in next cycle
#     ydf['after_next_cycle'] = ydf.after.shift(-1)
#     ydf.fillna(0, inplace = True)
    
#     # compute arrival volume on yellow
#     ydf['volume_adv_yellow'] = ydf.before + ydf.after_next_cycle
#     ydf = ydf[['CycleNum', 'volume_adv_yellow']]
    
#     return ydf

# =============================================================================
# process match pairs
# =============================================================================

def bulkProcessMatchPairs(data_type):
    # input, file, and output paths by data type
    if data_type == 'test':
        input_path = "ignore/data_ground_truth/processed"
        file_path = "data/candidate_match_pairs_ground_truth.csv"
        raw_path = "ignore/data_ground_truth/raw"
        cycle_path = "ignore/data_ground_truth/cycle"
        output_file = "data/processed_match_pairs_ground_truth.txt"
    else:
        input_path = "ignore/data_train/processed_sub" # sub folder
        file_path = "data/candidate_match_pairs_train.csv"
        raw_path = "ignore/data_train/raw"
        cycle_path = "ignore/data_train/cycle"
        output_file = "data/processed_match_pairs_train.txt"

    # match pairs for each file and append
    file_list = os.listdir(input_path)
    result = []
    
    for file in file_list:
        print("Processing match pairs for file: ", file)
        
        # read validated match pairs dataset
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
        df['arrival_after_yellow'] = ((df.AIY < df.TUY) | (df.TUY == 0)).astype(int)
        df.loc[df.arrival_after_yellow == 1, 'arrival_time'] = df[['AIY', 'TUY']].min(axis = 1) # arrival after yellow
        df.loc[df.arrival_after_yellow == 0, 'arrival_time'] = -df[['AIY', 'TUY']].min(axis = 1) # arrival before yellow
        
        # get cycle volume at advance det
        volume_adv_cycle = arrivalVolumeCycle(file, raw_path, cycle_path)
        
        # merge arrival volume on yellow to df
        df = df.merge(volume_adv_cycle, how = 'outer')
        
        # drop redundant columns
        drop_cols = ['Parameter', 'Det', 'CycleNum', 'CycleLength', 'YellowTime', 'GreenTime', 'arrival_after_yellow']
        df.drop(drop_cols, axis = 1, inplace = True)
        
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
        
        # rename column
        mdf.rename(columns = {'volume_adv': 'volume_adv_15'}, inplace = True)
        
        # append mdf to result
        result.append(mdf)
        
    fdf = pd.concat(result)
    
    # drop rows with Nan values
    fdf.dropna(axis = 0, inplace = True)
    fdf.reset_index(drop = True, inplace = True)
    fdf.to_csv(output_file, sep = '\t', index = False)

bulkProcessMatchPairs('test')
bulkProcessMatchPairs('train')