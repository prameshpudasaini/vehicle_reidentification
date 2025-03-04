import os
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir("/Users/prameshpudasaini/Documents/vehicle_reidentification")

# compute arrival volume per cycle at advance det
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

# =============================================================================
# process match pairs
# =============================================================================

def bulkProcessMatchPairs():
    # input, file, and output paths by data type
    input_path = "ignore/data_ground_truth_additional/processed"
    file_path = "ignore/data_ground_truth_additional/candidate_match_pairs_ground_truth_additional.csv"
    raw_path = "ignore/data_ground_truth_additional/raw"
    cycle_path = "ignore/data_ground_truth_additional/cycle"
    output_file = "data/processed_match_pairs_ground_truth_additional.txt"

    # match pairs for each file and append
    file_list = os.listdir(input_path)
    result = []
    bulk_timestamp_join = []
    
    for file in file_list:
        print("Processing match pairs for file: ", file)
        
        # read validated match pairs dataset
        vdf = pd.read_csv(file_path)
        vdf = vdf[vdf.file == file[:-4]] # discard ".txt" in file
        
        # select thru match pairs
        vdf = vdf.drop('rear', axis = 1) # drop rear column
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
        
        # merge validated df with adf and sdf
        mdf = pd.merge(vdf, adf, left_on = 'adv', right_on = 'ID_adv')
        mdf = pd.merge(mdf, sdf, left_on = 'stop', right_on = 'ID_stop')
        
        # mdf with timestamp (for analyzing training dataset)
        file_timestamp_join = mdf.copy(deep = True)
            
        # drop redundant columns
        drop_cols = ['travel_time', 'ID_adv', 'volume_stop', 'Lane_stop', 'ID_stop']
        mdf.drop(drop_cols, axis = 1, inplace = True)
        
        # add travel time (includes travel time for IDs with "lane change" in test dataset)
        mdf['travel_time'] = (mdf.TimeStamp_stop - mdf.TimeStamp_adv).dt.total_seconds().round(1)
        mdf.drop(['TimeStamp_adv', 'TimeStamp_stop'], axis = 1, inplace = True)
        
        # rename column
        mdf.rename(columns = {'volume_adv': 'volume_adv_15'}, inplace = True)
        
        # append mdf to result, timestamp df to result with timestamp
        result.append(mdf)
        bulk_timestamp_join.append(file_timestamp_join)
        
    fdf = pd.concat(result)
    full_timestamp_join = pd.concat(bulk_timestamp_join)
    
    # drop rows with Nan values
    fdf.dropna(axis = 0, inplace = True)
    
    # save file
    fdf.reset_index(drop = True, inplace = True)
    fdf.to_csv(output_file, sep = '\t', index = False)
    
    return {'final_df': fdf, 'full_timestamp_join': full_timestamp_join}

result_test = bulkProcessMatchPairs()

# test/train datasets with timestamp joins
df_test = result_test['full_timestamp_join']

# drop redundant columns
drop_cols = ['TimeStamp_adv', 'HeadwayLead_adv', 'GapLead_adv', 'ID_adv', 'TimeStamp_stop', 
             'volume_stop', 'Lane_stop', 'HeadwayLead_stop', 'GapLead_stop', 'ID_stop',
             'arrival_time_stop', 'volume_adv_cycle_stop']

df_test.drop(drop_cols, axis = 1, inplace = True)
