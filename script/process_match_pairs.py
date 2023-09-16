import os
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\vehicle_reidentification")

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

def bulkProcessMatchPairs(data_type):
    # input, file, and output paths by data type
    if data_type == 'test':
        input_path = "ignore/data_ground_truth/processed"
        file_path = "data/candidate_match_pairs_ground_truth.csv"
        raw_path = "ignore/data_ground_truth/raw"
        cycle_path = "ignore/data_ground_truth/cycle"
        output_file = "data/processed_match_pairs_ground_truth.txt"
    else:
        input_path = "ignore/data_train/processed_subset" # sub-folder
        file_path = "data/candidate_match_pairs_train.csv"
        raw_path = "ignore/data_train/raw"
        cycle_path = "ignore/data_train/cycle"
        output_file = "data/processed_match_pairs_train.txt"

    # match pairs for each file and append
    file_list = os.listdir(input_path)
    result = []
    bulk_timestamp_join = []
    
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
        
        # mdf with timestamp (for analyzing training dataset)
        file_timestamp_join = mdf.copy(deep = True)
            
        # drop redundant columns
        drop_cols = ['file', 'adv', 'stop', 'travel_time', 'remark', 'ID_adv', 'volume_stop', 'Lane_stop', 'ID_stop']
        mdf.drop(drop_cols, axis = 1, inplace = True)
        
        # add travel time (includes travel time for IDs with "lane change")
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
    
    if data_type == 'train':
        # remove matches with adv occupancy time > threshold
        occ_adv_limit = 4
        fdf = fdf[fdf.OccTime_adv <= occ_adv_limit]
        
        # remove matches with 'go' decisions and stop-bar occ time > 4
        occ_stop_limit = 4
        sca_stopbar_go = ['GY', 'YY', 'YR', 'RR']
        fdf = fdf[~((fdf.OccTime_stop >= occ_stop_limit) & (fdf.SCA_stop.isin(sca_stopbar_go)))]
    
    # save file
    fdf.reset_index(drop = True, inplace = True)
    fdf.to_csv(output_file, sep = '\t', index = False)
    
    return {'final_df': fdf, 'full_timestamp_join': full_timestamp_join}

result_test = bulkProcessMatchPairs('test')
result_train = bulkProcessMatchPairs('train')

# test/train datasets with timestamp joins
df_test = result_test['full_timestamp_join']
df_train = result_train['full_timestamp_join']

(df_test.columns == df_train.columns).all()

# drop redundant columns
drop_cols = ['remark', 'TimeStamp_adv', 'HeadwayLead_adv', 'GapLead_adv', 'ID_adv',
             'TimeStamp_stop', 'volume_stop', 'Lane_stop', 'HeadwayLead_stop', 'GapLead_stop', 'ID_stop',
             'arrival_time_stop', 'volume_adv_cycle_stop']

df_test.drop(drop_cols, axis = 1, inplace = True)
df_train.drop(drop_cols, axis = 1, inplace = True)

# =============================================================================
# analyze training dataset
# =============================================================================

# # check occupany time over advance detector
# df_occ_adv = df_train.copy(deep = True)[df_train.OccTime_adv >= 1.5]
# px.scatter(df_occ_adv, x = 'OccTime_adv', y = 'OccTime_stop').show()

# # remove matches with adv occupancy time > threshold
# occ_adv_limit = 4
# df_train = df_train[df_train.OccTime_adv <= occ_adv_limit]

# # check occupancy time over stop-bar detector on 'go' decisions
# sca_stopbar_go = ['GY', 'YY', 'YR', 'RR']
# df_occ_stop = df_train.copy(deep = True)[(df_train.OccTime_stop >= 2) & (df_train.SCA_stop.isin(sca_stopbar_go))]
# px.scatter(df_occ_stop, x = 'OccTime_stop', y = 'travel_time').show()

# # remove matches with 'go' decisions and stop-bar occ time > 4
# occ_stop_limit = 4
# df_train = df_train[~((df_train.OccTime_stop >= occ_stop_limit) & (df_train.SCA_stop.isin(sca_stop_go)))]

# # check relationship between headway/gap at adv and stop-bar
# df_train.HeadwayFoll_adv.corr(df_train.HeadwayFoll_stop)
# df_train.GapFoll_adv.corr(df_train.GapFoll_stop)

# px.scatter(df_train, x = 'HeadwayFoll_adv', y = 'HeadwayFoll_stop').show()
# px.scatter(df_train, x = 'GapFoll_adv', y = 'GapFoll_stop').show()

# # signal change during actuation over advance and stop-bar det
# sca_adv = ['GG', 'GY', 'YY', 'YR']
# sca_stop = ['GY', 'YY', 'YR', 'YG', 'RR', 'RG']

# # possible sca combinations
# sca_comb = {'GG': sca_stop,
#             'GY': sca_stop[1:],
#             'YY': sca_stop[1:],
#             'YR': sca_stop[-2:]}

# def plotAdvStop(xdf):
#     px.scatter(xdf, x = 'OccTime_adv', y = 'travel_time').show()
#     px.scatter(xdf, x = 'HeadwayFoll_adv', y = 'HeadwayFoll_stop').show()
#     px.scatter(xdf, x = 'OccTime_adv', y = 'OccTime_stop').show()
    
# def dataSCA(sca_adv, sca_stop):
#     xdf = df_train.copy(deep = True)[(df_train.SCA_adv == sca_adv) & (df_train.SCA_stop == sca_stop)]
#     return xdf

# # SCA: GG & GY
# df_GG_GY = dataSCA('GG', 'GY')
# plotAdvStop(df_GG_GY)

# # SCA: GG & YY
# df_GG_YY = dataSCA('GG', 'YY')
# plotAdvStop(df_GG_YY)

# # SCA: GG & YR
# df_GG_YR = dataSCA('GG', 'YR')
# plotAdvStop(df_GG_YR)

# # SCA: GG & YG
# df_GG_YG = dataSCA('GG', 'YG')
# plotAdvStop(df_GG_YG)

# # SCA: GG & RR
# df_GG_RR = dataSCA('GG', 'RR')
# plotAdvStop(df_GG_RR)

# # SCA: GG & RG
# df_GG_RG = dataSCA('GG', 'RG')
# plotAdvStop(df_GG_RG)

# # SCA: GY & YY
# df_GY_YY = dataSCA('GY', 'YY')
# plotAdvStop(df_GY_YY)

# # SCA: GY & YR
# df_GY_YR = dataSCA('GY', 'YR')
# plotAdvStop(df_GY_YR)

# # SCA: GY & YG
# df_GY_YG = dataSCA('GY', 'YG')
# plotAdvStop(df_GY_YG)

# # SCA: GY & RR
# df_GY_RR = dataSCA('GY', 'RR')
# plotAdvStop(df_GY_RR)

# # SCA: GY & RG
# df_GY_RG = dataSCA('GY', 'RG')
# plotAdvStop(df_GY_RG)

# # SCA: YY & YY
# df_YY_YY = dataSCA('YY', 'YY') # empty

# # SCA: YY & YR
# df_YY_YR = dataSCA('YY', 'YR')
# plotAdvStop(df_YY_YR)

# # SCA: YY & YG
# df_YY_YG = dataSCA('YY', 'YG') # empty

# # SCA: YY & RR
# df_YY_RR = dataSCA('YY', 'RR')
# plotAdvStop(df_YY_RR)

# # SCA: YR & RR
# df_YR_RR = dataSCA('YR', 'RR') # empty

# # SCA: YR & RG
# df_YR_RG = dataSCA('YR', 'RG')
# plotAdvStop(df_YR_RG)
