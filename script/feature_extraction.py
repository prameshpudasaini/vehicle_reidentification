import os
import pandas as pd

os.chdir(r"D:\GitHub\vehicle_reidentification")

def extractFeatures(data_type):
    # input, output paths by data type
    if data_type == 'test':
        input_file = "data/processed_match_pairs_ground_truth.txt"
        output_file = "data/final_match_pairs_ground_truth.txt"
    else:
        input_file = "data/processed_match_pairs_train.txt"
        output_file = "data/final_match_pairs_train.txt"
        
    print("Extracting features for", data_type.upper(), "dataset")
    
    df = pd.read_csv(input_file, sep = '\t')
    
    # updated rows with SCA = 'RR' at advance location
    df.loc[(df.SCA_adv == 'RR') & (df.arrival_time_adv == 3.6), 'SCA_adv'] = 'YR'
    
    lane_count = df.Lane_adv.value_counts().to_dict()
    print("Vehicle count by lane: ", lane_count)
    
    SCA_count_adv = df.SCA_adv.value_counts().to_dict()
    print("Vehicle count by SCA: ", SCA_count_adv, "\n")
    
    # one-hot enconding: arrival lane
    lane_dummies = pd.get_dummies(df['Lane_adv'], prefix = 'is_lane', prefix_sep = '_', drop_first = True, dtype = int)
    df = pd.concat([df, lane_dummies], axis = 1)
    df.drop('Lane_adv', axis = 1, inplace = True)
    
    # one-hot enconding: SCA over advance det
    SCA_adv_dummies = pd.get_dummies(df['SCA_adv'], prefix = 'SCA_adv', prefix_sep = '_', drop_first = True, dtype = int)
    df = pd.concat([df, SCA_adv_dummies], axis = 1)
    df.drop('SCA_adv', axis = 1, inplace = True)
    
    # car-following
    
    # drop redundant columns
    adv_cols_drop = ['AIY_adv', 'TUY_adv', 'HeadwayLead_adv', 'GapLead_adv']
    stop_cols_drop = [col for col in df.columns if col.endswith('_stop')]
    df.drop(adv_cols_drop + stop_cols_drop, axis = 1, inplace = True)
    
    # remove common suffix '_adv' from column names
    df.columns = [col.replace('_adv', '') for col in df.columns]
    
    # # boolean feature: stop/go decision
    # df['decision'] = (df.SCA_stop.isin(['YG', 'RG'])).astype(int)
    
    # # boolean features: arrival/departure over adv det
    # df['on_green_adv'] = (df.SCA_adv.str.startswith('G')).astype(int)
    # df['off_green_adv'] = (df.SCA_adv.str.endswith('G')).astype(int)
    # df['off_yellow_adv'] = (df.SCA_adv.str.endswith('Y')).astype(int)
    # df.drop('SCA_adv', axis = 1, inplace = True)
    
    # # boolean features: arrival/departure over stop-bar det
    # df['on_green_stop'] = (df.SCA_stop.str.startswith('G')).astype(int)
    # df['on_yellow_stop'] = (df.SCA_stop.str.startswith('Y')).astype(int)
    # df['off_yellow_stop'] = (df.SCA_stop.str.endswith('Y')).astype(int)
    # df['off_red_stop'] = (df.SCA_stop.str.endswith('R')).astype(int)
    # df.drop('SCA_stop', axis = 1, inplace = True)
    
    # # drop leading headway and gap
    # df.drop(['HeadwayLead_adv', 'GapLead_adv', 'HeadwayLead_stop', 'GapLead_stop'], axis = 1, inplace = True)
    
    # # drop AIY and TUY
    # df.drop(['AIY_adv', 'TUY_adv', 'AIY_stop', 'TUY_stop', 'arrival_time_stop'], axis = 1, inplace = True)
    
    # # one-hot enconding: SCA over stop-bar det
    # SCA_stop_dummies = pd.get_dummies(df['SCA_stop'], prefix = 'SCA_stop', prefix_sep = '_', drop_first = True, dtype = int)
    # df = pd.concat([df, SCA_stop_dummies], axis = 1)
    # df.drop('SCA_stop', axis = 1, inplace = True)
    
    # # ratios of occupancy time, headway, gap over adv and stop-bar det
    # df['occ_time_ratio'] = (df.OccTime_adv / df.OccTime_stop).round(6)
    # df['headway_foll_ratio'] = (df.HeadwayFoll_adv / df.HeadwayFoll_stop).round(6)
    # df['headway_lead_ratio'] = (df.HeadwayLead_adv / df.HeadwayLead_stop).round(6)
    # df['gap_foll_ratio'] = (df.GapFoll_adv / df.GapFoll_stop).round(6)
    # df['gap_lead_ratio'] = (df.GapLead_adv / df.GapLead_stop).round(6)
    
    # save file
    df.to_csv(output_file, sep = '\t', index = False)
    
extractFeatures('test')
extractFeatures('train')