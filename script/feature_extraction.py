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
    
    # car following as a boolean feature
    gap_foll_limit = 1
    df['car_follow'] = (df.GapFoll_adv <= gap_foll_limit).astype(int)
    
    # drop redundant columns
    adv_cols_drop = ['AIY_adv', 'TUY_adv', 'GapLead_adv']
    stop_cols_drop = [col for col in df.columns if col.endswith('_stop')]
    df.drop(adv_cols_drop + stop_cols_drop, axis = 1, inplace = True)
    
    # remove common suffix '_adv' from column names
    df.columns = [col.replace('_adv', '') for col in df.columns]
    
    # save file
    df.to_csv(output_file, sep = '\t', index = False)
    
extractFeatures('test')
extractFeatures('train')