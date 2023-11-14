import os
import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

os.chdir(r"D:\GitHub\vehicle_reidentification")

data = pd.read_csv("data/final_match_pairs_train.txt", sep = '\t') # inferred match pairs

# hyperparameters
param = {
    'max_depth': [5, 10],
    'min_samples_split': [2, 5, 10]
}

# function to produce combination of hyperparameters to test reidentification accuracy
def parameterCombination(model_param):
    keys = model_param.keys()
    values = model_param.values()
    
    value_comb = list(itertools.product(*values)) # all possible combinations of values
    comb_list = [] # store list of dictionaries

    for comb in value_comb:
        comb_dict = dict(zip(keys, comb))
        comb_list.append(comb_dict)
        
    return comb_list

# function to fit and predict travel time
def modelFitPredict(model, X_train, y_train, X_valid, y_valid):
    # fit model with hyperparameters on training folds
    model.fit(X_train, y_train)
    
    # make predictions on validation fold
    y_pred = model.predict(X_valid)
    
    # evaluate metrics
    mape = mean_absolute_percentage_error(y_valid, y_pred)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    
    # select features for prediction on candidate actuations at advance det
    X_adv = data.drop(['file', 'adv', 'stop', 'match', 'travel_time'], axis = 1)
    
    # make predictions on candidate actuations at advance det
    y_adv = model.predict(X_adv)
    
    # add predicted travel time to dataset with full match (0s & 1s)
    data_pred = data.copy(deep = True)
    data_pred['y_pred'] = y_adv
    
    return {'mape': mape, 'rmse': rmse, 'data_pred': data_pred}

tt_thru_min, tt_thru_max = 2.5, 12 # empirical min, max of through travel time
index_cols = ['file', 'adv', 'stop']
file_path = "ignore/data_train/processed_subset"

# function for reidentification algorithm
def reidentifyMatchPairs(df_pred, match_truth):
    print("Running reidentification algorithm")
    
    files = list(match_truth.file.unique()) # list of files to iterate through
    df_result = [] # store reidentified match pairs from each file
    
    for file in files:
        # print(f"Processing file: {file}")
        
        # read events-processed file with timestamp data
        df = pd.read_csv(os.path.join(file_path, file + '.txt'), sep = '\t')
        df.TimeStamp = pd.to_datetime(df.TimeStamp, format = '%Y-%m-%d %H:%M:%S.%f').sort_values()
        df.dropna(axis = 0, inplace = True) # drop rows with Nan
        
        # data frames for adv and stop-bar det
        adf = df[df.Det == 'adv']
        sdf = df[df.Det == 'stop']
        id_adv = list(sorted(adf.ID))
        
        thru_match_initial = [] # store initial candidate match pairs of adv to stop-bar det
        
        for i in id_adv:
            adv_time = adf[adf.ID == i].TimeStamp.values[0]
            adv_lane = adf[adf.ID == i].Lane.values[0]
            
            # stop-bar det IDs on the same lane to look for a match
            id_stop_look = set(sdf[sdf.Lane == adv_lane].ID)
            
            for j in id_stop_look:
                stop_time = sdf[sdf.ID == j].TimeStamp.values[0]
                
                if stop_time > adv_time: # look forward in timestamp
                    tt_adv_stop = (stop_time - adv_time) / np.timedelta64(1, 's') # paired travel time
                
                    if tt_thru_min <= tt_adv_stop <= tt_thru_max:
                        # get predicted travel time for file and id_adv
                        X_i = df_pred.copy(deep = True)
                        X_i = X_i[(X_i.file == file) & (X_i.adv == i)].reset_index(drop = True)
                        
                        if X_i.shape[0] == 0: # the candidate adv ID was filtered out
                            pass
                        else:
                            tt_predict = X_i.loc[0, 'y_pred']
                            tt_diff = round(abs(tt_adv_stop - tt_predict), 4) # abs diff between paired & predicted
                        
                            # store adv ID, stop ID, travel time diff
                            thru_match_initial.append([i, j, tt_diff])
        
        # dicts to store the lowest error for each adv, stop ID
        seen_adv_id, seen_stop_id = {}, {}
        
        # iterate through each candidate pair
        for pair in thru_match_initial:
            adv_id, stop_id, error = pair
            
            # check if adv ID not seen or if error is lower than seen error for that adv ID
            if (adv_id not in seen_adv_id) or (error < seen_adv_id[adv_id][1]):
                seen_adv_id[adv_id] = list([stop_id, error])
            
            # check if stop ID not seen or if error is lower than seen error for that stop ID
            if (stop_id not in seen_stop_id) or (error < seen_stop_id[stop_id][1]):
                seen_stop_id[stop_id] = list([adv_id, error])
        
        # match pairs for adv with lowest error
        df_adv = pd.DataFrame(seen_adv_id, index = ['adv', 'stop']).T.reset_index()
        df_adv.columns = ['adv', 'stop', 'error']
        
        # match pairs for stop with lowest error
        df_stop = pd.DataFrame(seen_stop_id, index = ['stop', 'adv']).T.reset_index()
        df_stop.columns = ['stop', 'adv', 'error']
        
        # store resulting common match pairs
        if (df_adv.shape[0] == 0) & (df_stop.shape[0] == 0): # no match pairs in the file
            pass
        else:
            df_match_pair = df_adv.merge(df_stop, on = ['adv', 'stop', 'error'])
            df_match_pair['file'] = file
            df_result.append(df_match_pair)
    
    print("Processing files complete\n")
    match_result_all_adv = pd.concat(df_result) # combines result for all adv IDs
    
    # filter match result for file & adv combination in match truth
    match_result = match_result_all_adv.merge(match_truth[['file', 'adv']], on = ['file', 'adv'], how = 'inner')
    
    # get true positive (TP), false positive (FP), and false negative (FN) matches
    match_TP = pd.merge(match_result, match_truth, on = index_cols)
    match_FP = match_result.merge(match_truth, on = index_cols, how = 'left', indicator = True).query('_merge == "left_only"').drop(columns = '_merge')
    match_FN = match_truth.merge(match_result, on = index_cols, how = 'left', indicator = True).query('_merge == "left_only"').drop(columns = '_merge')
    
    # number of TP, FP, FN matches
    TP, FP, FN = match_TP.shape[0], match_FP.shape[0], match_FN.shape[0]
    
    num_ground_match_pairs = match_truth.shape[0]

    # compute accuracy, precision, recall, f1 metrics
    accuracy = TP / num_ground_match_pairs
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# define num of folds and initialize KFold cross-validator
random_state = 42
num_folds = 10
kf = KFold(n_splits = num_folds, shuffle = True, random_state = random_state)

# combination of hyperparameters
param_comb = parameterCombination(param)
comb = param_comb[3] # test a combination
model = DecisionTreeRegressor(**comb, random_state = random_state) # model specification for hyperparameters
    
# variables to store metrics from each fold
fold_pred_mape, fold_pred_rmse = [], []
fold_match_accuracy, fold_match_precision, fold_match_recall, fold_match_f1 = [], [], [], []

# perform 10 fold CV
fold = 1
    
for train_idx, valid_idx in kf.split(data):
    print(f"Running cross validation for fold {fold}")
    
    # training and validation folds
    data_train, data_valid = data.iloc[train_idx], data.iloc[valid_idx]
    # print(f"Train & validation data size: {len(data_train)}, {len(data_valid)}")
    
    # number of 1 and 0 matches in training & validation folds
    count_match_train = data_train['match'].value_counts()
    count_match_valid = data_valid['match'].value_counts()
    # print(f"Count of matches in train: 1s {count_match_train.get(1, 0)}, 0s {count_match_train.get(0, 0)}")
    # print(f"Count of matches in valid: 1s {count_match_valid.get(1, 0)}, 0s {count_match_valid.get(0, 0)}")
        
    # fiter rows with match = 1
    df_train = data_train.copy(deep = True)[data_train.match == 1]
    df_valid = data_valid.copy(deep = True)[data_valid.match == 1]
    
    # drop columns redundant for travel time prediction
    df_train = df_train.drop(index_cols + ['match'], axis = 1).reset_index(drop = True)
    df_valid = df_valid.drop(index_cols + ['match'], axis = 1).reset_index(drop = True)
    
    # training and validation features
    X_train = df_train.drop('travel_time', axis = 1)
    y_train = df_train.travel_time
    X_valid = df_valid.drop('travel_time', axis = 1)
    y_valid = df_valid.travel_time
    
    # predict travel time and evaluate metrics    
    model_fit_result = modelFitPredict(model, X_train, y_train, X_valid, y_valid)
    fold_pred_mape.append(model_fit_result['mape'])
    fold_pred_rmse.append(model_fit_result['rmse'])
    
    # input parameters for reidentification algorithm
    data_pred = model_fit_result['data_pred']
    match_valid = data_valid.copy(deep = True)[data_valid.match == 1].drop('match', axis = 1)[index_cols]
    
    # results of running reidentification algorithm
    match_result = reidentifyMatchPairs(data_pred, match_valid)
    fold_match_accuracy.append(match_result['accuracy'])
    fold_match_precision.append(match_result['precision'])
    fold_match_recall.append(match_result['recall'])
    fold_match_f1.append(match_result['f1'])
    
    fold += 1