import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

os.chdir(r"D:\GitHub\vehicle_reidentification")

file_train = "data/final_match_pairs_train.txt"
file_test = "data/final_match_pairs_ground_truth.txt"
df_train = pd.read_csv(file_train, sep = '\t')
df_test = pd.read_csv(file_test, sep = '\t')

# copy df test for reidentifying algorithm
cdf = df_test.copy(deep = True)

# filter rows with match = 1
df_train = df_train[df_train.match == 1]
df_test = df_test[df_test.match == 1]

# drop redundant columns
drop_cols = ['file', 'adv', 'stop', 'match']
df_train.drop(drop_cols, axis = 1, inplace = True)
df_test.drop(drop_cols, axis = 1, inplace = True)

X = df_train.drop('travel_time', axis = 1)
y = df_train.travel_time
rand_state = 42

# split training dataset into train, validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = rand_state)

# test dataset from ground-truth
X_test = df_test.drop('travel_time', axis = 1)
y_test = df_test.travel_time

# =============================================================================
# function to perform hyperparameter tuning and cross-validation
# =============================================================================

cv = 5

def performGridSearch(model_type, param_grid):
    # create grid search object
    grid_search = GridSearchCV(
        estimator = model_type,
        param_grid = param_grid,
        cv = cv,
        scoring = 'neg_mean_squared_error'
    )
    
    # fit grid search object to training dataset
    grid_search.fit(X, y)
    
    # get best hyperparameters and best MSE for current CV value
    best_params = grid_search.best_params_
    best_mse = -grid_search.best_score_
            
    return {'best_mse': best_mse,
            'best_params': best_params}

# =============================================================================
# function to evaluate performance
# =============================================================================

def evaluatePerformance(param_model):
    # fit model with hypertuned parameters on train dataset
    param_model.fit(X_train, y_train)
    
    # make predictions on validation and test sets
    y_pred_valid = param_model.predict(X_valid)
    y_pred_test = param_model.predict(X_test)

    # evaluation metrics on validation set
    mape_valid = mean_absolute_percentage_error(y_valid, y_pred_valid)
    rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))

    # evaluation metrics on test set
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("Model with hypertuned parameters and trained on training dataset only")
    print("MAPE on validation set: {:.4f}".format(mape_valid))
    print("MAPE on test set: {:.4f}".format(mape_test))
    print("RMSE on validation set: {:.4f}".format(rmse_valid))
    print("RMSE on test set: {:.4f}".format(rmse_test), "\n")
    
    # fit model with hypertuned parameters on train + validation datasets
    param_model.fit(X, y)
    
    # make predictions on test set
    y_pred_test_full = param_model.predict(X_test)

    # evaluation metrics on test set
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test_full)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test_full))
    
    print("Model with hypertuned parameters and trained on training + validation datasets")
    print("MAPE on test set: {:.4f}".format(mape_test))
    print("RMSE on test set: {:.4f}".format(rmse_test), "\n")

# =============================================================================
# Ridge Regression
# =============================================================================

method = 'Ridge Regression'
print(method)

# create ridge regression model
ridge_reg = Ridge()

# hyperparameters for grid search
ridge_param_grid = {'alpha': [1e-15, 1e-10, 1e-5, 0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 500]}

# results from grid search
ridge_grid_result = performGridSearch(ridge_reg, ridge_param_grid)

# specify model with tuned parameters
param_ridge_reg = Ridge(**ridge_grid_result['best_params'])

evaluatePerformance(param_ridge_reg)

# =============================================================================
# reidentifying algorithm
# =============================================================================

# reidentified vehicle matches from ground-truth data
match_ground = cdf.copy(deep = True)[cdf.match == 1]
match_ground = match_ground[['file', 'adv', 'stop']]

# empirical min, max of through travel time
tt_thru_min = 2.5
tt_thru_max = 12

# function to predict travel time for individual vehicle actuation over adv det
def predictTravelTime(file, id_adv):
    # filter feature vector X for adv ID
    X_i = cdf.copy(deep = True)[(cdf.file == file[:-4]) & (cdf.adv == id_adv)] # discard .txt
    
    # drop columns redundant for travel time prediction
    drop_cols = ['file', 'adv', 'stop', 'match', 'travel_time']
    X_i.drop(drop_cols, axis = 1, inplace = True)
    
    tt_predict = param_ridge_reg.predict(X_i)[0] # predicted travel time
    return tt_predict

# processed files to run through reidentifying algorithm
file_path = "ignore/data_ground_truth/processed"
files = os.listdir(file_path)

# store reidentified match pairs from each file
df_result = []

for file in files:
    print("Running reidentification algorithm for file: ", file)
    # read events-processed file with timestamp data
    df = pd.read_csv(os.path.join(file_path, file), sep = '\t')
    df.TimeStamp = pd.to_datetime(df.TimeStamp, format = '%Y-%m-%d %H:%M:%S.%f').sort_values()
    df.dropna(axis = 0, inplace = True) # drop rows with Nan
    
    # data frames for adv and stop-bar det
    adf = df[df.Det == 'adv']
    sdf = df[df.Det == 'stop']
    
    # actuation IDs
    id_adv = list(sorted(adf.ID))
    id_stop = list(sorted(sdf.ID))
    
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
                    tt_predict = predictTravelTime(file, i) # predicted travel time
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
    
    # resulting common match pairs
    df_match_pair = df_adv.merge(df_stop, on = ['adv', 'stop', 'error'])
    df_match_pair['file'] = file[:-4]
    df_result.append(df_match_pair)
    
match_result = pd.concat(df_result)

# get true positive (TP), false positive (FP), and false negative (FN) matches
match_cols = ['file', 'adv', 'stop']
match_TP = pd.merge(match_result, match_ground, on = match_cols)
match_FP = match_result.merge(match_ground, on = match_cols, how = 'left', indicator = True).query('_merge == "left_only"').drop(columns = '_merge')
match_FN = match_ground.merge(match_result, on = match_cols, how = 'left', indicator = True).query('_merge == "left_only"').drop(columns = '_merge')
