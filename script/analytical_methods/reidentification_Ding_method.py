import os
import numpy as np
import pandas as pd

os.chdir(r"D:\GitHub\vehicle_reidentification")

# detector length & spacing parameters
len_stop = 40 # length of stop-bar det
len_adv = 5 # length of advance det
dist_det = 300 # end-end distance between stop-bar and advance det
# dist_adv_stop = dist_det - len_stop

# other parameters
veh_length = list(range(16, 24))
acc_max = 6

# =============================================================================
# Ding's method to reidentify vehicles
# =============================================================================

def reidentifyMatchPairs(adf, sdf, id_adv, file):
    thru_match_initial = [] # store initial candidate match pairs of adv to stop-bar det
    
    for i in id_adv:
        adv_time = adf[adf.ID == i].TimeStamp.values[0]
        adv_lane = adf[adf.ID == i].Lane.values[0]
        adv_occ = adf[adf.ID == i].OccTime.values[0]
        
        adv_vel = eff_length_adv / adv_occ
        tt_max = 2*dist_det / adv_vel
        tt_min = 2*dist_det / (adv_vel + ((adv_vel)**2 + 2*acc_max*dist_det)**0.5)

        # stop-bar det IDs on the same lane to look for a match
        id_stop_look = set(sdf[sdf.Lane == adv_lane].ID)

        for j in id_stop_look:
            stop_time = sdf[sdf.ID == j].TimeStamp.values[0]

            if stop_time > adv_time: # look forward in timestamp
                tt_adv_stop = (stop_time - adv_time) / np.timedelta64(1, 's') # paired travel time

                if tt_min <= tt_adv_stop <= tt_max:
                    stop_occ = sdf[sdf.ID == j].OccTime.values[0]
                    stop_vel = eff_length_stop / stop_occ
                    
                    tt_ideal = 2*dist_det / (adv_vel + stop_vel) # ideal travel time
                    tt_error = abs(1 - tt_ideal/tt_adv_stop)
                    
                    # store adv ID, stop ID, travel time error
                    thru_match_initial.append([i, j, tt_error])

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
    
    return {'df_adv': df_adv, 'df_stop': df_stop}

# =============================================================================
# process each file for reidentifying match pairs
# =============================================================================

file_path = "ignore/data_ground_truth/processed"
files = os.listdir(file_path)  # list of processed files to run through reidentifying algorithm

def processFiles():
    temp_result = [] # store reidentified match pairs from each file
    
    for file in files:
        # read events-processed file with timestamp data
        df = pd.read_csv(os.path.join(file_path, file), sep = '\t')
        df.TimeStamp = pd.to_datetime(df.TimeStamp, format = '%Y-%m-%d %H:%M:%S.%f').sort_values()
        df.dropna(axis = 0, inplace = True) # drop rows with Nan
        
        # data frames for adv and stop-bar det
        adf = df[df.Det == 'adv']
        sdf = df[df.Det == 'stop']
        id_adv = list(sorted(adf.ID))
        
        # process candidate match pairs to get datasets of adv and stop pairs
        candidate_match_result = reidentifyMatchPairs(adf, sdf, id_adv, file)
        df_adv = candidate_match_result['df_adv']
        df_stop = candidate_match_result['df_stop']
        
        # resulting common match pairs
        df_match_pair = df_adv.merge(df_stop, on = ['adv', 'stop', 'error'])
        df_match_pair['file'] = file[:-4]
        temp_result.append(df_match_pair)
        
    match_result = pd.concat(temp_result)
    return match_result

# =============================================================================
# compute reidentification metrics
# =============================================================================

index_cols = ['file', 'adv', 'stop']

# read ground-truth match pairs
file_match_pairs = "data/final_match_pairs_ground_truth.txt"
match_ground = pd.read_csv(file_match_pairs, sep = '\t')

# filter ground-truth match pairs for match == 1 and select index cols
match_ground = match_ground[match_ground.match == 1][index_cols]
num_ground_match_pairs = match_ground.shape[0]

def computeReidMetrics():
    # get match result from processing files for reidentifying algorithm
    match_result = processFiles()
    
    # get true positive (TP), false positive (FP), and false negative (FN) matches   
    match_TP = pd.merge(match_result, match_ground, on = index_cols)
    match_FP = match_result.merge(match_ground, on = index_cols, how = 'left', indicator = True).query('_merge == "left_only"').drop(columns = '_merge')
    match_FN = match_ground.merge(match_result, on = index_cols, how = 'left', indicator = True).query('_merge == "left_only"').drop(columns = '_merge')
    
    # num of TP, FP, FN
    TP, FP, FN = match_TP.shape[0], match_FP.shape[0], match_FN.shape[0]
    
    # compute metrics
    accuracy = round(TP / num_ground_match_pairs, 4)
    precision = round(TP / (TP + FP), 4)
    recall = round(TP / (TP + FN), 4)
    f1 = round(2*precision*recall / (precision + recall), 4)
    
    reid_metrics = {'TP': TP,
                    'FP': FP,
                    'FN': FN,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1}
    
    return reid_metrics

# =============================================================================
# sensitivity analysis of effective vehicle length
# =============================================================================

temp_metrics = []

for i in veh_length:
    print("Effective vehicle length: ", i)
    
    eff_length_adv = len_adv + i
    eff_length_stop = len_stop + i
    
    reid_metrics = computeReidMetrics()
    df_reid_metrics = pd.DataFrame([reid_metrics])
    df_reid_metrics['veh_length'] = i
    temp_metrics.append(df_reid_metrics)
    
metrics = pd.concat(temp_metrics)
metrics.to_csv("output/analytical_method_results/metrics_Ding_method.txt", sep = '\t', index = False)
        