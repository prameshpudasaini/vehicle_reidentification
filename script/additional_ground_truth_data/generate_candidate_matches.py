import os
import pandas as pd
import numpy as np

os.chdir("/Users/prameshpudasaini/Documents/vehicle_reidentification")

# for additional ground-truth data
input_path = "ignore/data_ground_truth_additional/processed"
output_path = "ignore/data_ground_truth_additional"

# detector length & spacing parameters
len_stop = 40 # length of stop-bar det
dist_det = 300 # end-end distance between stop-bar and advance det
dist_adv_stop = dist_det - len_stop

# min, max travel time for constraining search space
tt_min, tt_max = 2.5, 13

# =============================================================================
# generate candidate match pairs
# =============================================================================

def matchAcutuationEvents():
    
    # data frames for adv, stop, left-turn
    adf = df.copy(deep = True)[df.Det == 'adv']
    sdf = df.copy(deep = True)[df.Det == 'stop']
    rdf = df.copy(deep = True)[df.Det == 'rear']
    
    # actuation IDs
    id_adv = set(sorted(adf.ID))
    id_stop = set(sorted(sdf.ID))
    id_rear = set(sorted(rdf.ID))
    
    # store candidate matches
    adv_stop_match = []
    adv_rear_match = []
    
    for i in id_adv:
        adv_time = adf[adf.ID == i].TimeStamp.values[0]
        adv_lane = adf[adf.ID == i].Lane.values[0]
        
        # candidate matches from adv to stop-bar det
        for j in id_stop:
            stop_time = sdf[sdf.ID == j].TimeStamp.values[0]
            stop_lane = sdf[sdf.ID == j].Lane.values[0]
            
            if stop_time > adv_time and stop_lane == adv_lane:
                tt_adv_stop = (stop_time - adv_time) / np.timedelta64(1, 's')
                
                if tt_adv_stop < tt_min or tt_adv_stop > tt_max:
                    pass
                else:
                    adv_stop_match.append([i, j, adv_time, stop_time, tt_adv_stop])
        
        # candidate matches from adv to rear det
        if adv_lane == 'L':
            for k in id_rear:
                rear_time = rdf[rdf.ID == k].TimeStamp.values[0]
                
                if rear_time > adv_time:
                    tt_adv_rear = (rear_time - adv_time) / np.timedelta64(1, 's')
                    
                    if tt_adv_rear < tt_min or tt_adv_rear > tt_max:
                        pass
                    else:
                        adv_rear_match.append([i, k, adv_time, rear_time, tt_adv_rear])
    
    # df of candidate matches
    df_adv_stop = pd.DataFrame(adv_stop_match, columns = ['adv', 'stop', 'TimeStamp_adv', 'TimeStamp_stop', 'travel_time'])
    df_adv_rear = pd.DataFrame(adv_rear_match, columns = ['adv', 'rear', 'TimeStamp_adv', 'TimeStamp_rear', 'travel_time'])
    
    # merged data frame
    mdf = pd.concat([df_adv_stop, df_adv_rear], ignore_index = True).sort_values(by = 'adv')
    mdf.reset_index(drop = True, inplace = True)
      
    return mdf

# =============================================================================
# match events in bulk
# =============================================================================

# list of raw files
file_list = os.listdir(input_path)
result = []

# match events for each file
for file in file_list:
    print("Generating candidate match pairs for file: ", file)
    
    # read data
    df = pd.read_csv(os.path.join(input_path, file), sep = '\t')
    df.TimeStamp = pd.to_datetime(df.TimeStamp, format = '%Y-%m-%d %H:%M:%S.%f').sort_values()    
    mdf = matchAcutuationEvents()
    
    # add file name
    mdf['file'] = file[:-4]
    mdf = mdf[['file', 'adv', 'stop', 'rear', 'travel_time']]

    result.append(mdf)

fdf = pd.concat(result, ignore_index = True)
fdf.to_csv(os.path.join(output_path, 'candidate_match_pairs.csv'), index = False)