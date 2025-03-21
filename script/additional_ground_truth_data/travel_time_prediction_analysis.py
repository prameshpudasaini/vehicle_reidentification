import os
import pandas as pd

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir("/Users/prameshpudasaini/Library/CloudStorage/OneDrive-UniversityofArizona/GitHub/vehicle_reidentification")

# list of files for which ML/DL models were developed
files = ['20221206_0845_0915', '20221206_0945_1200', '20221214_0645_0715',
         '20221214_0945_1015', '20230327_0700_1400', '20230327_1415_1900']

index_cols = ['file', 'adv', 'stop']

# read ground-truth match pairs data and filter match pairs
gdf = pd.read_csv("data/final_match_pairs_ground_truth_additional.txt", sep = '\t')
gdf = gdf[gdf.file.isin(files)]
gdf = gdf[gdf.match == 1]

def analyzeTravelTimePrediction(model): 
    file_name = model + '.txt'
    
    # read travel time prediction and reidentification result datasets
    tdf = pd.read_csv(os.path.join("ignore/predicted_travel_time", file_name), sep = '\t')
    mdf = pd.read_csv(os.path.join("ignore/reidentification_result", file_name), sep = '\t')
    
    # filter datasets for selected files
    tdf = tdf[tdf.file.isin(files)]
    mdf = mdf[mdf.file.isin(files)]
    
    # add binary variable for model prediction
    mdf['is_predicted'] = 1
    
    # filter required columns
    tdf = tdf[index_cols + ['travel_time', 'y_pred']]
    mdf = mdf[index_cols + ['is_predicted']]
    
    # combine resulting datasets
    cdf = tdf.merge(mdf, how = 'left', on = index_cols)
    
    # merge with ground-truth data
    mdf = gdf[index_cols].merge(cdf, how = 'left', on = index_cols)
    
    # create True Positive and False Negative group
    mdf['Group'] = mdf['is_predicted'].apply(lambda x: 'True Positive' if x == 1 else 'False Negative')
    mdf['Model'] = model
    
    return mdf

models = ['SVR', 'RF', 'XGB', 'FCNN', 'TabNet', 'NODE']
list_df = []

for model in models:
    print(f"Running model: {model}")
    list_df.append(analyzeTravelTimePrediction(model))
    
df = pd.concat(list_df, ignore_index = True)
df.to_csv("ignore/predicted_travel_time_all_models.txt", sep = '\t', index = False)

px.scatter(df, x = 'travel_time', y = 'y_pred', color = 'Group', facet_col = 'Model').show()

