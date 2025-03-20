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

# ground-truth match pairs
gdf = pd.read_csv("data/final_match_pairs_ground_truth_additional.txt", sep = '\t')
gdf = gdf[gdf.file.isin(files)]
num_candidate_pairs = gdf.shape[0]
print(f"\nNum of candidate pairs: {num_candidate_pairs}\n")

# filter ground-truth match pairs for match == 1 and select index cols
gdf = gdf[gdf.match == 1][index_cols]
print(f"Num of ground-truth match pairs: {gdf.shape[0]}")

# model results
def analyzeModelResult(model):
    file = model + '.txt'
    mdf = pd.read_csv(os.path.join("ignore/reidentification_result", file), sep = '\t')
    
    # get true positive (TP), false positive (FP), and false negative (FN) matches   
    match_TP = pd.merge(mdf, gdf, on = index_cols)
    match_FP = mdf.merge(gdf, on = index_cols, how = 'left', indicator = True).query('_merge == "left_only"').drop(columns = '_merge')
    match_FN = gdf.merge(mdf, on = index_cols, how = 'left', indicator = True).query('_merge == "left_only"').drop(columns = '_merge')
    
    # num of TP, FP, FN
    TP, FP, FN = match_TP.shape[0], match_FP.shape[0], match_FN.shape[0]
    
    # compute metrics
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    
    print(f"\nModel: {model}")
    print(f"TP, FP, FN: {TP}, {FP}, {FN}")
    print(f"Precision, Recall, F1: {precision:.4f}, {recall:.4f}, {f1:.4f}")
    
models = ['SVR', 'RF', 'XGB', 'FCNN', 'TabNet', 'NODE'] # list of models
for model in models:
    analyzeModelResult(model)