import os
import pandas as pd
import json

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

os.chdir(r"D:\GitHub\vehicle_reidentification")
result_path = "output/model_results"
plot_path = "output/interactive_plots"

models = ['dt', 'sv', 'rf', 'xgb']
index_cols = ['file', 'adv', 'stop']

# function to return model name
def getModelName(model):
    if model == 'dt':
        method = 'Decision Tree Regression'
    elif model == 'sv':
        method = 'Support Vector Regression'
    elif model == 'rf':
        method = 'Random Forest'
    elif model == 'xgb':
        method = 'XGBoost'
    return method

# =============================================================================
# error in travel time prediction from best match and best pred models
# =============================================================================

match_ground = pd.read_csv("data/final_match_pairs_ground_truth.txt", sep = '\t')
match_ground = match_ground[index_cols + ['match']]

def processPredictionError(model):    
    # read prediction error for best match and best pred models
    file_match_error = model + '_best_match_data_pred.txt'
    file_pred_error = model + '_best_pred_data_pred.txt'
    df_match = pd.read_csv(os.path.join(result_path, file_match_error), sep = '\t')
    df_pred = pd.read_csv(os.path.join(result_path, file_pred_error), sep = '\t')
    
    # merge match and pred dfs with ground-truth data
    df_match = df_match.merge(match_ground, on = index_cols, how = 'left')
    df_pred = df_pred.merge(match_ground, on = index_cols, how = 'left')
    
    # filter match = 1
    df_match = df_match[df_match.match == 1]
    df_pred = df_pred[df_pred.match == 1]
    
    # rename error columns
    df_match.rename(columns = {'error': 'best_match_error'}, inplace = True)
    df_pred.rename(columns = {'error': 'best_pred_error'}, inplace = True)
    df_match = df_match[index_cols + ['best_match_error']]
    df_pred = df_pred[index_cols + ['best_pred_error']]
    
    # merge prediction error for best match and best pred models
    df_error = df_match.merge(df_pred, on = index_cols, how = 'inner')
    df_error.drop(index_cols, axis = 1, inplace = True)

    # add model name
    df_error['model'] = getModelName(model)
    return df_error

list_error = [] # store processed prediction error for each model
for model in models:
    df_error = processPredictionError(model)
    list_error.append(df_error)
        
error = pd.concat(list_error) # combined best match and best pred errors from all models
error_long = pd.melt(error, id_vars = 'model', var_name = 'type', value_name = 'error') # long format
error_long['type'] = error_long['type'].replace({'best_match_error': 'Best reidentification model',
                                                 'best_pred_error': 'Best prediction model'})

fig_error_scatter = px.scatter(
    error, 
    x = 'best_pred_error', 
    y = 'best_match_error', 
    color = 'model',
    labels = {'best_pred_error': 'Error obtained from best prediction model (sec)',
              'best_match_error': 'Error obtained from best reidentification model (sec)'},
    title = 'Comparison of error in travel time prediction across all ground-truth match pairs'
)
fig_error_scatter.update_layout(
    title_x = 0.5,
    legend_title_text = '',
    legend = dict(orientation = 'h', x = 0.5, y = 1)
).show()

fig_error_violin = px.violin(
    error_long, 
    x = 'model', 
    y = 'error', 
    color = 'type',
    box = True,
    labels = {'model': '',
              'error': 'Error in travel time prediction (sec)'},
    title = 'Comparison of error in travel time prediction across models'
)
fig_error_violin.update_layout(
    title_x = 0.5,
    legend_title_text = '',
    legend = dict(orientation = 'h', x = 0.5, y = 1)
).show()

fig_error_scatter.write_html(os.path.join(plot_path, "fig_error_scatter.html"))
fig_error_violin.write_html(os.path.join(plot_path, "fig_error_violin.html"))

# =============================================================================
# match & pred metrics across all hyperparameter combinations
# =============================================================================

def processHyperparameterMetrics(model):
    file_all_metrics = model + '_all_comb_metrics.json'
    
    comb_metrics = [] # store each list from json file
    with open(os.path.join(result_path, file_all_metrics), 'r') as json_file:
        for line in json_file:
            comb_metrics.append(json.loads(line.strip()))
    
    metrics = {'match_f1': [], 'pred_mape': [], 'match_accuracy': [], 'pred_rmse': []}
    for item in comb_metrics:
        metrics['match_f1'].append(item[1]['f1'])
        metrics['match_accuracy'].append(item[1]['accuracy'])
        metrics['pred_mape'].append(item[2]['mape_test_full'])
        metrics['pred_rmse'].append(item[2]['rmse_test_full'])
        
    df_metrics = pd.DataFrame(metrics)
    df_metrics['model'] = getModelName(model)
    return df_metrics

list_metrics = [] # store metrics from all hyperparameter combinations for all models
for model in models:
    df_metrics = processHyperparameterMetrics(model)
    list_metrics.append(df_metrics)
    
all_metrics = pd.concat(list_metrics)

fig_f1_mape = px.scatter(
    all_metrics, 
    x = 'match_f1', 
    y = 'pred_mape', 
    color = 'model',
    labels = {'match_f1': 'F1 score for best reidentification model',
              'pred_mape': 'MAPE for best prediction model (%)'},
    title = 'F1 score vs. MAPE across all hyperparameter combinations'
)
fig_f1_mape.update_layout(
    title_x = 0.5,
    legend_title_text = '',
    legend = dict(orientation = 'h', x = 0.6, y = 1)
).show()

fig_accuracy_rmse = px.scatter(
    all_metrics, 
    x = 'match_accuracy', 
    y = 'pred_rmse', 
    color = 'model',
    labels = {'match_accuracy': 'Accuracy for best reidentification model (%)',
              'pred_rmse': 'RMSE for best prediction model (sec)'},
    title = 'Accuracy vs. RMSE across all hyperparameter combinations'
)
fig_accuracy_rmse.update_layout(
    title_x = 0.5,
    legend_title_text = '',
    legend = dict(orientation = 'h', x = 0.6, y = 1)
).show()

fig_f1_mape.write_html(os.path.join(plot_path, "fig_f1_mape.html"))
fig_accuracy_rmse.write_html(os.path.join(plot_path, "fig_accuracy_mape.html"))

# =============================================================================
# matching and prediction metrics
# =============================================================================

# function to flatten the number of match class
def flattenModelMetrics(x):
    new_x = {
        'TP': x['num_match'][0],
        'FP': x['num_match'][1],
        'FN': x['num_match'][2],
        **{k: v for k, v in x.items() if k not in ('num_match')}
    }
    return new_x

def processBestModelMetrics(model):
    file_best_metrics = model + '_best_comb_metrics.json'
    
    list_metrics = [] # store each dictionary from json file
    with open(os.path.join(result_path, file_best_metrics), 'r') as json_file:
        for line in json_file:
            list_metrics.append(json.loads(line.strip()))

    match_metrics = pd.concat([
        pd.DataFrame(flattenModelMetrics(list_metrics[1]), index = ['best_match']), # match metrics for best match
        pd.DataFrame(flattenModelMetrics(list_metrics[5]), index = ['best_pred']) # match metrics for best pred
        ])
    match_metrics.reset_index(drop = False, inplace = True)
    match_metrics.rename(columns = {'index': 'type'}, inplace = True)
    match_metrics['method'] = getModelName(model)

    pred_metrics = pd.concat([
        pd.DataFrame(list_metrics[2], index = ['best_match']),
        pd.DataFrame(list_metrics[4], index = ['best_pred'])
        ])
    pred_metrics.reset_index(drop = False, inplace = True)
    pred_metrics.rename(columns = {'index': 'type'}, inplace = True)
    pred_metrics['method'] = getModelName(model)
    
    print(getModelName(model))
    print("Hyperparameters for best reidentification model")
    print(list_metrics[0])
    print("Hyperparameters for best prediction model")
    print(list_metrics[3])
    print("\n")
    
    return {'match_metrics': match_metrics, 'pred_metrics': pred_metrics}

match_metrics_list, pred_metrics_list = [], []

for model in models:
    result_metrics = processBestModelMetrics(model)
    match_metrics_list.append(result_metrics['match_metrics'])
    pred_metrics_list.append(result_metrics['pred_metrics'])
    
match_metrics_models = pd.concat(match_metrics_list)
pred_metrics_models = pd.concat(pred_metrics_list)
result = pred_metrics_models.merge(match_metrics_models, on = ['type', 'method'], how = 'left')

# round metrics
cols_round = ['mape_valid', 'rmse_valid', 'mape_test', 'rmse_test', 'mape_test_full', 'rmse_test_full',
              'accuracy', 'precision', 'recall', 'f1']
result[cols_round] = result[cols_round].round(4)