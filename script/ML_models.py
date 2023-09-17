import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

os.chdir(r"D:\GitHub\vehicle_reidentification")

file_train = "data/final_match_pairs_train.txt"
file_test = "data/final_match_pairs_ground_truth.txt"
df_train = pd.read_csv(file_train, sep = '\t')
df_test = pd.read_csv(file_test, sep = '\t')

X = df_train.drop('travel_time', axis = 1)
y = df_train.travel_time
rand_state = 42

# split training dataset into train, validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = rand_state)

# test dataset from ground-truth
X_test = df_test.drop('travel_time', axis = 1)
y_test = df_test.travel_time

# =============================================================================
# EDA: correlation between features
# =============================================================================

corr = df_train.corr()

# plot heatmap
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df_train.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df_train.columns)
ax.set_yticklabels(df_train.columns)
plt.show()

# =============================================================================
# function to perform cross-validation
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

def evaluatePerformance(model_best_fit):
    # make predictions on validation and test sets
    y_pred_valid = model_best_fit.predict(X_valid)
    y_pred_test = model_best_fit.predict(X_test)

    # evaluation metrics on validation set
    mape_valid = mean_absolute_percentage_error(y_valid, y_pred_valid)
    rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))

    # evaluation metrics on test set
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("MAPE on validation set: {:.4f}".format(mape_valid))
    print("MAPE on test set: {:.4f}".format(mape_test))

    print("RMSE on validation set: {:.4f}".format(rmse_valid))
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

# train final model with the tuned parameters
best_ridge_reg = Ridge(**ridge_grid_result['best_params'])
best_ridge_reg.fit(X_train, y_train)

evaluatePerformance(best_ridge_reg)

# =============================================================================
# Decision Tree Regression
# =============================================================================

method = 'Decision Tree Regression'
print(method)

# create decision tree regression model
dt_reg = DecisionTreeRegressor()

# hyperparameters for grid search
dt_param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'criterion': ['friedman_mse', 'absolute_error']
}

# results from grid search
dt_grid_result = performGridSearch(dt_reg, dt_param_grid)

# train final model with the tuned parameters
best_dt_reg = DecisionTreeRegressor(**dt_grid_result['best_params'])
best_dt_reg.fit(X_train, y_train)

evaluatePerformance(best_dt_reg)

# =============================================================================
# Support Vector Regression
# =============================================================================

method = 'Support Vector Regression'
print(method)

# create support vector regression model
sv_reg = SVR()

# hyperparameters for grid search
sv_param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.5],
}

# results from grid search
sv_grid_result = performGridSearch(sv_reg, sv_param_grid)

# train final model with the tuned parameters
best_sv_reg = SVR(**sv_grid_result['best_params'])
best_sv_reg.fit(X_train, y_train)

evaluatePerformance(best_sv_reg)

# =============================================================================
# Random Forest
# =============================================================================

method = 'Random Forest'
print(method)

# create random forest model
rf_reg = RandomForestRegressor()

# hyperparameters for grid search
rf_param_grid = {
    'n_estimators': [25, 50, 100, 150, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# results from grid search
rf_grid_result = performGridSearch(rf_reg, rf_param_grid)

# train final ridge regression model with the best parameters on training dataset
best_rf_reg = RandomForestRegressor(**rf_grid_result['best_params'], random_state = 42)
best_rf_reg.fit(X_train, y_train)

evaluatePerformance(best_rf_reg)

# =============================================================================
# XGBoost
# =============================================================================

method = 'XGBoost'
print(method)

# create XGBoost model
xgb_reg = xgb.XGBRegressor(objective = 'reg:squarederror', random_state = rand_state)

# hyperparameters for grid search
xgb_param_grid = {
    'n_estimators': [25, 50, 100, 150, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [None, 3, 4, 5],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'colsample_bylevel': [0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# get results from grid search
xgb_grid_result = performGridSearch(xgb_reg, xgb_param_grid)

# train final ridge regression model with the best alpha on training dataset
best_xgb_reg = xgb.XGBRegressor(**xgb_grid_result['best_params'], random_state = 42)
best_xgb_reg.fit(X_train, y_train)

evaluatePerformance(best_rf_reg)

xgb_reg.fit(X_train, y_train) # model fitting
y_pred = xgb_reg.predict(X_test) # predicting

xgb_mape = mean_absolute_percentage_error(y_test, y_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAPE for {}: {:.4f}".format(method, xgb_mape))
print("RMSE for {}: {:.4f}".format(method, xgb_rmse), "\n")

# # feature importance
# feature_importances = xgb_reg.feature_importances_
    
# # data frame of features and importances
# importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
# importance_df = importance_df.sort_values(by='Importance', ascending=False) # sorting

# # plot feature importances
# plt.figure(figsize=(10, 6))
# plt.barh(importance_df['Feature'], importance_df['Importance'])
# plt.xlabel('Feature Importance')
# plt.ylabel('Feature')
# plt.title('XGBoost Feature Importances')
# plt.show()
