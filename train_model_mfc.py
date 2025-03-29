import os
import numpy as np
import pandas as pd
import glob

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
data_path = os.path.join(parent_dir, 'data')
csv_files = glob.glob(os.path.join(data_path, "*.csv"))

for file_path in csv_files:  ##读取data目录下所有csv文件
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path)
    print(f"读取文件 {file_name}.csv，DataFrame 大小为 {df.shape}")
    globals()[file_name] = df

EDAHL = data[['D_HOMO','D_LUMO','A_HOMO','A_LUMO']]
EHL = data[['HOMO','LUMO']]
OL = data[['overlap','distance']]
MFC = data[['initial','ground']]
    
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,make_scorer
from sklearn.model_selection import GridSearchCV,KFold,cross_val_score,cross_val_predict

regressors = [
    ('Decision Tree', DecisionTreeRegressor()),
    ('Random Forest', RandomForestRegressor()),
    ('Gradient Boosting', GradientBoostingRegressor()),
    ('XGBoost', XGBRegressor()),
    ('LightGBM', LGBMRegressor(verbose=-1)),
        ]

param_grid = {
     'Decision Tree': {
                      'estimator__max_depth': [None, 5, 10]
                      },
     'Random Forest': {
                       'estimator__n_estimators': [100,200],
                       'estimator__max_depth':[3],
                       'estimator__random_state':[42]
                       },
    'Gradient Boosting': {
                      'estimator__learning_rate': [0.1],
                      'estimator__n_estimators': [100],
                      'estimator__max_depth': [7],
                      'estimator__min_samples_split':[14,15],
                      'estimator__random_state':[8]
                      },
    'XGBoost': {
                'estimator__learning_rate': [0.01],
                 'estimator__max_depth': [5],
                 'estimator__gamma': [0.01],
                'estimator__reg_alpha': [0, 0.1, 0.5],
                 'estimator__reg_lambda': [1, 2, 5]
                 },
     'LightGBM': {

                 'estimator__learning_rate': [0.01],
                 'estimator__max_depth': [5],
                 'estimator__n_estimators':[100,200]
                 }
                }
def model_selection(features):
    kf = KFold(n_splits=10, shuffle=True,random_state =1)
    results = []
    for name, regressor in regressors:
        model = MultiOutputRegressor(regressor)
        grid_search = GridSearchCV(model, param_grid[name], cv=kf, 
                                   scoring='r2', n_jobs = -1,verbose = 1,return_train_score=True)
        grid_search.fit(features,MFC)

        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_

        r2_list = [grid_search.cv_results_[f'split{i}_test_score'][0] for i in range(10)]
        mae = cross_val_score(best_model, features, MFC, cv=kf, scoring='neg_mean_absolute_error')
        rmse = cross_val_score(best_model, features, MFC, cv=kf, scoring='neg_mean_squared_error')
    
        result = {
            'Model': name,
            'R2_list':r2_list,
            'MAE_list': abs(mae),
            'RMSE_list': np.sqrt(abs(rmse)),
            'R2': best_score,
            'MAE':np.mean(abs(mae)),
            'RMSE':np.mean(np.sqrt(abs(rmse)))
        }
        results.append(result)
    results_df = pd.DataFrame(results)
    filename = [x for x in globals() if globals()[x] is features][0]
    results_df.to_csv ( os.path.join ( resultSaveLocation ,filename  + '_results.csv' ) )
    return results_df

if __name__ == '__main__' :
    resultSaveLocation = '../results/MFC_result/'
    if not os.path.exists ( resultSaveLocation ) :
        os.makedirs ( resultSaveLocation )
        
    regressor_EDAHL = model_selection(EDAHL)
    
    regressor_EHL = model_selection(EHL)
    
    regressor_OL = model_selection(OL)
    
    regressor_morgan = model_selection(x_morg)
    
    regressor_daylight = model_selection(x_rd)
    
    regressor_Atompairs = model_selection(x_AP)
    
    regressor_Topological = model_selection(x_torsion)
    
    EDAHL_Morgan = pd.concat([EDAHL,x_morg],axis = 1 )
    regressor_EDAHL_Morgan = model_selection(EDAHL_Morgan)

    EDAHL_Daylight = pd.concat([EDAHL,x_rd],axis = 1 )
    regressor_EDAHL_Daylight = model_selection(EDAHL_Daylight)
    
    EDAHL_Atompairs = pd.concat([EDAHL,x_AP],axis = 1 )
    regressor_EDAHL_Atompairs = model_selection(EDAHL_Atompairs)
    
    EDAHL_Topological = pd.concat([EDAHL,x_torsion],axis = 1 )
    regressor_EDAHL_Topological = model_selection(EDAHL_Topological)
