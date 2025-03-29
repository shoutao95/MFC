import os
import numpy as np
import pandas as pd
import glob
import warnings
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
AIE = data['AIE']

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def model_selection(features):
    warnings.filterwarnings('ignore')
    classifiers = [
        ('GradientBoosting', GradientBoostingClassifier(), {
            'classification__learning_rate': [0.01],
            'classification__n_estimators': [700],
            'classification__max_depth': [4],
            'classification__min_samples_split':[6],
            'classification__random_state':[20]
        }),
        ('RandomForest', RandomForestClassifier(), {
            'classification__n_estimators': [110,115,120,125,130],
            'classification__max_depth': [8],
            'classification__min_samples_split': [13],
            'classification__random_state':[4,5,6,15,20,23,42]
        }),
        ('KNN', KNeighborsClassifier(), {
              'classification__n_neighbors': [1],
              'classification__weights': ['uniform'],
              'classification__algorithm': ['ball_tree'],
              'classification__leaf_size': [10],
              'classification__p': [1],
              'classification__n_jobs': [-1]
         }),
         ('LogisticRegression', LogisticRegression(), {
             'classification__C': [0.1, 1, 10]
         })
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    results = []

    for name, classifier, param_grid in classifiers:
        pipeline = Pipeline([
            ('sampling', RandomOverSampler(random_state=30)),
            ('classification', classifier)
        ])

        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=skf, scoring='accuracy', verbose=1,n_jobs = -1,return_train_score=True)
        grid_search.fit(features, AIE)

        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        
        acc_list = [grid_search.cv_results_[f'split{i}_test_score'][0] for i in range(5)]
        f1_list = cross_val_score(best_model, features, AIE, cv=skf, scoring='f1')
        auc_list = cross_val_score(best_model, features, AIE, cv=skf, scoring='roc_auc')
        
        result = {
            'Model': name,
            'ACC_list':acc_list,
            'F1_list':f1_list,
            'AUC_list':auc_list,
            'ACC': best_score,
            'F1': f1_list.mean(),
            'AUC': auc_list.mean()
        }

        results.append(result)
    results_df = pd.DataFrame(results)
    filename = [x for x in globals() if globals()[x] is features][0]
    results_df.to_csv ( os.path.join ( resultSaveLocation ,filename  + '_results.csv' ) )
    return results_df
    
if __name__ == '__main__' :
    resultSaveLocation = '../results/AIE_result/'
    if not os.path.exists ( resultSaveLocation ) :
        os.makedirs ( resultSaveLocation )
        
    classifier_EDAHL = model_selection(EDAHL)
    
    classifier_EHL = model_selection(EHL)
    
    classifier_OL = model_selection(OL)
    
    classifier_morgan = model_selection(x_morg)
    
    classifier_daylight = model_selection(x_rd)
    
    classifier_Atompairs = model_selection(x_AP)
    
    classifier_Topological = model_selection(x_torsion)
    
    EDAHL_Morgan = pd.concat([EDAHL,x_morg],axis = 1 )
    classifier_EDAHL_Morgan = model_selection(EDAHL_Morgan)

    EDAHL_Daylight = pd.concat([EDAHL,x_rd],axis = 1 )
    classifier_EDAHL_Daylight = model_selection(EDAHL_Daylight)
    
    EDAHL_Atompairs = pd.concat([EDAHL,x_AP],axis = 1 )
    classifier_EDAHL_Atompairs = model_selection(EDAHL_Atompairs)
    
    EDAHL_Topological = pd.concat([EDAHL,x_torsion],axis = 1 )
    classifier_EDAHL_Topological = model_selection(EDAHL_Topological)
    
    classifier_descriptors = model_selection(descriptors)
    
    Morgan_descriptors = pd.concat([x_morg,descriptors],axis = 1)
    classifier_Morgan_descriptors = model_selection(Morgan_descriptors)
    
    Daylight_descriptors = pd.concat([x_rd,descriptors],axis = 1)
    classifier_Daylight_descriptors = model_selection(Daylight_descriptors)
    
    Atompairs_descriptors = pd.concat([x_AP,descriptors],axis = 1)
    classifier_Atompairs_descriptors = model_selection(Atompairs_descriptors)
    
    Topological_descriptors = pd.concat([x_torsion,descriptors],axis = 1)
    classifier_Topological_descriptors = model_selection(Topological_descriptors)
    
    
