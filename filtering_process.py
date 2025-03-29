import os
import numpy as np
import pandas as pd
import glob

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
data_path = os.path.join(parent_dir, 'data')
csv_files = glob.glob(os.path.join(data_path, "*.csv"))
com_data_path = os.path.join(parent_dir, 'combined_data')
com_csv_files = glob.glob(os.path.join(com_data_path, "*.csv"))

for file_path in csv_files:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path)
    print(f"读取文件 {file_name}.csv，DataFrame 大小为 {df.shape}")
    globals()[file_name] = df
for file_path in com_csv_files:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path)
    print(f"读取文件 {file_name}.csv，DataFrame 大小为 {df.shape}")
    globals()[file_name] = df
    
EDAHL = data[['D_HOMO','D_LUMO','A_HOMO','A_LUMO']]
EHL = data[['HOMO','LUMO']]
OL = data[['overlap','distance']]
MFC = data[['initial','ground']]
AIE = data['AIE']

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
import pickle

GB_model = MultiOutputRegressor(GradientBoostingRegressor(learning_rate = 0.1,max_depth=7,
                                    n_estimators = 100,random_state = 8,n_iter_no_change = 10))# 
X_mfc = pd.concat([EDAHL,x_morg],axis = 1)
X_mfc.columns = X_mfc.columns.astype(str)
y_mfc = MFC

# 计算差值
diff = y_mfc['ground'] - y_mfc['initial']
y_labels = pd.Series(pd.qcut(diff, q=10, labels=False), index=diff.index)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=9162)
train_indices, test_indices = next(splitter.split(X_mfc, y_labels))
X_train_mfc, y_train_mfc = X_mfc.iloc[train_indices], y_mfc.iloc[train_indices]
X_test_mfc, y_test_mfc = X_mfc.iloc[test_indices], y_mfc.iloc[test_indices]
    
GB_model.fit(X_train_mfc,y_train_mfc)

with open('../model/mfc_model.pkl', 'wb') as f:
    pickle.dump(GB_model, f)

r2_train = r2_score(y_train_mfc,GB_model.predict(X_train_mfc))
r2_test = r2_score(y_test_mfc,GB_model.predict(X_test_mfc))
    
mae_train = mean_absolute_error(y_train_mfc, GB_model.predict(X_train_mfc))
mae_test = mean_absolute_error(y_test_mfc, GB_model.predict(X_test_mfc))
    
mse_train = mean_squared_error(y_train_mfc,GB_model.predict(X_train_mfc))
mse_test = mean_squared_error(y_test_mfc,GB_model.predict(X_test_mfc))

rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

mfc_score = {
    'Metric': ['R2', 'MAE', 'MSE', 'RMSE'],
    'Train': [r2_train, mae_train, mse_train, rmse_train],
    'Test': [r2_test, mae_test, mse_test, rmse_test]
}

mfc_scores = pd.DataFrame(mfc_score)

mfc_scores.to_csv('../results/MFC_result/mfc_scores.csv', index=False)

print('mfc_model_scores:','\n','Train:','R2:',r2_train,'MAE:',mae_train,'RMSE:',rmse_train,'\n',
        'Test:','R2:',r2_test,'MAE:',mae_test,'RMSE:',rmse_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
RF_model = RandomForestClassifier(max_depth=8,n_estimators = 120,random_state = 23,min_samples_split=13)
X_aie = pd.concat([x_rd,descriptors],axis=1)
X_aie.columns = X_aie.columns.astype(str)

y_aie = AIE

X_train_aie,X_test_aie,y_train_aie,y_test_aie = train_test_split(X_aie,y_aie,test_size=0.1, random_state=31)
RF_model.fit(X_train_aie,y_train_aie)
with open('../model/aie_model.pkl', 'wb') as f:
    pickle.dump(RF_model, f)
ACC_train = accuracy_score(y_train_aie,RF_model.predict(X_train_aie))
ACC_test = accuracy_score(y_test_aie,RF_model.predict(X_test_aie))
    
F1_train = f1_score(y_train_aie, RF_model.predict(X_train_aie))
F1_test = f1_score(y_test_aie, RF_model.predict(X_test_aie))
    
AUC_train = roc_auc_score(y_train_aie,RF_model.predict(X_train_aie))
AUC_test = roc_auc_score(y_test_aie,RF_model.predict(X_test_aie))

aie_score = {
    'Metric': ['Accuracy', 'F1', 'AUC'],
    'Train': [ACC_train, F1_train, AUC_train],
    'Test': [ACC_test, F1_test, AUC_test]
}

aie_scores = pd.DataFrame(aie_score)

aie_scores.to_csv('../results/AIE_result/aie_scores.csv', index=False)

print('AIE_model_scores:','\n'
        'Train:','acc:',ACC_train,'F1:',F1_train,'auc:',AUC_train,'\n',
        'Test:','acc:',ACC_test,'F1:',F1_test,'auc:',AUC_test,'\n')



com_mfc_x = pd.concat([combined_scores[['D_HOMO','D_LUMO','A_HOMO','A_LUMO']],com_morg],axis = 1)
com_mfc_pred = GB_model.predict(com_mfc_x)
ok_index = combined_scores[(com_mfc_pred[:,1] - com_mfc_pred[:,0]) > 100].index.tolist() ## 中间苯环的筛选出的数据的索引
ok_pred = pd.DataFrame(com_mfc_pred[ok_index])
ok_pred.columns = ['initial','ground']
ok_data = combined_scores.iloc[ok_index]
ok_data_pred = pd.concat([ok_data.reset_index(drop=True), ok_pred.reset_index(drop=True)], axis=1)
ok_data_sorted = ok_data_pred.sort_values('sascore')
print('combined_filter_mfc_number:',len(ok_data_sorted))
ok_data_sorted['Combined_SMILES'].to_csv('../combined_filter_mfc.csv',index = False)

com_aie_x = pd.concat([com_rd,com_descriptors],axis = 1)
com_aie_predict = RF_model.predict(com_aie_x)
num_ones = np.sum(com_aie_predict.astype(bool))
idx_ones = np.where(com_aie_predict == 1)[0]
combined_filter_aie = combined_scores.iloc[idx_ones]['Combined_SMILES']
print('combined_filter_aie_number:',len(combined_filter_aie))
combined_filter_aie.to_csv('../combined_filter_aie.csv',index = False)

set1 = set(ok_data_sorted['Combined_SMILES'])
set2 = set(combined_filter_aie)
common_elements = set1.intersection(set2)

common_indices = [i for i, x in enumerate(ok_data_sorted['Combined_SMILES']) if x in common_elements]
aie_mfc_data = ok_data_sorted.iloc[common_indices]

aie_mfc_data.to_csv('../aie_mfc_data.csv',index = False)













