# Reading the Dataset
from typing import Union, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import neighbors as nb
import sklearn.model_selection as model_selection
import sklearn.tree as tree
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
import pickle

# import seaborn as sns
data = pd.read_csv('mypolls/owid-covid-data_new.csv')
for i in data.columns:
    if data[i].dtype != type(object):
        q1 = data[i].quantile(0.25)
        q3 = data[i].quantile(0.75)
        IQR = q3 - q1
        ub = q3 + (1.5 * IQR)
        lb = q1 - (1.5 * IQR)
        for j in range(0, len(data[i])):
            if data[i][j] > ub:
                data[i].replace(data[i][j], ub, inplace=True)
            elif data[i][j] < lb:
                data[i].replace(data[i][j], lb, inplace=True)

for i in data.columns:
    if data[i].dtype != type(object):
        q1 = data[i].quantile(0.25)
        q3 = data[i].quantile(0.75)
        IQR = q3 - q1
        ub = q3 + (1.5 * IQR)
        lb = q1 - (1.5 * IQR)
        out_ub = data[data[i] > ub][i]
        out_lb = data[data[i] < lb][i]
        print("no of outliers in", i, "is", len(out_ub) + len(out_lb))

numerical_feats = data.dtypes[data.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = data.dtypes[data.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))

data.dropna(how='all', inplace=True)
data.fillna(value=0, axis=0, inplace=True)

X = data.drop(['iso_code', 'continent', 'location', 'date', 'tests_units'], axis=1)
Y = data['life_expectancy']
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print(model.summary())

# cat_X = data.drop(list(numerical_feats), axis=1)
# cat_Y = data['life_expectancy']
# model = smf.ols(formula="life_expectancy ~ C(iso_code)+C(continent)+C(location)+C(date)+C(tests_units)",
#                 data=data).fit()
# print(model.summary())

Drop_feats = ['human_development_index', 'hospital_beds_per_thousand', 'handwashing_facilities', 'male_smokers',
              'female_smokers',
              'diabetes_prevalence', 'extreme_poverty', 'aged_70_older', 'aged_65_older', 'median_age', 'total_tests',
              'weekly_hosp_admissions_per_million',
              'weekly_hosp_admissions', 'icu_patients_per_million', 'icu_patients', 'new_deaths_smoothed_per_million',
              'iso_code',
              'continent', 'location', 'date', 'tests_units', 'total_vaccinations', 'people_vaccinated',
              'people_fully_vaccinated',
              'people_fully_vaccinated_per_hundred', 'new_vaccinations_smoothed_per_million']
data.drop(Drop_feats, axis=1, inplace=True)

X = data.drop('life_expectancy', axis=1)
X.corr()

scaler = MinMaxScaler()
X = scaler.fit_transform(data.drop('life_expectancy', axis=1))
Y = scaler.fit_transform(data['life_expectancy'].values.reshape(-1, 1))

bestfeatures = SelectKBest(score_func=f_regression, k=10)
fit = bestfeatures.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(data.drop('life_expectancy', axis=1).columns)
# Concat two Data Frames for Better Visulaization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
print(featureScores)

important_feats = featureScores.nlargest(12, 'Score')['Specs']
important_feats = important_feats.to_list()

# important_feats = ['gdp_per_capita', 'stringency_index', 'reproduction_rate',
#                    'new_tests_smoothed_per_thousand', 'population_density', 'population',
#                    'new_tests_per_thousand', 'new_tests_smoothed', 'total_tests_per_thousand',
#                    'new_tests', 'tests_per_case' 'new_vaccinations']

X = data[important_feats]
y = data['life_expectancy']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
Y = scaler.fit_transform(data['life_expectancy'].values.reshape(-1, 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#### instantiate and fit
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
LR_score_train = model_LR.score(X_train, Y_train)
print("Train Accuracy :", LR_score_train)

LR_score_test = model_LR.score(X_test, Y_test)
print("Test Accuracy  :", LR_score_test)
predictions_LR = model_LR.predict(X_test)

print('MAE:', metrics.mean_absolute_error(Y_test, predictions_LR))
print('MSE:', metrics.mean_squared_error(Y_test, predictions_LR))
LR_RMSE = np.sqrt(metrics.mean_squared_error(Y_test, predictions_LR))
print('RMSE:', LR_RMSE)
final_results = []
dict_LR = {'MODEL': 'Linear Regression',
           'Train_ACCURACY': LR_score_train,
           'Test_ACCURACY': LR_score_test,
           'RMSE': LR_RMSE
           }
final_results.append(dict_LR)

model_LR_Ridge = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
model_LR_Ridge.fit(X_train, Y_train)
LR_Ridge_score_train = model_LR_Ridge.score(X_train, Y_train)
print("Train Accuracy :", LR_Ridge_score_train)

LR_Ridge_score_test = model_LR_Ridge.score(X_test, Y_test)
print("Test Accuracy  :", LR_Ridge_score_test)

predictions_LR_Ridge = model_LR_Ridge.predict(X_test)
print('MAE:', metrics.mean_absolute_error(Y_test, predictions_LR_Ridge))
print('MSE:', metrics.mean_squared_error(Y_test, predictions_LR_Ridge))
LR_Ridge_RMSE = np.sqrt(metrics.mean_squared_error(Y_test, predictions_LR_Ridge))
print('RMSE:', LR_Ridge_RMSE)

dict_LR_Ridge = {'MODEL': 'Ridge Regression',
                 'Train_ACCURACY': LR_Ridge_score_train,
                 'Test_ACCURACY': LR_Ridge_score_test,
                 'RMSE': LR_Ridge_RMSE
                 }
final_results.append(dict_LR_Ridge)

model_LR_Lasso = linear_model.LassoCV(alphas=np.logspace(-6, 6, 13))
model_LR_Lasso.fit(X_train, Y_train)

LR_Lasso_score_train = model_LR_Lasso.score(X_train, Y_train)
print("Train Accuracy :", LR_Lasso_score_train)

LR_Lasso_score_test = model_LR_Lasso.score(X_test, Y_test)
print("Test Accuracy  :", LR_Lasso_score_test)

predictions_LR_Lasso = model_LR_Lasso.predict(X_test)
print('MAE:', metrics.mean_absolute_error(Y_test, predictions_LR_Lasso))
print('MSE:', metrics.mean_squared_error(Y_test, predictions_LR_Lasso))
LR_Lasso_RMSE = np.sqrt(metrics.mean_squared_error(Y_test, predictions_LR_Lasso))
print('RMSE:', LR_Lasso_RMSE)

dict_LR_Lasso = {'MODEL': 'Lasso Regression',
                 'Train_ACCURACY': LR_Lasso_score_train,
                 'Test_ACCURACY': LR_Lasso_score_test,
                 'RMSE': LR_Lasso_RMSE
                 }
final_results.append(dict_LR_Lasso)

model_KNN = nb.KNeighborsRegressor(n_neighbors=4, n_jobs=-1)
model_KNN.fit(X_train, Y_train)

KNN_score_train = model_KNN.score(X_train, Y_train)
print("Train Accuracy :", KNN_score_train)

KNN_score_test = model_KNN.score(X_test, Y_test)
print("Test Accuracy  :", KNN_score_test)

model_KNN1 = model_selection.GridSearchCV(model_KNN, param_grid={'n_neighbors': [i for i in range(1, 13)],
                                                                 'weights': ['uniform', 'distance']})
model_KNN1.fit(X_train, Y_train)
print(model_KNN1.best_params_)
KNN_score_train = model_KNN1.score(X_train, Y_train)
print("Train Accuracy :", KNN_score_train)

KNN_score_test = model_KNN1.score(X_test, Y_test)
print("Test Accuracy  :", KNN_score_test)

predictions_KNN = model_KNN1.predict(X_test)

print('MAE:', metrics.mean_absolute_error(Y_test, predictions_KNN))
print('MSE:', metrics.mean_squared_error(Y_test, predictions_KNN))
KNN_RMSE = np.sqrt(metrics.mean_squared_error(Y_test, predictions_KNN))
print('RMSE:', KNN_RMSE)

# dict_KNN = {'MODEL':'KNN Regressor',
#             'Train_ACCURACY':KNN_score_train,
#             'Test_ACCURACY':KNN_score_test,
#             'RMSE':KNN_RMSE}
# final_results.append(dict_KNN)

#
# model_DT=tree.DecisionTreeRegressor(max_depth=3)
# model_DT.fit(X_train,Y_train)
# DT_score_train = model_DT.score(X_train,Y_train)
# print("Train Accuracy :",DT_score_train)
#
# DT_score_test = model_DT.score(X_test,Y_test)
# print("Test Accuracy  :",DT_score_test)

# features  = important_feats
# plt.figure(figsize=(15,8))
# plot_tree(model_DT, feature_names=features, filled = True)
#
# print('MAE:', metrics.mean_absolute_error(Y_test, predictions_DT))
# print('MSE:', metrics.mean_squared_error(Y_test, predictions_DT))
# DT_RMSE=np.sqrt(metrics.mean_squared_error(Y_test, predictions_DT))
# print('RMSE:', DT_RMSE)

#
# dict_DT = {'MODEL':'DT Regressor',
#            'Train_ACCURACY':DT_score_train,
#            'Test_ACCURACY':DT_score_test,
#            'RMSE':DT_RMSE}
# final_results.append(dict_DT)


# list_RFR=[]
# #Tune number of trees
# for i in range(10,200,10):
#     model_RFR=RandomForestRegressor(n_estimators=i,random_state=10)
#     model_RFR.fit(X_train,Y_train)
#     dict_RFR={}
#     dict_RFR["Number of trees"] = str(i)
#     dict_RFR["ACCURACY"]=model_RFR.score(X_test,Y_test)
#     list_RFR.append(dict_RFR)
# (pd.DataFrame(list_RFR)
#      .sort_values(by=['ACCURACY'],ascending=False)
#      .reset_index(drop=True)
#      .style.background_gradient(cmap='Blues'))
#
#
# model_RFR = RandomForestRegressor(n_estimators=30,random_state=10)
# model_RFR.fit(X_train, Y_train)
#
# RFR_score_train = model_RFR.score(X_train, Y_train)
# print("Train Accuracy :",RFR_score_train)
# RFR_score_test = model_RFR.score(X_test,Y_test)
# print("Test Accuracy  :",RFR_score_test)
# predictions_RFR = model_RFR.predict(X_test)
# print('MAE:', metrics.mean_absolute_error(Y_test, predictions_RFR))
# print('MSE:', metrics.mean_squared_error(Y_test, predictions_RFR))
# RFR_RMSE=np.sqrt(metrics.mean_squared_error(Y_test, predictions_RFR))
# print('RMSE:', RFR_RMSE)
#
# dict_RFR = {'MODEL':'Random Forest Regressor',
#             'Train_ACCURACY':RFR_score_train,
#             'Test_ACCURACY':RFR_score_test,
#             'RMSE':RFR_RMSE
#            }
# final_results.append(dict_RFR)
#
# list_BR=[]
# Tune number of trees
# for i in range(10,200,10):
#     model_BR=BaggingRegressor(n_estimators=i,oob_score=True,random_state=200)
#     model_BR.fit(X_train,Y_train)
#     dict_BR={}
#     dict_BR["Number of trees"] = str(i)
#     dict_BR["ACCURACY"]=model_BR.score(X_test,Y_test)
#     list_BR.append(dict_BR)
# (pd.DataFrame(list_BR)
#      .sort_values(by=['ACCURACY'],ascending=False)
#      .reset_index(drop=True)
#      .style.background_gradient(cmap='Blues'))


# model_BR=BaggingRegressor(n_estimators=30,oob_score=True,random_state=200)
# model_BR.fit(X_train,Y_train)
#
# BR_score_train = model_BR.score(X_train,Y_train)
# print("Train Accuracy :",BR_score_train)
# BR_score_test = model_BR.score(X_test,Y_test)
# print("Test Accuracy :",BR_score_test)
# predictions_BR = model_BR.predict(X_test)
# print('MAE:', metrics.mean_absolute_error(Y_test, predictions_BR))
# print('MSE:', metrics.mean_squared_error(Y_test, predictions_BR))
# BR_RMSE=np.sqrt(metrics.mean_squared_error(Y_test, predictions_BR))
# print('RMSE:', BR_RMSE)
#
# dict_BR = {'MODEL':'Bagging Regressor',
#            'Train_ACCURACY':BR_score_train,
#            'Test_ACCURACY':BR_score_test,
#            'RMSE':BR_RMSE
#           }
# final_results.append(dict_BR)


# model_ENR = ElasticNet(random_state=0)
# model_ENR.fit(X_train,Y_train)

# ENR_score_train = model_ENR.score(X_train,Y_train)
# print("Train Accuracy :",ENR_score_train)
#
# ENR_score_test = model_ENR.score(X_test,Y_test)
# print("Test Accuracy  :",ENR_score_test)
# predictions_ENR = model_ENR.predict(X_test)
# print('MAE:', metrics.mean_absolute_error(Y_test, predictions_ENR))
# print('MSE:', metrics.mean_squared_error(Y_test, predictions_ENR))
# ENR_RMSE=np.sqrt(metrics.mean_squared_error(Y_test, predictions_ENR))
# print('RMSE:', ENR_RMSE)
#
# dict_ENR = {'MODEL':'ElasticNet Regressor',
#             'Train_ACCURACY':ENR_score_train,
#             'Test_ACCURACY':ENR_score_test,
#             'RMSE':ENR_RMSE
#            }
# final_results.append(dict_ENR)
#

# model_GBR = GradientBoostingRegressor()
# model_GBR.fit(X_train,Y_train)
# GBR_score_train = model_GBR.score(X_train,Y_train)
# print("Train Accuracy :",GBR_score_train)
# GBR_score_test = model_GBR.score(X_test,Y_test)
# print("Test Accuracy  :",GBR_score_test)
# predictions_GBR = model_GBR.predict(X_test)
# print('MAE:', metrics.mean_absolute_error(Y_test, predictions_GBR))
# print('MSE:', metrics.mean_squared_error(Y_test, predictions_GBR))
# GBR_RMSE=np.sqrt(metrics.mean_squared_error(Y_test, predictions_GBR))
# print('RMSE:', GBR_RMSE)
# dict_GBR = {'MODEL':'Gradient Boosting Regressor',
#             'Train_ACCURACY':GBR_score_train,
#             'Test_ACCURACY':GBR_score_test,
#             'RMSE':GBR_RMSE
#            }
# final_results.append(dict_GBR)
#
#
# model_HGBR = HistGradientBoostingRegressor()
# model_HGBR.fit(X_train,Y_train)
# HGBR_score_train = model_HGBR.score(X_train,Y_train)
# print("Train Accuracy :",HGBR_score_train)
# HGBR_score_test = model_HGBR.score(X_test,Y_test)
# print("Test Accuracy  :",HGBR_score_test)
# predictions_HGBR = model_HGBR.predict(X_test)
# print('MAE:', metrics.mean_absolute_error(Y_test, predictions_HGBR))
# print('MSE:', metrics.mean_squared_error(Y_test, predictions_HGBR))
# HGBR_RMSE=np.sqrt(metrics.mean_squared_error(Y_test, predictions_HGBR))
# print('RMSE:', HGBR_RMSE)
# dict_HGBR = {'MODEL':'Histogram Based GBR',
#              'Train_ACCURACY':HGBR_score_train,
#              'Test_ACCURACY':HGBR_score_test,
#              'RMSE':HGBR_RMSE
#             }
# final_results.append(dict_HGBR)
#
#
# model_XGBR = XGBRegressor()
# model_XGBR.fit(X_train,Y_train)
# XGBR_score_train = model_XGBR.score(X_train,Y_train)
# print("Train Accuracy :",XGBR_score_train)
# XGBR_score_test = model_XGBR.score(X_test,Y_test)
# print("Train Accuracy :",XGBR_score_test)
# predictions_XGBR = model_XGBR.predict(X_test)
# print('MAE:', metrics.mean_absolute_error(Y_test, predictions_XGBR))
# print('MSE:', metrics.mean_squared_error(Y_test, predictions_XGBR))
# XGBR_RMSE=np.sqrt(metrics.mean_squared_error(Y_test, predictions_XGBR))
# print('RMSE:', XGBR_RMSE)
# dict_XGBR = {'MODEL':'XGBoost Regressor',
#              'Train_ACCURACY':XGBR_score_train,
#              'Test_ACCURACY':XGBR_score_test,
#              'RMSE':XGBR_RMSE
#             }
# final_results.append(dict_XGBR)
#
#
# model_LGBR = LGBMRegressor()
# model_LGBR.fit(X_train,Y_train)
# LGBR_score_train = model_LGBR.score(X_train,Y_train)
# print("Train Accuracy :",LGBR_score_train)
# LGBR_score_test = model_LGBR.score(X_test,Y_test)
# print("Test Accuracy  :",LGBR_score_test)
# predictions_LGBR = model_LGBR.predict(X_test)
# print('MAE:', metrics.mean_absolute_error(Y_test, predictions_LGBR))
# print('MSE:', metrics.mean_squared_error(Y_test, predictions_LGBR))
# LGBR_RMSE=np.sqrt(metrics.mean_squared_error(Y_test, predictions_LGBR))
# print('RMSE:', LGBR_RMSE)
# dict_LGBR = {'MODEL':'LightGBM Regressor',
#              'Train_ACCURACY':LGBR_score_train,
#              'Test_ACCURACY':LGBR_score_test,
#              'RMSE':LGBR_RMSE
#             }
# final_results.append(dict_LGBR)
pickle.dump(scaler, open('Scaler.pkl', 'wb'))
pickle.dump(model_KNN, open('knn_model.pkl', 'wb'))
