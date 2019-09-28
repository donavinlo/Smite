import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.simplefilter(action='ignore')


########################################Preprocessing###################################################################
df = pd.read_csv('InconStats.csv')
df.drop(['Unnamed: 0', 'KDA','God', 'MMR or Elo', 'Skill Rank', 'Total_Gold', 'Level', 'Game Mode'], axis=1, inplace=True)
dfx = df.drop(['Game Result'], axis=1)
y = df['Game Result']

#Create Dummy Variables
dfx = pd.get_dummies(dfx, columns=['Relic 1', 'Relic 2', 'Ranked', 'Team Healing', 'Class'], drop_first=True)

#Split Data into test and training sets
x_train, x_test, y_train, y_test = train_test_split(dfx, y, test_size=0.2, random_state=10)

#Standardize Data
cont_col_names = ['Time Length', 'Gold Per Minute', 'Damage Dealt', 'In Hand Damage Dealt', 'Damage Taken',
                  'Damage Mitigated', 'Self Healing', 'Structure Damage', 'Wards', 'Distance Traveled', 'Kills',
                  'Deaths', 'Assists']
scaler1 = StandardScaler().fit(x_train[cont_col_names])
x_train[cont_col_names] = scaler1.transform(x_train[cont_col_names])
x_test[cont_col_names] = scaler1.transform(x_test[cont_col_names])

############################################ Model Selection ##########################################################
## Goal: Go above 55%

## Instantiate Model Types
#Random Forest
forest = RandomForestClassifier()

#KNN
knn = KNeighborsClassifier()

#Log
log = LogisticRegression()


#Train the models
#Random Forest
forest.fit(x_train, y_train)

#KNN
knn.fit(x_train, y_train)

#Log
log.fit(x_train, y_train)


#Predicitions of Model
forest_pred = forest.predict(x_test)

knn_pred = knn.predict(x_test)

log_pred = log.predict((x_test))

def cross_val_perf(model_name, model_instaniation, X, Y, subsets, score_type):
    scores = cross_val_score(model_instaniation, X, Y, cv=subsets, scoring = score_type)
    print(model_name)
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard Deviation:', scores.std())
    print('-------------------------------------------------------------------------------------------------------')


def eval_perf(model_title, real_results, pred):
    print(model_title + '\n' + classification_report(real_results, pred))
    print('\n--------------------------------------------------------------------------------------------------\n')



#Cross Validation Evaluation
#Random Forest
cross_val_perf('Random Forest', forest, x_train, y_train, 10, 'accuracy')

#KNN
cross_val_perf('KNN', knn, x_train, y_train, 10, 'accuracy')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
#Log
cross_val_perf('Logistic Regression', log, x_train, y_train, 10, 'accuracy')


#Random Forest Classification Performance
eval_perf('Random Forest', y_test, forest_pred)
#KNN Classification Performance
eval_perf('KNN', y_test, knn_pred)
#Log Classification Performance
eval_perf('Logistic Regression', y_test, log_pred)

# log_grid = dict(
#         C = [0.01, 0.1, 1, 10, 50, 100],
#         penalty = ['l1', 'l2']
#         )
def get_best_params(X, Y, param_dict, estimator, subset_num, model_name):
    #Grid Search
    grid_cv = GridSearchCV(estimator=estimator, param_grid=param_dict, cv=subset_num)
    grid_cv.fit(X, Y)
    print('Best ' + model_name + ' Parameters: {} '.format(grid_cv.best_params_))
    print('Best Score is: {}'.format(grid_cv.best_score_))
#get_best_params(x_train, y_train, log_grid, log,  5, 'Logistic Regression')
#Best Parameters for C =10, penalty = l2. Best Score =0.8875

#Fit the model using parameters found by GridSearch
log_final = LogisticRegression(C = 10, penalty='l2')
log_final.fit(x_train, y_train)
log_final_pred = log_final.predict(x_test)
eval_perf('Logistic Regression After Grid Search', y_test, log_final_pred)

#Finding the most important coefficients
def find_importance(df, model):
    columns = df.columns
    weights = model.coef_.tolist()[0]
    feature_importance = pd.DataFrame({'Feature': columns, 'Weight':np.abs(weights), 'Effect': weights})
    feature_importance.loc[feature_importance['Effect'] > 0, 'Direction'] = 'Positive'
    feature_importance.loc[feature_importance['Effect'] < 0, 'Direction'] = 'Negative'
    feature_importance.loc[feature_importance['Effect'] == 0, 'Direction'] = 'None'
    feature_importance.drop('Effect', axis=1, inplace=True)
    feature_importance = feature_importance.sort_values(by='Weight', ascending=False)
    return feature_importance
importance = find_importance(x_train, log_final)
importance.head(12).plot.bar(x='Feature')
plt.title('Feature Importance')

# #def get_vifs(df):
#     vif_df = pd.DataFrame()
#     vif_df['Feature'] = df.columns
#     vif_df['VIF Factor'] = [vif(df.values, i) for i in range(df.shape[1])]
#     vif_df = vif_df.sort_values(by='VIF Factor', ascending=False)
#     return vif_df
# get_vifs(x_train)

#Drop damage Mitigated and wards
x_train.drop(['Damage Mitigated', 'Wards'], axis=1, inplace=True)
x_test.drop(['Damage Mitigated', 'Wards'], axis=1, inplace=True)

#Fit our final/last model
log_last = LogisticRegression(C = 10, penalty='l2')
log_last.fit(x_train, y_train)
log_last_pred = log_last.predict(x_test)

importance = find_importance(x_train, log_last)
importance.head(12).plot.bar(x='Feature')
plt.title('Feature Importance')
plt.xticks(fontsize=6, rotation=45)

##############################################Model Evaluations#########################################################
#Confuaion Matrix
confusion_matrix(y_test, log_last_pred)
eval_perf('Final Logistic Regression (Damage Mitigated and Wards Dropped)', y_test, log_last_pred)

#Precision and Recall Curve Plot
prob = log_last.predict_proba(x_train)[:,1]
prec, recall, thresh = precision_recall_curve(y_train, prob)
plt.plot(thresh, prec[:-1], label="precision")
plt.plot(thresh, recall[:-1], label="recall")
plt.legend(loc='upper right')
plt.xlabel("Threshold")
plt.title('Precision/Recall Threshold')

#Roc/AUC Curve
fpr, tpr, thresh_roc = roc_curve(y_train, prob)
plt.plot(fpr, tpr, 'g-')
plt.plot([0,1], [0,1], 'r')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC AUC Curve')

roc_auc_score(y_train, prob)

