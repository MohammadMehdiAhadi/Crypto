from Models.MLPClassifier_Model import *
from Models.Knn_Model import *
from Models.RandomForestClassifier_Model import *
from Models.LogisticRegression_Model import *
from Models.SVC_Model import *
from Models.DecisionTreeClassifier_Model import *
from data_clean import *
import time
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix,roc_curve, auc



print()
print('fitting and predicting with models')
print('it takes some time')
print('please wait . . .')

l_t0 = time.time()
logistic_predict = logistic_pred(x_train, y_train, x_test)
l_t1 = time.time()
logistic_fit_predict_time = l_t1-l_t0

m_t0 = time.time()
mlp_predict = mlp_pred(x_train, y_train, x_test)
m_t1 = time.time()
mlp_fit_predict_time = m_t1-m_t0

k_t0 = time.time()
knn_predict = knn_pred(x_train, y_train, x_test)
k_t1 = time.time()
knn_fit_predict_time = k_t1-k_t0

s_t0 = time.time()
svm_predict = svm_pred(x_train, y_train, x_test)
s_t1 = time.time()
svm_fit_predict_time = s_t1-s_t0

d_t0 = time.time()
decision_tree_predict = dt_pred(x_train, y_train, x_test)
d_t1 = time.time()
decision_tree_fit_predict_time = d_t1-d_t0

r_t0 = time.time()
random_forest_predict = rf_pred(x_train, y_train, x_test)
r_t1 = time.time()
random_forest_fit_predict_time = r_t1-r_t0

print('Done')
print()
