from Models.MLPClassifier_Model import *
from Models.Knn_Model import *
from Models.RandomForestClassifier_Model import *
from Models.LogisticRegression_Model import *
from Models.SVC_Model import *
from Models.DecisionTreeClassifier_Model import *
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix,roc_curve, auc
print('reading data from the csv file')
print('please wait . . .')
data = pd.read_csv("final_dataframe.csv", index_col="Date")

if not data.empty:
    n = len(data)

    # Define features and target
    X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Next_Hour_Open',
              'ema6', 'ema12', 'ema24', 'ema48', 'ema72', 'histogram',
              'sma', 'squeeze', 'upper_band', 'lower_band', 'macd',
              'day_of_week']].iloc[71:n - 1]
    y = data["Benefit"].iloc[71:n - 1]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.0005, shuffle=False)
    print('Done ')
    print()
    print('fitting and predicting with models')
    print('it takes some time')
    print('please wait . . .')
    logistic_predict = logistic_pred(x_train, y_train, x_test)
    mlp_predict = mlp_pred(x_train, y_train, x_test)
    knn_predict = knn_pred(x_train, y_train, x_test)
    svm_predict = svm_pred(x_train, y_train, x_test)
    decision_tree_predict = dt_pred(x_train, y_train, x_test)
    random_forest_predict = rf_pred(x_train, y_train, x_test)
    print('Done')
    print()
else:
    print("Couldn't Load The Data")
    print("Try Again")