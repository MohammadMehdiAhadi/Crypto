import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from Models.MLPClassifier_Model import *
from Models.Knn_Model import *
from Models.RandomForestClassifier_Model import *
from Models.LogisticRegression_Model import *
from Models.SVC_Model import *
from Models.DecisionTreeClassifier_Model import *

from Models.Final_Knn_Model import *

data = pd.read_csv("final_dataframe.csv", index_col="Date")
if not data.empty:

    n = len(data)

    # Define features and target
    X = data[['Open', 'High', 'Low', 'Close', "Next_Hour_Open", 'Volume', "histogram", "ema7", "ema14", "ema21",
              'sma', 'squeeze', 'upper_band', 'lower_band', 'macd',
              'day_of_week']].iloc[29:n - 1]
    y = data["Benefit"].iloc[29:n - 1]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.005, shuffle=False, random_state=17)

    # Stacking predictions
    predictions_stacking = np.vstack([logistic_pred(x_train, y_train, x_test),
        mlp_pred(x_train, y_train, x_test),
        knn_pred(x_train, y_train, x_test),
        svm_pred(x_train, y_train, x_test),
        dt_pred(x_train, y_train, x_test),
        rf_pred(x_train, y_train, x_test)]).T



models = [KNeighborsClassifier(),MLPClassifier(),RandomForestClassifier(),SVC(),DecisionTreeClassifier(),LogisticRegression()]
params = {}
model = GridSearchCV(estimator=models,
                     param_grid=params,
                     scoring="accuracy")
model.fit(predictions_stacking, y_test)
model.predict(predictions_stacking)
print(model.best_params_)
print(model.best_score_)
print(model.best_estimator_)
# predictions_stacking, y_test, predictions_stacking