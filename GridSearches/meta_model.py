import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from Models.MLPClassifier_Model import *
from Models.Knn_Model import *
from Models.RandomForestClassifier_Model import *
from Models.LogisticRegression_Model import *
from Models.SVC_Model import *
from Models.DecisionTreeClassifier_Model import *

from Models.Final_Model import *

data = pd.read_csv("final_dataframe.csv")
if not data.empty:
    n = len(data)

    # Define features and target
    X = data[['Open', 'High', 'Low', 'Close', "Next_Hour_Open", 'Volume', "histogram",
              "ema7", "ema14", "ema21",
              'sma', 'squeeze', 'upper_band', 'lower_band', 'macd',
              'day_of_week']].iloc[29:n - 1]
    y = data["Benefit"].iloc[29:n - 1]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.0005, shuffle=False, random_state=17)

    # Stacking predictions
    predictions_stacking = np.vstack([logistic_pred(x_train, y_train, x_test),
                                      mlp_pred(x_train, y_train, x_test),
                                      knn_pred(x_train, y_train, x_test),
                                      svm_pred(x_train, y_train, x_test),
                                      dt_pred(x_train, y_train, x_test),
                                      rf_pred(x_train, y_train, x_test)]).T

    # Define models and parameter grids
    models = {
        'knn': KNeighborsClassifier(),
        'mlp': MLPClassifier(max_iter=1000),
        'rf': RandomForestClassifier(),
        'svc': SVC(),
        'dt': DecisionTreeClassifier(),
        'lr': LogisticRegression(max_iter=1000)
    }

    params = {
        'knn': {'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']},
        'mlp': {'hidden_layer_sizes': [(200,), (100,)],
                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'solver': ['adam', 'sgd']},
        'rf': {'n_estimators': [400, 200],
               "criterion": ["gini", "entropy", "log_loss"],
               "min_samples_split": [2, 3, 4],
               "min_samples_leaf": [1, 2, 3]},
        'svc': {'C': [0.1, 1, 5], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
        'dt': {'criterion': ['gini', 'entropy', 'log_loss'],
               'min_samples_split': [5, 6, 7],
               'min_samples_leaf': [3, 4, 5]},
        'lr': {"penalty": ['l1', 'l2'],
               "C": [5, 7, 9],
               "solver": ['lbfgs', 'liblinear', 'saga']}
    }

    best_models = {}
    for name, model in models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=params[name], scoring='accuracy', cv=5)
        grid_search.fit(predictions_stacking, y_test)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best score for {name}: {grid_search.best_score_}")

    # Print the best model
    best_model_name = max(best_models, key=lambda name: best_models[name].score(predictions_stacking, y_test))
    best_model = best_models[best_model_name]
    print(f"Best model: {best_model_name} with score {best_model.score(predictions_stacking, y_test)}")

# predictions_stacking, y_test, predictions_stacking
