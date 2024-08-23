from sklearn.svm import SVC

from Models.Final_Knn_Model import *
from sklearn.model_selection import GridSearchCV

# Load data from CSV
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

    params = {"C": [3.0, 2.0],
              "kernel": ['poly', 'rbf', 'sigmoid', 'precomputed']
              }
    model = GridSearchCV(estimator=SVC(tol=1e-5, gamma='scale'),
                         param_grid=params,
                         scoring="accuracy",
                         )
    model.fit(x_train, y_train)
    model.predict(x_test)
    print(model.best_params_)
    print(model.best_score_)

