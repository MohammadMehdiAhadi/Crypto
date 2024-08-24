from sklearn.neural_network import MLPClassifier

from Models.Final_Model import *
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

    params = {"hidden_layer_sizes": [(100,), (200,), (300,), (50, 50,), (50, 30, 20,)],
              "activation": ['identity', 'logistic', 'tanh', 'relu'],
              "solver": ['lbfgs', 'sgd', 'adam'],
              "max_iter": [1000,1500]
              }
    model = GridSearchCV(estimator=MLPClassifier(),
                         param_grid=params,
                         scoring="accuracy")
    model.fit(x_train, y_train)
    model.predict(x_test)
    print(model.best_params_)
    print(model.best_score_)
