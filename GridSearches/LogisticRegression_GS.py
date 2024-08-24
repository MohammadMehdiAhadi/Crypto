from Models.Final_Model import *
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
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


    params = {"penalty": ['l1', 'l2'],
              "C": [5,7,9],
              "solver" : ['lbfgs', 'liblinear', 'saga'],
              "max_iter" : [250,500,300,700]
              }
    model = GridSearchCV(estimator=LogisticRegression(),
                         param_grid=params,
                         scoring="accuracy")
    model.fit(x_train, y_train)
    model.predict(x_test)
    print(model.best_params_)
    print(model.best_score_)
