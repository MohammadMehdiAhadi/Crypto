import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as Knn

data = pd.read_csv("C:\\Crypto\\final_dataframe.csv", index_col="Date")

# Define features and target
X = data[['Open', 'High', 'Low', 'Close', "Tommorow_Open", 'Volume', "histogram","ema7","ema14","ema21",
          'sma', "ema", 'squeeze', 'upper_band', 'lower_band', 'macd',
          'day_of_week']]["2014-10-20 00:00:00+00:00":"2024-07-31 00:00:00+00:00"]
y = data["Benefit"]["2014-10-20 00:00:00+00:00":"2024-07-31 00:00:00+00:00"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.006, shuffle=False, random_state=17)





params = {"n_neighbors": [5,7,8,9,10,20],
          "weights": ['uniform', 'distance'],
          "algorithm": [ 'kd_tree','ball_tree', 'brute']}
model = GridSearchCV(estimator=Knn(),
                     param_grid=params,
                     scoring="accuracy")
model.fit(x_train, y_train)
model.predict(x_test)
print(model.best_params_)
print(model.best_score_)