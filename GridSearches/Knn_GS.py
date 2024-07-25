from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV

X, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {"n_neighbors": [5, 10, 15],
          "weights": ['uniform', 'distance'],
          "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']}
model = GridSearchCV(estimator=Knn(),
                     param_grid=params,
                     scoring="accuracy")
model.fit(x_train, y_train)
model.predict(x_test)
print(model.best_params_)
print(model.best_score_)
