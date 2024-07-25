from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV

X, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {"n_estimators": [100, 200, 300, 500],
          "criterion": ["gini", "entropy", "log_loss"],
          "min_samples_split": [2, 3, 4, 5],
          "min_samples_leaf": [1, 2, 3, 4, 5]
          }
model = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),
                     param_grid=params,
                     scoring="accuracy")
model.fit(x_train, y_train)
model.predict(x_test)
print(model.best_params_)
print(model.best_score_)
