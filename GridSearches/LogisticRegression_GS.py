from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV

X, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {"penalty": ['l1', 'l2'],
          "dual": [True, False],
          "C": [1,3, 5, 7],
          "solver" : ['lbfgs', 'liblinear', 'saga'],
          "max_iter" : [100,250,500]
          }
model = GridSearchCV(estimator=LogisticRegression(n_jobs=-1),
                     param_grid=params,
                     scoring="accuracy")
model.fit(x_train, y_train)
model.predict(x_test)
print(model.best_params_)
print(model.best_score_)
