from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV

X, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {"hidden_layer_sizes": [(100,), (50, 50,), (50, 30, 20)],
          "activation" : ['identity', 'logistic', 'tanh', 'relu'],
          "solver" : ['lbfgs', 'sgd', 'adam']
          }
model = GridSearchCV(estimator=MLPClassifier(max_iter=500),
                     param_grid=params,
                     scoring="accuracy")
model.fit(x_train, y_train)
model.predict(x_test)
print(model.best_params_)
print(model.best_score_)
