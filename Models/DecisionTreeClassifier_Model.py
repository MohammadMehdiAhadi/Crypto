from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


def dt_model_maker():
    model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=4, min_samples_split=3, splitter='best')
    return model


def dt_fit(x_train, y_train):
    model = dt_model_maker()
    model.fit(x_train, y_train)
    return model


def dt_pred(x_train, y_train, x_test):
    model = dt_fit(x_train, y_train)
    pred = model.predict(x_test)
    return pred
