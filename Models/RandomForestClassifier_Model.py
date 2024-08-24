from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


def rf_model_maker():
    model = RandomForestClassifier(criterion='gini', min_samples_leaf=5, min_samples_split=4, n_estimators=500)
    return model


def rf_fit(x_train, y_train):
    model = rf_model_maker()
    model.fit(x_train, y_train)
    return model


def rf_pred(x_train, y_train, x_test):
    model = rf_fit(x_train, y_train)
    pred = model.predict(x_test)
    return pred
