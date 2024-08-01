from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


def svm_model_maker():
    model = SVC(C=3.0,
                kernel="rbf",
                tol=1e-5,
                gamma='scale')
    return model


def svm_fit(x_train, y_train):
    model = svm_model_maker()
    model.fit(x_train, y_train)
    return model


def svm_pred(x_train, y_train, x_test):
    model = svm_fit(x_train, y_train)
    pred = model.predict(x_test)
    return pred
