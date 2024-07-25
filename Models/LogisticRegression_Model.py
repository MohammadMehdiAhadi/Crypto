from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits



def logistic_model_maker ():
    model = LogisticRegression(max_iter=200)
    return model


def logistic_fit(x_train,y_train):
    model = logistic_model_maker()
    model.fit(x_train,y_train)
    return model

def logistic_pred(x_train, y_train,x_test):
    model = logistic_fit(x_train,y_train)
    pred = model.predict(x_test)
    return pred
