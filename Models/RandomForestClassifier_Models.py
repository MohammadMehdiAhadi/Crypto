from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


X,y = load_digits(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8)

def rf_model_maker ():
    model = RandomForestClassifier(n_estimators=100,  min_samples_split=2, min_samples_leaf=2)
    return model


def rf_fit(x_train,y_train):
    model = rf_model_maker()
    model.fit(x_train,y_train)
    return model

def rf_pred(x_train,y_train,x_test,y_test):
    model = rf_fit(x_train,y_train)
    pred = model.predict(x_test)
    return (classification_report(y_test,pred),
            accuracy_score(y_test,pred)
            )
print(rf_pred(x_train,y_train,x_test,y_test))