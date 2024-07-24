from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


X,y = load_digits(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8)


def knn_model_maker ():
    model = Knn(n_jobs=-1)
    return model

def knn_fit (x_train,y_train):
    model = knn_model_maker()
    model.fit(x_train,y_train)
    return model

def knn_pred(x_train,y_train,x_test,y_test):
    model = knn_fit(x_train,y_train)
    pred = model.predict(x_test)
    return classification_report(y_test,pred),accuracy_score(y_test,pred)


print(knn_pred(x_train,y_train,x_test,y_test))