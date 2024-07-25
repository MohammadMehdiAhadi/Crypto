from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits



def mlp_model_maker ():
    model = MLPClassifier(hidden_layer_sizes=(100,50,10,2,))
    return model


def mlp_fit(x_train,y_train):
    model = mlp_model_maker()
    model.fit(x_train,y_train)
    return model

def mlp_pred(x_test,x_train,y_train):
    model = mlp_fit(x_train,y_train)
    pred = model.predict(x_test)
    return pred
    # return (classification_report(y_test,pred),
    #         accuracy_score(y_test,pred)
    #         )
# print(mlp_pred(x_train,y_train,x_test,y_test))