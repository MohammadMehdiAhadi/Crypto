import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import pandas as pd
import pandas_ta as ta


def knn_model_maker():
    model = Knn(algorithm='kd_tree', n_neighbors=8, weights='distance')
    return model


def knn_fit(x_train, y_train):
    model = knn_model_maker()
    model.fit(x_train, y_train)
    return model


def final_knn_pred(x_train, y_train, x_test):
    model = knn_fit(x_train, y_train)
    pred = model.predict(x_test)
    return pred


