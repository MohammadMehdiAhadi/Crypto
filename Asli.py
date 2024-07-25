from sklearn.datasets import load_digits
from Models.MLPClassifier_Model import *
from Models.Knn_Model import *
from Models.RandomForestClassifier_Model import *
from Models.LogisticRegression_Model import *
from Models.SVC_Model import *
from sklearn.model_selection import train_test_split
from Models.DecisionTreeClassifier_Model import *
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression





X,y = load_digits(return_X_y=True)
X = X.astype(np.float32) / 255.0

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


predictions_stacking = np.vstack([dt_pred(x_train, y_train,x_test),
                                  mlp_pred(x_train, y_train,x_test),
                                  rf_pred(x_train, y_train,x_test)]).T

model_meta = SVC()
model_meta.fit(predictions_stacking, y_test)


predictions_final = model_meta.predict(predictions_stacking)


accuracy = np.mean(predictions_final == y_test)
print("دقت مدل Stacking:", accuracy)

print(classification_report(y_test, dt_pred(x_train, y_train,x_test)))
print(classification_report(y_test, mlp_pred(x_train, y_train,x_test)))
print(classification_report(y_test, rf_pred(x_train, y_train,x_test)))
print(classification_report(y_test, predictions_final))


