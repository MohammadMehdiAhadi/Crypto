import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from Models.MLPClassifier_Model import *
from Models.Knn_Model import *
from Models.RandomForestClassifier_Model import *
from Models.LogisticRegression_Model import *
from Models.SVC_Model import *
from Models.DecisionTreeClassifier_Model import *
import seaborn as sns
from sklearn.metrics import confusion_matrix
from Models.Final_Knn_Model import  *




# Load data from CSV
data = pd.read_csv("final_dataframe.csv", index_col="Date")
n = len(data)

# Define features and target
X = data[['Open', 'High', 'Low', 'Close', "Tommorow_Open", 'Volume', "histogram","ema7","ema14","ema21",
          'sma', "ema", 'squeeze', 'upper_band', 'lower_band', 'macd',
          'day_of_week']].iloc[30:n-1]
y = data["Benefit"].iloc[30:n-1]

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.005, shuffle=False, random_state=17)

# Stacking predictions
predictions_stacking = np.vstack([#logistic_pred(x_train, y_train, x_test),
                                  mlp_pred(x_train, y_train, x_test),
                                  knn_pred(x_train, y_train, x_test),
                                  svm_pred(x_train, y_train, x_test),
                                  dt_pred(x_train, y_train, x_test),
                                  rf_pred(x_train, y_train, x_test)]).T

# Meta model prediction
predictions_final = knn_pred(predictions_stacking, y_test, predictions_stacking)
accuracy = np.mean(predictions_final == y_test)
print("دقت مدل Stacking:", accuracy * 100)

# Classification reports
print(classification_report(y_test, mlp_pred(x_train, y_train, x_test)))
# print(classification_report(y_test, logistic_pred(x_train, y_train, x_test)))
print(classification_report(y_test, knn_pred(x_train, y_train, x_test)))
print(classification_report(y_test, svm_pred(x_train, y_train, x_test)))
print(classification_report(y_test, dt_pred(x_train, y_train, x_test)))
print(classification_report(y_test, rf_pred(x_train, y_train, x_test)))
print(classification_report(y_test, predictions_final))

# Create a heatmap for the confusion matrix

conf_matrix = confusion_matrix(y_test, predictions_final)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("final_predict.jpg")
plt.show()

