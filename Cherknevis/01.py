# this page is for training and testing models

import numpy as np
from Models.MLPClassifier_Model import *
from Models.Knn_Model import *
from Models.RandomForestClassifier_Model import *
from Models.LogisticRegression_Model import *
from Models.SVC_Model import *
from Models.DecisionTreeClassifier_Model import *
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load data from CSV
data = pd.read_csv("../final_dataframe.csv", index_col="Date")

# Define features and target
X = data[['Open', 'High', 'Low', 'Close', "Tommorow_Open", 'Volume', "histogram", "ema7", "ema14", "ema21",
          'sma', "ema", 'squeeze', 'upper_band', 'lower_band', 'macd',
          'day_of_week']]["2014-10-20 00:00:00+00:00":"2024-07-31 00:00:00+00:00"]
y = data["Benefit"]["2014-10-20 00:00:00+00:00":"2024-07-31 00:00:00+00:00"]

# Split data
model = MLPClassifier(activation='logistic', hidden_layer_sizes=(100,), solver='adam', max_iter=1500)
model.fit(X, y)

pred = model.predict(X)
print(classification_report(y, pred))
# Stacking predictions
