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
import pandas_ta as ta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.DataFrame()
df = df.ta.ticker("BTC-USD", period="10y", interval="1d")

df["Tommorw_Close"] = df["Close"].shift(-1)

df["roc_7"] = ta.roc(df["Close"], length=7)

df["rsi_7"] = ta.rsi(df["Close"], length=7)

df["ema_7"] = ta.ema(df["Close"], length=7)

df["sma_7"] = ta.sma(df["Close"], length=7)

sq = ta.squeeze(df["High"], df["Low"], df["Close"])

df["squeeze"] = sq["SQZ_20_2.0_20_1.5"]

df["cci"] = ta.cci(df["High"], df["Low"], df["Close"], length=7)

df["rma"] = ta.rma(df["Close"], length=7)

df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=7)

df["Benefit"] = df["Tommorw_Close"] - df["Open"]

df["Benefit"] = df["Benefit"].apply(lambda x: 1 if x >= 0 else -1)

df.to_csv("dataframe_knn.csv")

data = pd.read_csv("dataframe_knn.csv", index_col="Date")

X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
        'roc_7', 'rsi_7', 'ema_7', 'sma_7', 'squeeze', 'cci',
        'rma', 'atr']]["2014-10-04 00:00:00+00:00":]
y = df["Benefit"]["2014-10-04 00:00:00+00:00":]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=17)

predictions_stacking = np.vstack([mlp_pred(x_train, y_train, x_test),
                                  logistic_pred(x_train, y_train, x_test),
                                  knn_pred(x_train, y_train, x_test),
                                  rf_pred(x_train, y_train, x_test)]
                                 ).T

model_meta = LogisticRegression()
model_meta.fit(predictions_stacking, y_test)

predictions_final = model_meta.predict(predictions_stacking)

accuracy = np.mean(predictions_final == y_test)
print("دقت مدل Stacking:", accuracy)

print(classification_report(y_test, mlp_pred(x_train, y_train, x_test)))
print(classification_report(y_test, logistic_pred(x_train, y_train, x_test)))
print(classification_report(y_test, knn_pred(x_train, y_train, x_test)))
print(classification_report(y_test, rf_pred(x_train, y_train, x_test)))
print(classification_report(y_test, predictions_final))
