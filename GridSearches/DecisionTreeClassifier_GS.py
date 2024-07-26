from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
import pandas_ta as ta
import pandas as pd

df = pd.DataFrame()
df = df.ta.ticker("BTC-USD", period="1y", interval="1d")

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

# print(data.index)
# print(data.info)

X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
        'roc_7', 'rsi_7', 'ema_7', 'sma_7', 'squeeze', 'cci',
        'rma', 'atr']]["2023-08-13 00:00:00+00:00":]
y = df["Benefit"]["2023-08-13 00:00:00+00:00":]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=17)

params = {"criterion": ["gini", "entropy", "log_loss"],
          "splitter": ["best","random"],
          "min_samples_split": [2,3,4,5],
          "min_samples_leaf": [2,3,4,5]
          }
model = GridSearchCV(estimator=DecisionTreeClassifier(),
                     param_grid=params,
                     scoring="accuracy",
                     error_score='raise')
model.fit(x_train, y_train)
model.predict(x_test)
print(model.best_params_)
print(model.best_score_)
