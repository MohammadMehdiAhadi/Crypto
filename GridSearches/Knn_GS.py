from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
import pandas_ta as ta
import pandas as pd

# roc, rsi, ema,sma, smi ----> MAMAD

df = pd.DataFrame()
df = df.ta.ticker("BTC-USD", period="1y", interval="1d")

df["Tommorw_Close"] = df["Close"].shift(-1)

df["roc_7"] = ta.roc(df["Close"], length=7)

df["rsi_7"] = ta.rsi(df["Close"], length=7)

df["ema_7"] = ta.ema(df["Close"], length=7)

df["sma_7"] = ta.sma(df["Close"], length=7)

sq = ta.squeeze(df["High"], df["Low"], df["Close"])
df["squeeze"] = sq["SQZ_20_2.0_20_1.5"]



df["cci"] = ta.cci(df["High"], df["Low"],  df["Close"], length = 7)
df["rma"] = ta.rma(df["Close"], length=7)
# df["MACD_12_26_9"], df["MACDh_12_26_9"], df["MACDs_12_26_9"] = ta.macd(df["Close"])
df["atr"] = ta.atr(df["High"], df["Low"],  df["Close"], length = 7)

df["Benefit"] = df["Tommorw_Close"] - df["Open"]
df["Benefit"] = df["Benefit"].apply(lambda x: 1 if x >= 0 else -1)
# print(df.columns)


X=df[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
       'Tommorw_Close', 'roc_7', 'rsi_7', 'ema_7', 'sma_7', 'squeeze', 'cci',
       'rma',  'atr']].dropna()
y = df["Benefit"].dropna()

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3,shuffle=False,random_state=17)
params = {"n_neighbors": [5, 10, 15],
          "weights": ['uniform', 'distance'],
          "algorithm": ['ball_tree', 'kd_tree', 'brute']}
model = GridSearchCV(estimator=Knn(),
                     param_grid=params,
                     scoring="accuracy")
model.fit(x_train, y_train)
model.predict(x_test)
print(model.best_params_)
print(model.best_score_)
