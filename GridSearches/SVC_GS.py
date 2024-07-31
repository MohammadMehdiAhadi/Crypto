from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV

import pandas_ta as ta
import pandas as pd

# df = pd.DataFrame()
# df = df.ta.ticker("BTC-USD", period="10y", interval="1d")
#
# # Feature engineering
# df["Tommorw_Close"] = df["Close"].shift(-1)
# df["roc_7"] = ta.roc(df["Close"], length=7)
# df["rsi_7"] = ta.rsi(df["Close"], length=7)
# df["ema_7"] = ta.ema(df["Close"], length=7)
# df["sma_7"] = ta.sma(df["Close"], length=7)
# df["wcp"] = ta.wcp(df["High"], df["Low"], df["Close"])
# sq = ta.squeeze(df["High"], df["Low"], df["Close"])
# df["squeeze"] = sq["SQZ_20_2.0_20_1.5"]
# df["cci"] = ta.cci(df["High"], df["Low"], df["Close"], length=7)
# df["rma"] = ta.rma(df["Close"], length=7)
# df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=7)
#
# # Add date and day of week
# df["Date"] = df.index
# df["day_of_week"] = df["Date"].dt.weekday
#
# # Calculate benefit
# df["Benefit"] = df["Tommorw_Close"] - df["Open"]
# df["Benefit"] = df["Benefit"].apply(lambda x: 1 if x >= 0 else -1)
#
# # Drop unnecessary columns
# df.drop(["Dividends", "Stock Splits"], inplace=True, axis=1)
#
# # Save to CSV
# df.to_csv("final_dataframe.csv")

# Load data from CSV
data = pd.read_csv("final_dataframe.csv", index_col="Date")

# Define features and target
X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'day_of_week',
        'roc_7', 'rsi_7', 'ema_7', 'sma_7', 'wcp', 'squeeze', 'cci',
        'rma', 'atr']]["2014-10-07 00:00:00+00:00":]
y = data["Benefit"]["2014-10-07 00:00:00+00:00":]

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=17)

params = {"C": [3.0,2.0,5.0],
          "kernel": ["sigmoid","rbf"]
          }
model = GridSearchCV(estimator=SVC(tol=1e-5,gamma='auto'),
                     param_grid=params,
                     scoring="accuracy",
                     )
model.fit(x_train, y_train)
model.predict(x_test)
print(model.best_params_)
print(model.best_score_)
# 'linear', 'poly','rbf' ,'sigmoid'
# {'C': 3.0, 'kernel': 'sigmoid'}    0.549163580641042 gamma = auto
# {'C': 3.0, 'kernel': 'rbf'}        0.549163580641042 gamma = auto