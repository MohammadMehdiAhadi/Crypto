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
# df["Tommorow_Close"] = df["Close"].shift(-1)
# df["Tommorow_Open"] = df["Open"].shift(-1)
# df["roc"] = ta.roc(df["Close"])
# df["rsi"] = ta.rsi(df["Close"])
# df["ema"] = ta.ema(df["Close"])
# df["sma"] = ta.sma(df["Close"])
# df["wcp"] = ta.wcp(df["High"], df["Low"], df["Close"])
# sq = ta.squeeze(df["High"], df["Low"], df["Close"])
# df["squeeze"] = sq["SQZ_20_2.0_20_1.5"]
# df["cci"] = ta.cci(df["High"], df["Low"], df["Close"])
# df["rma"] = ta.rma(df["Close"])
# df["atr"] = ta.atr(df["High"], df["Low"], df["Close"])
#
# df['std_dev'] = ta.stdev(df['Close'])
# df['ema12'] = df['Close'].ewm(span=12).mean()
#
# # Calculate the 26-day EMA (long-term)
# df['ema26'] = df['Close'].ewm(span=26).mean()
#
# # Calculate the MACD line
# df['macd'] = df['ema12'] - df['ema26']
#
# # Calculate the 9-day EMA of the MACD (signal line)
# df['signal'] = df['macd'].ewm(span=9).mean()
#
# # Calculate the histogram
# df['histogram'] = df['macd'] - df['signal']
# # Calculate Bollinger Bands
# df['upper_band'] = df['sma'] + (2 * df['std_dev'])
# df['lower_band'] = df['sma'] - (2 * df['std_dev'])
# # Add date and day of week
# df["Date"] = df.index
# df["day_of_week"] = df["Date"].dt.weekday
#
# # Calculate benefit
# df["Benefit"] = df["Tommorow_Close"] - df["Tommorow_Open"]
# df["Benefit"] = df["Benefit"].apply(lambda x: 1 if x >= 0 else 0)
#
# # Drop unnecessary columns
# df.drop(["Dividends", "Stock Splits"], inplace=True, axis=1)
#
# # Save to CSV
# df.to_csv("final_dataframe.csv")

# Load data from CSV
data = pd.read_csv("final_dataframe.csv", index_col="Date")

# Define features and target
X = data[['Open', 'High', 'Low', 'Close', "Tommorow_Open", 'Volume', "histogram",
          'sma', "ema", 'squeeze', 'upper_band', 'lower_band', 'macd',
          'day_of_week']]["2014-10-20 00:00:00+00:00":"2024-07-30 00:00:00+00:00"]
y = data["Benefit"]["2014-10-20 00:00:00+00:00":"2024-07-30 00:00:00+00:00"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=17)

params = {"C": [3.0, 2.0],
          "kernel": ['poly', 'rbf', 'sigmoid', 'precomputed']
          }
model = GridSearchCV(estimator=SVC(tol=1e-5, gamma='scale'),
                     param_grid=params,
                     scoring="accuracy",
                     )
model.fit(x_train, y_train)
model.predict(x_test)
print(model.best_params_)
print(model.best_score_)

