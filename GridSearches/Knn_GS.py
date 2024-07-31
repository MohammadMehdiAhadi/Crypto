from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas_ta as ta
import pandas as pd

# roc, rsi, ema,sma, smi ----> MAMAD

df = pd.DataFrame()
df = df.ta.ticker("BTC-USD", period="10y", interval="1d")

# Feature engineering
df["Tommorow_Close"] = df["Close"].shift(-1)
df["Tommorow_Open"] = df["Open"].shift(-1)
df["roc"] = ta.roc(df["Close"])
df["rsi"] = ta.rsi(df["Close"])
df["ema"] = ta.ema(df["Close"])
df["sma"] = ta.sma(df["Close"])
df["wcp"] = ta.wcp(df["High"], df["Low"], df["Close"])
sq = ta.squeeze(df["High"], df["Low"], df["Close"])
df["squeeze"] = sq["SQZ_20_2.0_20_1.5"]
df["cci"] = ta.cci(df["High"], df["Low"], df["Close"])
df["rma"] = ta.rma(df["Close"])
df["atr"] = ta.atr(df["High"], df["Low"], df["Close"])

# Add date and day of week
df["Date"] = df.index
df["day_of_week"] = df["Date"].dt.weekday

# Calculate benefit
df["Benefit"] = df["Tommorow_Close"] - df["Tommorow_Open"]
df["Benefit"] = df["Benefit"].apply(lambda x: 1 if x >= 0 else -1)

# Drop unnecessary columns
df.drop(["Dividends", "Stock Splits"], inplace=True, axis=1)

# Save to CSV
df.to_csv("final_dataframe.csv")

# Load data from CSV
data = pd.read_csv("final_dataframe.csv", index_col="Date")

# Define features and target
X = data[['Open', 'High', 'Low', 'Close', "Tommorow_Open",'Volume',
          'roc', 'rsi', 'ema', 'sma', 'wcp', 'squeeze', 'cci',
          'rma', 'atr', 'day_of_week']]["2014-10-07 00:00:00+00:00":"2024-07-30 00:00:00+00:00"]
y = data["Benefit"]["2014-10-07 00:00:00+00:00":"2024-07-30 00:00:00+00:00"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=17)

params = {"n_neighbors": [5,7,8,9,10],
          "weights": ['uniform', 'distance'],
          "algorithm": [ 'kd_tree','ball_tree', 'brute']}
model = GridSearchCV(estimator=Knn(),
                     param_grid=params,
                     scoring="accuracy")
model.fit(x_train, y_train)
model.predict(x_test)
print(model.best_params_)
print(model.best_score_)
