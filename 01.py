# Roozi Mokhi
import pandas_ta as ta
import pandas as pd

df = pd.DataFrame()
df = df.ta.ticker("BTC-USD", period="10y", interval="1d")
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

# print(rma)
print(df)
# print(help(ta.cci))
