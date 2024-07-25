# Roozi Mokhi
import pandas_ta as ta
import pandas as pd

df = pd.DataFrame()
df = df.ta.ticker("BTC-USD", period="1y", interval="1d")
df["cci"] = ta.cci(df["High"], df["Low"],  df["Close"], length = 7)
df["rma"] = ta.rma(df["Close"], length=7)
df["MACD_12_26_9"], df["MACDh_12_26_9"], df["MACDs_12_26_9"] = ta.macd(df["Close"])
df["atr"] = ta.atr(df["High"], df["Low"],  df["Close"], length = 7)
df["vwap"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
# print(rma)
print(df)
# print(help(ta.cci))
