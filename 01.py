# Roozi Mokhi
import pandas_ta as ta
import pandas as pd

df = pd.DataFrame()
df = df.ta.ticker("BTC-USD", period="1y", interval="1d")
df["rma"] = ta.rma(df["close"], length=7, offset=None)
df["macd"] = ta.macd(df["close"], slow=0, fast=0, talib=None)
df["atr"] = ta.atr(df["hight"], df["low"], df["open"], df["close"], df["volume"])
df["vwamp"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"], df["date"])
df["aroon"] = ta.aroon(df["high"], df["low"], df["length"], df["volume"], df["open"], df["offset"])

print(help(ta.aroon))
