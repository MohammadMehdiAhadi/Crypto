# Roozi Mokhi
import pandas_ta as ta
import pandas as pd

df = pd.DataFrame()
df = df.ta.ticker("BTC-USD", period="1y", interval="1d")
df["rma"] = ta.rma(df["close"], length=7)
df["macd"] = ta.macd(df["close"])
df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=7)
df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
# df["aroon"] = ta.aroon(df["high"], df["low"], length=7)

print(help(ta.rma))
