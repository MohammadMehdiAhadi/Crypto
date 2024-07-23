import pandas_ta as ta
import pandas as pd

df = pd.DataFrame()
df = df.ta.ticker("BTC-USD", period="1y", interval="1d")


print(help(df.ta.indicators()))
# print(help(ta.cci))
# df["cci"] = ta.cci(df["High"], df["Low"], df["Close"], length=7)
# print(df)
