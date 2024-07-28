import pandas_ta as ta
import pandas as pd

# roc, rsi, ema,sma, smi ----> MAMAD

df = pd.DataFrame()
df = df.ta.ticker("BTC-USD", period="1y", interval="1d")

# df["Tommorw_Close"] = df["Close"].shift(-1)
#
#
#
#
#
#
# # print(help(ta.roc))
# df["roc_7"] = ta.roc( df["Close"], length=7)
#
# # print(help(ta.rsi))
# df["rsi_7"] = ta.rsi( df["Close"], length=7)
#
# # print(help(ta.ema))
# df["ema_7"] = ta.ema( df["Close"], length=7)
#
# # print(help(ta.sma))
# df["sma_7"] = ta.sma( df["Close"], length=7)
#
# # print(help(ta.squeeze))
# sq = ta.squeeze(df["High"], df["Low"], df["Close"])
# # print(sq)
# df["squeeze"] = sq["SQZ_20_2.0_20_1.5"]
#
# 'Dividends', 'Stock Splits'
# df.drop(["Dividends","Stock Splits"],inplace = True,axis =1)
print(help(ta.wcp))
df["wcp"] = ta.wcp(df["High"],df["Low"],df["Close"])
# print(df.columns)
# print(ww)
#

# df["SMI_5_20_5"],df["SMIs_5_20_5"],df["SMIo_5_20_5"]= ta.smi(df["Close"])
# print(df["SMI_5_20_5"])
# print(df)
