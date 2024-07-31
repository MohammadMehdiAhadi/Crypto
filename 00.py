import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# # Assuming you have already installed pandas_ta
# df = pd.DataFrame()
# df = df.ta.ticker("BTC-USD", period="1y", interval="1d")

# print(help(ta.fisher))


# df["FISHERT_7_1"], df["FISHERTs_7_1"] = ta.fisher(df["High"], df["Low"], length=7).values(0)
# print(df["FISHERT_7_1"])
# print(df["FISHERTs_7_1"])
# df["vwap"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"]).values
# print(df["vwap"])
# df["Date"] = df.index
# df["Weekday"] = df["Date"].dt.weekday
# print(df["Weekday"])
#
#


# df = pd.read_csv('final_dataframe.csv',index_col="Date")
# data = df["2024-06-06 00:00:00+00:00"]
# print(data)
# fig = go.Figure(data=[go.Candlestick(x=df['Date'],
#                 open=df['Open'], high=df['High'],
#                 low=df['Low'], close=df['Close'])
#                      ])
#
# fig.add_trace(go.Scatter(x=df['Date'],
#                          y=[df['Low'].min()]*len(df['Date']),
#                          mode="lines",
#                          line=dict(color='red', width=2)))
#
# fig.add_trace(go.Scatter(x=df['Date'],
#                          y=[df['High'].min()]*len(df['Date']),
#                          mode="lines",
#                          line=dict(color='blue', width=2)))
# last = 0
# for i in range(len(df['Date'])):
#     temp = df['Low'].values[0]
#     if temp >= last:
#         fig.add_annotation(x=df['Date'], y=temp+3, text="▲", showarrow=False, font=dict(size=16, color='LightSeaGreen'))
#         last = temp
#     else:
#         fig.add_annotation(x=df['Date'], y=temp-3, text="▲", showarrow=False, font=dict(size=16, color='red'))
#
# fig.update_layout(xaxis_rangeslider_visible=False)
# fig.show(n_jobs = -1)
#
#


# print(df["Date"])


# df["Tommorw_Close"] = df["Close"].shift(-1)
# df = pd.DataFrame()
# df = df.ta.ticker("BTC-USD", period="10y", interval="1d")
# # print(help(ta.aberration))
# df["aberration"]= ta.aberration(df["High"],df["Low"], df["Close"]).values

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
# print(help(ta.wcp))
# df["wcp"] = ta.wcp(df["High"],df["Low"],df["Close"])
# print(df.columns)
# print(ww)
#

# df["SMI_5_20_5"],df["SMIs_5_20_5"],df["SMIo_5_20_5"]= ta.smi(df["Close"])
# print(df["SMI_5_20_5"])
# print(df)




import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import your models
from Models.MLPClassifier_Model import *
from Models.Knn_Model import *
from Models.RandomForestClassifier_Model import *
from Models.LogisticRegression_Model import *
from Models.SVC_Model import *
from Models.DecisionTreeClassifier_Model import *

import seaborn as sns
from sklearn.metrics import confusion_matrix



# Load data
df = pd.DataFrame()
df = df.ta.ticker("BTC-USD", period="10y", interval="1d")

# Feature engineering
df["Tommorw_Close"] = df["Close"].shift(-1)
df["roc_7"] = ta.roc(df["Close"], length=7)
df["rsi_7"] = ta.rsi(df["Close"], length=7)
df["ema_7"] = ta.ema(df["Close"], length=7)
df["sma_7"] = ta.sma(df["Close"], length=7)
df["wcp"] = ta.wcp(df["High"], df["Low"], df["Close"])
sq = ta.squeeze(df["High"], df["Low"], df["Close"])
df["squeeze"] = sq["SQZ_20_2.0_20_1.5"]
df["cci"] = ta.cci(df["High"], df["Low"], df["Close"], length=7)
df["rma"] = ta.rma(df["Close"], length=7)
df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=7)


# # Add date and day of week
df["Date"] = df.index
df["day_of_week"] = df["Date"].dt.weekday
df["aberration"]= (ta.aberration(df["High"],df["Low"], df["Close"]).values).reshape(-1,1)

# Calculate benefit
df["Benefit"] = df["Tommorw_Close"] - df["Open"]
df["Benefit"] = df["Benefit"].apply(lambda x: 1 if x >= 0 else -1)

# Drop unnecessary columns
df.drop(["Dividends", "Stock Splits"], inplace=True, axis=1)

print(df.corr())