import numpy as np
from Models.MLPClassifier_Model import *
from Models.Knn_Model import *
from Models.RandomForestClassifier_Model import *
from Models.LogisticRegression_Model import *
from Models.SVC_Model import *
from Models.DecisionTreeClassifier_Model import *
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load data
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

# Stacking predictions
predictions_stacking = np.vstack([mlp_pred(x_train, y_train, x_test),
                                  logistic_pred(x_train, y_train, x_test),
                                  knn_pred(x_train, y_train, x_test),
                                  svm_pred(x_train, y_train, x_test),
                                  dt_pred(x_train, y_train, x_test),
                                  rf_pred(x_train, y_train, x_test)]).T

# Meta model prediction
# predictions_final = dt_pred(predictions_stacking, y_test, predictions_stacking)
model = MLPClassifier(hidden_layer_sizes=(200,))
model.fit(predictions_stacking, y_test)
predictions_final = model.predict(predictions_stacking)
accuracy = np.mean(predictions_final == y_test)
print("دقت مدل Stacking:", accuracy * 100)

# Classification reports
print(classification_report(y_test, mlp_pred(x_train, y_train, x_test)))
print(classification_report(y_test, logistic_pred(x_train, y_train, x_test)))
print(classification_report(y_test, knn_pred(x_train, y_train, x_test)))
print(classification_report(y_test, svm_pred(x_train, y_train, x_test)))
print(classification_report(y_test, dt_pred(x_train, y_train, x_test)))
print(classification_report(y_test, rf_pred(x_train, y_train, x_test)))
print(classification_report(y_test, predictions_final))

# Assuming 'predictions_final' contains your model predictions
conf_matrix = confusion_matrix(y_test, predictions_final)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# data_new = pd.read_csv("final_dataframe.csv")
# predictions_df = pd.DataFrame({
#     "Date": data_new["Date"].values[-len(predictions_final):],  # Align with predictions
#     "Benefit": df["Benefit"].values[-len(predictions_final):],
#     "MLP_Prediction": mlp_pred(x_train, y_train, x_test),
#     "Logistic_Prediction": logistic_pred(x_train, y_train, x_test),
#     "KNN_Prediction": knn_pred(x_train, y_train, x_test),
#     # Add other model predictions here...
# })
#
#
# # Assuming you have already defined predictions_df
#
# # Bar plot for the dataset
# plt.figure(figsize=(10, 6))
# plt.bar(predictions_df["Date"], predictions_df["Benefit"], color='maroon', width=0.4)
# plt.xlabel("Date")
# plt.ylabel("Benefit")
# plt.title("BTC-USD Benefit and Model Predictions (Bar Chart)")
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(10, 6))
# plt.scatter(predictions_df["Date"], predictions_df["Benefit"], label="BTC Benefit", marker='o')
# plt.scatter(predictions_df["Date"], predictions_df["MLP_Prediction"], label="MLP", marker='x')
# plt.scatter(predictions_df["Date"], predictions_df["Logistic_Prediction"], label="Logistic Regression", marker='s')
# plt.scatter(predictions_df["Date"], predictions_df["KNN_Prediction"], label="KNN", marker='^')
# # Add other model predictions here...
#
# plt.xlabel("Date")
# plt.ylabel("Benefit")
# plt.title("BTC-USD Benefit and Model Predictions (Scatter Plot)")
# plt.legend()
# plt.tight_layout()
# plt.show()
# Plotting historical BTC benefit and model predictions
# plt.figure(figsize=(10, 6))
# plt.plot(predictions_df["Date"], predictions_df["Benefit"], label="BTC Benefit")
# plt.plot(predictions_df["Date"], predictions_df["MLP_Prediction"], label="MLP")
# plt.plot(predictions_df["Date"], predictions_df["Logistic_Prediction"], label="Logistic Regression")
# plt.plot(predictions_df["Date"], predictions_df["KNN_Prediction"], label="KNN")
# # Add other model predictions here...
#
# plt.xlabel("Date")
# plt.ylabel("Benefit")
# plt.title("BTC-USD Benefit and Model Predictions")
# plt.legend()
# plt.tight_layout()
# plt.show()


# df = pd.DataFrame()
# df = df.ta.ticker("BTC-USD", period="10y", interval="1d")
#
# df["Tommorw_Close"] = df["Close"].shift(-1)
#
# df["roc_7"] = ta.roc(df["Close"], length=7)
#
# df["rsi_7"] = ta.rsi(df["Close"], length=7)
#
# df["ema_7"] = ta.ema(df["Close"], length=7)
#
# df["sma_7"] = ta.sma(df["Close"], length=7)
#
# df["wcp"] = ta.wcp(df["High"], df["Low"], df["Close"])
#
# sq = ta.squeeze(df["High"], df["Low"], df["Close"])
#
# df["squeeze"] = sq["SQZ_20_2.0_20_1.5"]
#
# df["cci"] = ta.cci(df["High"], df["Low"], df["Close"], length=7)
#
# df["rma"] = ta.rma(df["Close"], length=7)
#
# df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=7)
#
# df["Date"] = df.index
#
# df["day_of_week"] = df["Date"].dt.weekday
#
# df["Benefit"] = df["Tommorw_Close"] - df["Open"]
#
# df["Benefit"] = df["Benefit"].apply(lambda x: 1 if x >= 0 else -1)
# df.drop(["Dividends", "Stock Splits"], inplace=True, axis=1)
#
# df.to_csv("final_dataframe.csv")
#
# data = pd.read_csv("final_dataframe.csv", index_col="Date")
#
# X = df[['Open', 'High', 'Low', 'Close', 'Volume','day-of-week',
#         'roc_7', 'rsi_7', 'ema_7', 'sma_7',"wcp", 'squeeze', 'cci',
#         'rma', 'atr']]["2014-10-06 00:00:00+00:00":]
# y = df["Benefit"]["2014-10-06 00:00:00+00:00":]
#
# x_train, x_test, y_train, y_test = train_test_split(X,
#                                                     y,
#                                                     test_size=0.2,
#                                                     shuffle=False,
#                                                     random_state=17
#                                                     )
#
# predictions_stacking = np.vstack([mlp_pred(x_train, y_train, x_test),
#                                   logistic_pred(x_train, y_train, x_test),
#                                   knn_pred(x_train, y_train, x_test),
#                                   svm_pred(x_train, y_train, x_test),
#                                   rf_pred(x_train, y_train, x_test)]
#                                  ).T
#

# predictions_final = logistic_pred(predictions_stacking, y_test, predictions_stacking)
# accuracy = np.mean(predictions_final == y_test)
# print("دقت مدل Stacking:", accuracy)
#
# print(classification_report(y_test, mlp_pred(x_train, y_train, x_test)))
# print(classification_report(y_test, logistic_pred(x_train, y_train, x_test)))
# print(classification_report(y_test, knn_pred(x_train, y_train, x_test)))
# print(classification_report(y_test, svm_pred(x_train, y_train, x_test)))
# print(classification_report(y_test, rf_pred(x_train, y_train, x_test)))
# print(classification_report(y_test, predictions_final))
#


# data_new = pd.read_csv("final_dataframe.csv")
#
#
# fig, ax1 = plt.subplots(1, 1)  # Corrected to unpack two axes
#
# # Plotting close price
# ax1.plot(
#     data_new["Date"].values,  # Corrected column name to lowercase
#     data_new["Close"].values,  # Corrected column name to lowercase
#     label="BTC Price",
# )
# # Plotting rolling mean
# ax1.plot(
#     data_new["Date"].values,  # Corrected column name to lowercase
#     mlp_pred(x_train, y_train, x_test),  # Corrected column name to lowercase
#     label="MLP",
# )
# ax1.set_ylabel("Close Price")  # Corrected spelling
# ax1.set_title("BTC-USD Price and Rolling Mean")
# ax1.legend()
#
# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()
