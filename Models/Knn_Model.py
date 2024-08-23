import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import pandas as pd
import pandas_ta as ta


def knn_model_maker():
    model = Knn(algorithm='kd_tree', n_neighbors=45, weights='uniform')
    return model


def knn_fit(x_train, y_train):
    model = knn_model_maker()
    model.fit(x_train, y_train)
    return model


def knn_pred(x_train, y_train, x_test):
    model = knn_fit(x_train, y_train)
    pred = model.predict(x_test)
    return pred




# Fetching data
# df = pd.DataFrame()
# df = df.ta.ticker("BTC-USD", period="1y", interval="1d")
#
# df.to_csv("final_dataframe.csv")
#
# data = pd.read_csv("final_dataframe.csv")
#
# # Plotting
# fig, ax1 = plt.subplots(1, 1)  # Corrected to unpack two axes
#
# # Plotting close price
# ax1.plot(
#     data["Date"].values,  # Corrected column name to lowercase
#     data["Close"].values,  # Corrected column name to lowercase
#     label="Close Price",
# )
# # Plotting rolling mean
# ax1.plot(
#     data["Date"].values,  # Corrected column name to lowercase
#     data["Close"].rolling(window=50).mean(),  # Corrected column name to lowercase
#     label="50-Day Rolling Mean",
# )
# ax1.set_ylabel("Close Price")  # Corrected spelling
# ax1.set_title("BTC-USD Price and Rolling Mean")
# ax1.legend()
#
# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()
