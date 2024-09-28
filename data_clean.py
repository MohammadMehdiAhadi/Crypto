import pandas as pd
from sklearn.model_selection import train_test_split


print('reading data from the csv file')
print('please wait . . .')
data = pd.read_csv("final_dataframe.csv", index_col="Date")

if not data.empty:
    n = len(data)

    # Define features and target
    X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Next_Hour_Open',
              'ema6', 'ema12', 'ema24', 'ema48', 'ema72', 'histogram',
              'sma', 'squeeze', 'upper_band', 'lower_band', 'macd',
              'day_of_week']].iloc[71:n - 1]
    y = data["Benefit"].iloc[71:n - 1]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.0005, shuffle=False)
    print('Done ')

else:
    print("Couldn't Load The Data")
    print("Try Again")