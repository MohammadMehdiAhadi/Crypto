try:
    import pandas as pd
    import pandas_ta as ta
    import yfinance as yf

    # Load data
    print("Downloading Data ...")
    print("Please Wait")

    # df = pd.DataFrame()
    # df = df.ta.ticker("BTC-USD", period="10y", interval="1d")
    df = yf.download("BTC-USD", period="730d", interval="1h")
    df.index = pd.to_datetime(df.index)

    if not df.empty:

        # Feature engineering
        df["Next_Hour_Close"] = df["Close"].shift(-1)
        df["Next_Hour_Open"] = df["Open"].shift(-1)
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
        df['std_dev'] = ta.stdev(df['Close'])
        df['ema12'] = df['Close'].ewm(span=12).mean()
        df['ema7'] = df['Close'].ewm(span=7).mean()
        df['ema14'] = df['Close'].ewm(span=14).mean()
        df['ema21'] = df['Close'].ewm(span=21).mean()
        df['ema30'] = df['Close'].ewm(span=30).mean()
        df["std_7"] = df["Close"].rolling(window=7).std()
        df["std_14"] = df["Close"].rolling(window=14).std()
        df["std_21"] = df["Close"].rolling(window=21).std()
        df["std_30"] = df["Close"].rolling(window=30).std()
        df["mean_7"] = df["Close"].rolling(window=7).mean()
        df["mean_14"] = df["Close"].rolling(window=14).mean()
        df["mean_21"] = df["Close"].rolling(window=21).mean()
        df["mean_30"] = df["Close"].rolling(window=30).mean()
        df["stdp7"] = df["mean_7"] + df["std_7"]
        df["stdp7"] = df["mean_7"] - df["std_7"]
        df["stdp14"] = df["mean_14"] + df["std_14"]
        df["stdn14"] = df["mean_14"] - df["std_14"]
        df["stdp21"] = df["mean_21"] + df["std_21"]
        df["stdn21"] = df["mean_21"] - df["std_21"]
        df["stdp30"] = df["mean_30"] + df["std_30"]
        df["stdn30"] = df["mean_30"] - df["std_30"]
        df["min"] = df["Close"].rolling(window=30).min()
        df["max"] = df["Close"].rolling(window=30).max()
        df['ema26'] = df['Close'].ewm(span=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9).mean()
        df['histogram'] = df['macd'] - df['signal']
        df['upper_band'] = df['sma'] + (2 * df['std_dev'])
        df['lower_band'] = df['sma'] - (2 * df['std_dev'])

        # Add date and day of week
        df["Date"] = df.index
        df["day_of_week"] = df["Date"].dt.weekday

        # Calculate benefit
        df["Benefit"] = df["Next_Hour_Close"] - df["Next_Hour_Open"]
        df["Benefit"] = df["Benefit"].apply(lambda x: 1 if x >= 0 else 0)

        # # Drop unnecessary columns
        # df.drop(["Dividends", "Stock Splits"], inplace=True, axis=1)

        # Save to CSV
        df.to_csv("final_dataframe.csv")

        print("File Is Ready To Use")
        print("Now Go And Run The Main.py File")
    else:
        print("Something Went Wrong")
        print("Check Your Connection please!")
except:
    pass
