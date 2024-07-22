import pandas as pd
import yfinance as yf
import ta

# Define the stock symbol (BTC-USD)
symbol = "BTC-USD"

# Get historical data
data = yf.download("BTC-USD", start="2020-01-01", end="2024-07-20")

# Calculate RSI
data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()

# Calculate moving averages (SMA and EMA)
data["SMA20"] = ta.trend.SMAIndicator(data["Close"], window=20).sma_indicator()
data["EMA50"] = ta.trend.EMAIndicator(data["Close"], window=50).ema_indicator()

# Calculate Bollinger Bands
data["BollingerUpper"], data["BollingerLower"] = ta.volatility.BollingerBands(data["Close"]).bollinger_hband_indicator(), ta.volatility.BollingerBands(data["Close"]).bollinger_lband_indicator()

# Calculate MACD
data["MACD"], data["SignalLine"], _ = ta.trend.MACD(data["Close"])

# Other indicators (Fibonacci, Ichimoku, Volume Profile) can be added similarly

# Print the first few rows of the data
print(data[["Close", "RSI", "SMA20", "EMA50", "BollingerUpper", "BollingerLower", "MACD", "SignalLine"]].head())
