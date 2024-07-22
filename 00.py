import pandas_ta as ta
import pandas as pd

df = pd.DataFrame()
df = df.ta.ticker("BTC-USD", period="1y", interval="1d")


print(help(df.ta.indicators()))
# print(help(ta.cci))
# df["cci"] = ta.cci(df["High"], df["Low"], df["Close"], length=7)
# print(df)
'''''''''
aberration, above, above_value, accbands, ad, adosc, adx,
alma, amat, ao, aobv, apo, aroon, atr, bbands, 
below, below_value, bias, bop, brar, cci, cdl_pattern, cdl_z,
cfo, cg, chop, cksp, cmf, cmo, coppock, cross, cross_value, 
cti, decay, decreasing, dema, dm, donchian, dpo, ebsw, efi, ema, 
entropy, eom, er, eri, fisher, fwma, ha, hilo, hl2, hlc3, hma, hwc,
hwma, ichimoku, increasing, inertia, jma, kama, kc, kdj, kst, kurtosis,
kvo, linreg, log_return, long_run, macd, mad, massi, mcgd, median, mfi, 
midpoint, midprice, mom, natr, nvi, obv, ohlc4, pdist, percent_return, pgo,
ppo, psar, psl, pvi, pvo, pvol, pvr, pvt, pwma, qqe, qstick, quantile, rma,
roc, rsi, rsx, rvgi, rvi, short_run, sinwma, skew, slope, sma, smi, squeeze,
squeeze_pro, ssf, stc, stdev, stoch, stochrsi, supertrend, swma, t3, td_seq,
tema, thermo, tos_stdevall, trima, trix, true_range, tsi, tsignals, ttm_trend, 
ui, uo, variance, vhf, vidya, vortex, vp, vwap, vwma, wcp, willr, wma, xsignals
zlma, zscore

'''''''''

