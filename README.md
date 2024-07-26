# Crypto

INDICATORS :

aberration, above, above_value, accbands, ad, adosc, adx,
alma, amat, ao, aobv, apo, bbands, 
below, below_value, bias, bop, brar, cci, cdl_pattern, cdl_z,
cfo, cg, chop, cksp, cmf, cmo, coppock, cross, cross_value, 
cti, decay, decreasing, dema, dm, donchian, dpo, ebsw, efi, 
entropy, eom, er, eri, fisher, fwma, ha, hilo, hl2, hlc3, hma, hwc,
hwma, ichimoku, increasing, inertia, jma, kama, kc, kdj, kst, kurtosis,
kvo, linreg, log_return, long_run,  mad, massi, mcgd, median, mfi, 
midpoint, midprice, mom, natr, nvi, obv, ohlc4, pdist, percent_return, pgo,
ppo, psar, psl, pvi, pvo, pvol, pvr, pvt, pwma, qqe, qstick, quantile, 
 rsx, rvgi, rvi, short_run, sinwma, skew, slope, squeeze,
squeeze_pro, ssf, stc, stdev, stoch, stochrsi, supertrend, swma, t3, td_seq,
tema, thermo, tos_stdevall, trima, trix, true_range, tsi, tsignals, ttm_trend, 
ui, uo, variance, vhf, vidya, vortex, vp, vwma, wcp, willr, wma, xsignals
zlma, zscore

-----------------------------------------------------------------------------------------------------
# roc, rsi, ema,sma, smi ----> MAMAD


# rma,macd, atr, vwap, aroon ----> ROOZI

-----------------------------------------------------------------------------------------------------
NEEDED :

# LogisticRegression , SVC , MLPClassifier , DecisionTreeClassifier , RandomForestClassifier , KNeighborsClassifier

-----------------------------------------------------------------------------------------------------
# GridSearchCv() , Corr() , ClassificationReport()
-----------------------------------------------------------------------------------------------------

# {'criterion': 'log_loss', 'min_samples_leaf': 4, 'min_samples_split': 3, 'splitter': 'random'} 0.5737012987012986 ----->DT


# {'algorithm': 'ball_tree', 'n_neighbors': 20, 'weights': 'distance'} 0.6378571428571429 ---->knn

# {'C': 7, 'dual': False, 'max_iter': 700, 'penalty': 'l1', 'solver': 'liblinear'} 0.7347402597402597 --->logistic

# {'activation': 'identity', 'hidden_layer_sizes': (100,), 'solver': 'adam'} 0.577077922077922 ---->mlp

# {'criterion': 'log_loss', 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 100} 0.5340259740259741 ---> rf