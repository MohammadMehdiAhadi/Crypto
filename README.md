# Crypto

INDICATORS :

aberration, above, above_value, accbands, ad, adosc, adx,
alma, amat, ao, aobv, apo, bbands,
below, below_value, bias, bop, brar, cci, cdl_pattern, cdl_z,
cfo, cg, chop, cksp, cmf, cmo, coppock, cross, cross_value,
cti, decay, decreasing, dema, dm, donchian, dpo, ebsw, efi,
entropy, eom, er, eri, fisher, fwma, ha, hilo, hl2, hlc3, hma, hwc,
hwma, ichimoku, increasing, inertia, jma, kama, kc, kdj, kst, kurtosis,
kvo, linreg, log_return, long_run, mad, massi, mcgd, median, mfi,
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

# {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 3, 'splitter': 'best'} 0.4763493074351218 ----->DT

# {'algorithm': 'kd_tree', 'n_neighbors': 8, 'weights': 'distance'} 0.5029570254614038 ---->knn

# {'C': 5, 'dual': False, 'max_iter': 300, 'penalty': 'l1', 'solver': 'saga'} 0.522566225368327 --->logistic

# {'activation': 'logistic', 'hidden_layer_sizes': (100,), 'solver': 'adam' , 'max_iter': 1500} 0.5369269959462605 ---->mlp

# {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 400} 0.4644477239048167 ---> rf

# {'C': 3.0, 'kernel': "rbf", 'tol': 1e-5, 'gamma': 'scale'}  0.5259269979462605  ---> svm
-----------------------------------------------------------------------------------------------------
