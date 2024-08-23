نحوه اجرای برنامه :
ابتدا صفحه data_feature.py را Run کرده تا دیتا های برنامه بروز شود و پروژه اطلاعات بروز را در دسترس داشته باشد
سپس صفحه Asli.py را Run کرده و نتیجه برنامه را بطور کل مشاهده میکنیم

#  استاد ممنون میشیم اگر قسمتی از برنامه ضعف داره یا نیاز به تقویت شدن داره  بفرمایید تا اصلاحش کنیم

-------------------------------------------------------------------------------------------------

پروژه پیشبینی ارز دیجیتال(بیتکوین)
نام استاد : احمد مصباح
نام دوره : machine learning mft 304520-21
پژوهشگران : روزبه مختارزاده  ||  محمد مهدی احدی
توضیحات : این پژوهش چگونگی پیشبینی ارز دیجیتال (برای مثال : بیتکوین) با استفاده از زبان ماشین (Machine Learning) را بررسی میکند.

اکنون به بررسی مراحل انجام این پروژه میپردازیم :


1- جمع اوری، مرتب سازی ، پاکسازی داده ها :

ابتدا بااستفاده از کتابخونه هایyfinance,pandas,pandas-ta داده های تاریخی قیمت ارز مورد نظر خود را می اوریم و سپس اندیکاتور های مورد نیاز را میسازیم
سپس 2 ستونtommorow close , tommorow open را تعریف میکنیم تا بتوانیم به قیمت های فردا هم دسترسی داشته باشیم
یک ستون به نام Benefit  تعریف میکنیم تا بتوانیم صعودی یا نزولی بودن کندل روز اتی را پیشبینی کنیم
سپس داده های خالی را با استفاده از مشخص کردن بازه زمانی  حذف میکنیم 
بین تمامی داده ها correlation گرفته و متناسب ترین انها را فیلتر و انتخاب میکنیم 

2-تقسیم بندی داده ها :

با استفاده از کتابخونه sklearn.model_selection import train_test_split  داده ها را به دو بخش تمرین و تست تقسیم میکنیم 
نکته : باید بخش تست رابرای کار با ارزهای دیجیتال بسیار کوچک قرار داده و shuffle =  False


3-بخش GridSearchCV 
در این بخش با استفاده از sklearn.model_selection import GridSearchCV برای هریک از مدل های مورد نظر ؛ بهترین فاکتورهای هر یک را که مربوط به پروژه میباشد را مشخص میکنیم تا بتوانیم بهترین دقت را از مدل های خود بگیریم


4-بخش مدل ها :
مدل های مورد نظر را داخل تابع های دلخواه تعریف میکنیم (با بهترین فاکتور ها)


5-استفاده از مدل ها:
با تمامی مدل ها داده های خود را fit و predict میکنیم


6-استفاده از مدل نهایی (دو بعدی):
یک مدل که بهترسین دقت را دارد برای پیشبینی مجدد از پیشبینی های ارایه شده از مدل ها انتخاب میکنیم تا بهترین پیشبینی ها را انتخاب کند و ارایه دهد


7-ارزیابی دقت مدل و تصویرسازی:
با استفاده از sklearn.metrics import classification_report دقت تمامی مدل ها را بدست اورده و نشان میدهیم و سپس با استفاده از seaborn,matplotlib  به تصویر سازی مدل نهایی میپردازیم


-----------------------------------------------------------------------------------------------------

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

# {'C': 9, 'max_iter': 500, 'penalty': 'l1', 'solver': 'liblinear'} 0.5275123839231892 --->logistic

# {'activation': 'logistic', 'hidden_layer_sizes': (300,), 'solver': 'lbfgs', 'max_iter' : 1500} 0.51514905584001 ---->mlp

# {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 400} 0.4644477239048167 ---> rf

# {'C': 3.0, 'kernel': "rbf", 'tol': 1e-5, 'gamma': 'scale'}  0.5259269979462605  ---> svm
-----------------------------------------------------------------------------------------------------
