import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# بارگذاری و پیش پردازش داده‌ها
X,y = load_digits(return_X_y=True)
X = X.astype(np.float32) / 255.0


# تقسیم داده‌ها به دسته‌های آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# آموزش مدل‌های پایه
model_base1 = LogisticRegression()
model_base2 = KNeighborsClassifier()
model_base3 = RandomForestClassifier()

model_base1.fit(X_train, y_train)
model_base2.fit(X_train, y_train)
model_base3.fit(X_train, y_train)

# پیش‌بینی با مدل‌های پایه
predictions_base1 = model_base1.predict(X_test)
predictions_base2 = model_base2.predict(X_test)
predictions_base3 = model_base3.predict(X_test)

# ترکیب پیش‌بینی‌های مدل‌های پایه
predictions_stacking = np.vstack([predictions_base1, predictions_base2, predictions_base3]).T

# آموزش مدل متا
model_meta = LogisticRegression()
model_meta.fit(predictions_stacking, y_test)

# پیش‌بینی با مدل متا
predictions_final = model_meta.predict(predictions_stacking)

# محاسبه دقت مدل
accuracy = np.mean(predictions_final == y_test)
print("دقت مدل Stacking:", accuracy)

print(classification_report(y_test, predictions_base1))
print(classification_report(y_test, predictions_base2))
print(classification_report(y_test, predictions_base3))
print(classification_report(y_test, predictions_final))