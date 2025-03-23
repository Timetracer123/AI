import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv("/Users/ham-uichan/Desktop/class/25-1/AI/abalone.csv", index_col=0)
print(df, "\n-------------------")

# 결측치 확인
print(df.isnull().sum(), "\n-------------------")

# 레이블 확인 (imbalanced data)
print(df['Sex'].value_counts(), "\n-------------------")

# 데이터 encoding
encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])
print(df, "\n-------------------")

# 데이터 분할
X = df.drop('Rings',axis=1)
y = df['Rings']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12, shuffle=True)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LR (표준화 데이터)
model = LinearRegression()
model.fit(X_train_scaled, y_train)  
ly_preds = model.predict(X_test_scaled) 
print('평균제곱근오차(LR)', mean_squared_error(ly_preds, y_test))

plt.figure(figsize=(10, 5))
plt.scatter(X_test['Length'], y_test, label='Actual')  # 실제 값
plt.scatter(X_test['Length'], ly_preds, c='y', label='ly_preds')  # 예측값
plt.xlabel("Length")  # X축 라벨
plt.ylabel("Rings")   # Y축 라벨
plt.legend()
plt.show()

# DT
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
dy_preds = model.predict(X_test)
print('평균제곱근오차(DT)', mean_squared_error(dy_preds, y_test))

plt.figure(figsize=(10, 5))
plt.scatter(X_test['Length'], y_test, label='Actual')  # 실제 값
plt.scatter(X_test['Length'], dy_preds, c='r', label='dy_preds')  # 예측값
plt.xlabel("Length")  # X축 라벨
plt.ylabel("Rings")   # Y축 라벨
plt.legend()   
plt.show()

# RF
model = RandomForestRegressor()
model.fit(X_train, y_train)
ry_preds = model.predict(X_test)
print('평균제곱근오차(RF)', mean_squared_error(ry_preds, y_test))

plt.figure(figsize=(10, 5))
plt.scatter(X_test['Length'], y_test, label='Actual')  # 실제 값
plt.scatter(X_test['Length'], ry_preds, c='g', label='ry_preds')  # 예측값
plt.xlabel("Length")  # X축 라벨
plt.ylabel("Rings")   # Y축 라벨
plt.legend()
plt.show()

# SVR (표준화 데이터)
model = SVR(kernel='linear')
model.fit(X_train_scaled, y_train)
sy_preds = model.predict(X_test_scaled)
print('평균제곱근오차(RF)', mean_squared_error(sy_preds, y_test))

plt.figure(figsize=(10, 5))
plt.scatter(X_test['Length'], y_test, label='Actual')  # 실제 값
plt.scatter(X_test['Length'], sy_preds, c='orange', label='sy_preds')  # 예측값
plt.xlabel("Length")  # X축 라벨
plt.ylabel("Rings")   # Y축 라벨
plt.legend()
plt.show()