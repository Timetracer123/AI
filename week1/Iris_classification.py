import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

file_path = "/Users/ham-uichan/Desktop/class/25-1/AI/iris.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("파일을 성공적으로 불러왔습니다.")
else:
    print("오류: 지정한 경로에 파일이 존재하지 않습니다.")

# 독립 변수(X)와 종속 변수(y) 지정
X = df.iloc[:, :-1] 
y = df.iloc[:, -1]   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("\n[Decision Tree]")
print("정확도:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("\n[Random Forest]")
print("정확도:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print("\n[SVM]")
print("정확도:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# Logistic Regression
lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("\n[Logistic Regression]")
print("정확도:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))