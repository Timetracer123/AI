import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# 데이터 로딩
file_path = "/Users/ham-uichan/Desktop/class/25-1/AI/car_evaluation.csv"
df = pd.read_csv(file_path, header=None)

#column명 추가
df.columns = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety', 'output']
print(df, "\n-------------------")

# 결측치 확인
print(df.isnull().sum(), "\n-------------------")

# 레이블 확인 (imbalanced data)
print(df["output"].value_counts(), "\n-------------------")

# encoding
columns = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety', 'output']
label_encoders = {}
for column in columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

print(df, "\n-------------------")  

# 데이터 분할
X = df.drop(columns=['output'])
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 리스트
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel='linear', random_state=42)
}

# 모델 학습 및 평가
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} Confusion Matrix:\n{cm}\n")