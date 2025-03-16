import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# file_path = "/Users/ham-uichan/Desktop/class/25-1/AI/titanic.csv"
# df = pd.read_csv(file_path)

# 데이터 로딩
url = "https://raw.githubusercontent.com/MyungKyuYi/AI-class/refs/heads/main/titanic.csv"
df = pd.read_csv(url)

# 결측치 확인
print(df.isnull().sum(), "\n-------------------")

# Age 평균값 구하고 채우기
age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)

# 레이블 확인 (imbalanced data)
print(df['Survived'].value_counts(), "\n-------------------")

# 불필요한 columns 제거
df = df.drop(columns=['Name', 'Ticket', 'Cabin'])

# 엔코딩
encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])  # 남자 - 1, 여자 - 0
df = pd.get_dummies(df, columns=['Embarked'])
print(df['Survived'].value_counts(), "\n-------------------")

print(df.head())
print(df.columns, "\n-------------------")

# 데이터 분할
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 리스트
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
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