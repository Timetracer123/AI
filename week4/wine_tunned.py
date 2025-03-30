import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/MyungKyuYi/AI-class/refs/heads/main/wine.csv"
df = pd.read_csv(url)

print("--- 데이터 샘플 ---")
print(df.head())
print("\n--- 컬럼 목록 ---")
print(df.columns)
print("\n--- 타겟 변수 분포 ---")
print(df['Wine'].value_counts())

# --- 데이터 전처리 (타겟 매핑 추가) ---
y_original = df['Wine']
X = df.drop('Wine', axis=1)
y_mapped = y_original - 1 # 1,2,3 -> 0,1,2
Y = pd.get_dummies(y_mapped).values
class_labels = list(pd.get_dummies(y_mapped).columns) # [0, 1, 2]
X = X.values

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=y_mapped)

# 특성 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\n--- 데이터 형태 ---")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"Class labels for CM: {class_labels}")


# --- 모델 정의 ---
model = Sequential()
n_features = X_train.shape[1] # 입력 특성 수 (13)
n_classes = Y.shape[1]      # 출력 클래스 수 (3)

# 입력층 및 첫 번째 dense layer
model.add(Dense(15, input_shape=(n_features,), activation='relu'))
# 두 번째 dense layer
model.add(Dense(20, activation='relu'))
# 세 번째 dense layer
model.add(Dense(20, activation='relu'))
# 네 번째 dense layer
model.add(Dense(20, activation='relu'))
# 다섯 번째 dense layer
model.add(Dense(10, activation='relu'))
# 출력층
model.add(Dense(n_classes, activation='softmax'))

# 모델 컴파일 (학습률 등은 직접 지정)
learning_rate = 0.001 # 학습률 (필요시 조절)
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 구조 요약 출력
print("\n--- 사용자 지정 모델 구조 ---")
model.summary()

# --- 모델 학습 ---
print("\n--- 모델 학습 시작 ---")

history = model.fit(x=X_train, y=y_train,
                    epochs=150, # 학습할 에포크 수
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    verbose=1) # 학습 과정 출력
print("--- 모델 학습 종료 ---")


# --- 모델 평가 및 시각화 (학습된 'model'과 'history' 사용) ---
# 테스트 데이터 예측
y_pred_prob = model.predict(X_test)

# 확률을 클래스 레이블로 변환
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred_prob, axis=1)

# 정확도 평가
print("\n--- 모델 평가 ---")
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"Test Accuracy (sklearn): {accuracy:.4f}")

# Keras evaluate 사용
loss, keras_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy (Keras evaluate): {keras_accuracy:.4f}")
print(f"Test Loss (Keras evaluate): {loss:.4f}")

# 분류 리포트
print("\nClassification Report:")
target_names = [str(label) for label in class_labels]
print(classification_report(y_test_class, y_pred_class, target_names=target_names))


# 학습 과정 시각화
plt.figure(figsize=(12, 5))
plt.suptitle("Fixed Model Training History", fontsize=14)

# 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'bo-', label='Training loss')
plt.plot(history.history['val_loss'], 'ro-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'bo-', label='Training acc')
plt.plot(history.history['val_accuracy'], 'ro-', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# 혼동 행렬(Confusion Matrix) 시각화
cm = confusion_matrix(y_test_class, y_pred_class)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix (Fixed Model)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()