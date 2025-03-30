import pandas as pd
import numpy as np # numpy import 추가
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/MyungKyuYi/AI-class/refs/heads/main/wine.csv"
df = pd.read_csv(url)

# 데이터프레임 확인
print("--- 데이터 샘플 ---")
print(df.head())
print("\n--- 컬럼 목록 ---")
print(df.columns)
print("\n--- 타겟 변수 분포 ---")
print(df['Wine'].value_counts()) # 타겟 변수 클래스 확인

# 특성(X)과 타겟(y) 분리
y = df['Wine']
X = df.drop('Wine', axis=1)

# 타겟 변수 원-핫 인코딩 (클래스가 1, 2, 3이므로 get_dummies 사용)
Y = pd.get_dummies(y).values

# 특성 데이터 numpy 배열 변환
X = X.values

# 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=y) # stratify 추가 권장

# 특성 스케일링 (StandardScaler 적용)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # 학습 데이터로 fit 및 transform
X_test = scaler.transform(X_test)       # 테스트 데이터는 transform만 적용

print(f"\n--- 데이터 형태 ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# 딥러닝 모델 정의
model = Sequential()
# 입력층: input_shape는 특성 개수(13)와 동일해야 함
model.add(Dense(10, input_shape=(X_train.shape[1],), activation='relu')) # input_shape 수정
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
# 출력층: 클래스 개수(3)와 동일해야 하며, 다중 클래스 분류이므로 softmax 사용
model.add(Dense(3, activation='softmax')) # 유닛 수 3으로 수정

# 모델 컴파일
# 옵티마이저: Adam 사용
# 손실함수: 원-핫 인코딩된 다중 클래스 분류이므로 categorical_crossentropy 사용
model.compile(optimizer=Adam(learning_rate=0.001), # lr 대신 learning_rate 사용 및 값 조정
              loss='categorical_crossentropy',      # 손실 함수 변경
              metrics=['accuracy'])

# 모델 구조 요약 출력
print("\n--- 모델 구조 ---")
model.summary()

# 모델 학습
print("\n--- 모델 학습 시작 ---")
model_history = model.fit(x=X_train, y=y_train,
                          epochs=50, # epochs 늘려보기
                          batch_size=32,
                          validation_data=(X_test, y_test),
                          verbose=1) # 학습 과정 출력
print("--- 모델 학습 종료 ---")

# 테스트 데이터 예측
y_pred_prob = model.predict(X_test) # 확률값 예측

# 확률을 클래스 레이블로 변환
y_test_class = np.argmax(y_test, axis=1) # 원-핫 인코딩된 테스트 타겟을 클래스 번호로
y_pred_class = np.argmax(y_pred_prob, axis=1) # 예측된 확률을 클래스 번호로

# 정확도 평가 (sklearn 사용)
print("\n--- 모델 평가 ---")
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"Test Accuracy (sklearn): {accuracy:.4f}")

# Keras evaluate 사용 (loss와 accuracy 반환)
loss, keras_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy (Keras evaluate): {keras_accuracy:.4f}")
print(f"Test Loss (Keras evaluate): {loss:.4f}")

# 학습 과정 시각화 (손실)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo-', label='Training loss') # 색상 및 마커 변경
plt.plot(epochs, val_loss, 'ro-', label='Validation loss') # 색상 및 마커 변경
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 학습 과정 시각화 (정확도)
plt.subplot(1, 2, 2)
acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
plt.plot(epochs, acc, 'bo-', label='Training acc') # 색상 및 마커 변경
plt.plot(epochs, val_acc, 'ro-', label='Validation acc') # 색상 및 마커 변경
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout() # 그래프 간격 자동 조절
plt.show()

# 혼동 행렬(Confusion Matrix) 시각화
cm = confusion_matrix(y_test_class, y_pred_class)
# 클래스 레이블 가져오기 (get_dummies가 생성한 컬럼 순서대로)
class_labels = pd.get_dummies(y).columns.tolist()

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()