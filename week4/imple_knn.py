from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 특징 및 종속 변수 데이터 생성
# x는 2차원 자료 / y는 label
x=np.array([[160, 60], [163, 60], [160, 64], [163, 64], [165, 61]])
# M : 0 / L : 1
y=np.array([0, 0, 1, 1, 1])

# kNN 모델 생성
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)    # 학습

# 예측
y_pred = model.predict([[162,61]]) 
y_pred_proba = model.predict_proba([[162,61]])
print(y_pred)
print(y_pred_proba)