import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt

# x값(입력)과 y값(정답)
ori_x = np.array([14, 16, 17, 18.75, 11, 15.5, 23.5])
ori_y = np.array([245, 312, 279, 308, 199, 219, 405])

# 사이킷런은 입력 데이터가 반드시 2차원 데이터여야 함. 따라서 (-1, 1)로 reshape 과정을 반드시 해야 함 (np)
x=np.array(ori_x).reshape(-1,1) 
y=np.array(ori_y)

# 선형회귀 모델 생성 및 fit()를 이용해 학습 (compile 할 필요 없음 (따로 layer을 만들지 않고 이미 만들어진 모델 사용하였기 때문에))
# sklearn, keras == model 생성을 도와주는 라이브러리
model = LinearRegression()
model.fit(x, y) # train_X, train_y를 이용해 학습시킴

# 기울기 a와 y 절편 b 출력 (확인)
print("기울기 a =", model.coef_)      # 기울기 출력
print("y 절편 b =", model.intercept_) # 절편 출력

# 예측
y_pred = model.predict([[14.25]])
print("예측값 =", y_pred)

# 모델 시각화
plt.scatter(x,y,color='black')
y_pred=model.predict(x)
plt.plot(x,y_pred,color='red')
plt.show()