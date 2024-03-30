from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pylab as plt

x=np.array([[185, 60], [180, 60], [185, 70], [165, 63], [155, 68], [170, 75], [175, 80]])

# k-평균 알고리즘 모델 생성
model = KMeans(n_clusters=2)
model.fit(x) # 학습 및 군집화

print(model.labels_) # 데이터 포인트 각각이 속한 집단 출력
print(model.cluster_centers_) # fit 메서드로 학습한 집단의 중심

# 시각화
plt.scatter(x[:,0], x[:,1], c=model.labels_,cmap='rainbow')
plt.show()