import numpy as np

np.random.seed(10)
a = np.random.rand(2, 3)
b = np.random.randn(5, 1)
c = np.random.rand(6)
d = np.random.randint(1, 100, 6)
c = np.reshape(c, (2, 3))              # 형태(shape) 변경 방법1
d = d.reshape(2,-1)                    # 형태(shape) 변경 방법2

print('a 형태:', a.shape, '\n', a)
print('b 형태:', b.shape, '\n', b)
print('c 형태:', c.shape, '\n', c)
print('d 형태:', d.shape, '\n', d)

print('다차원 객체 1차원 변환 방법' )
print('a =', a.flatten())                   # 다차원 ndarray 객체를 1차원 벡터로 변환
print('b =', np.ravel(b))                   # 다차원 모든 객체를 1차원 벡터로 변환
print('c =', np.reshape(c, (-1,)))          # 넘파이의 reshape() 함수 사용
print('d =', d.reshape(-1, ))               # ndarray 객체 내장 reshape() 함수 사용

