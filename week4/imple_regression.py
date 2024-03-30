import numpy as np

# x값(입력)과 y값(정답)
x=np.array([14, 16, 17, 18.75, 11, 15.5, 23.5])
y=np.array([245, 312, 279, 308, 199, 219, 405])

# x와 y의 평균값
mx = np.mean(x)
my = np.mean(y)
print("x의 평균값:", mx)
print("y의 평균값:", my)

# 기울기 공식의 분모
divisor = 0
for i in x :
  divisor += (mx - i)**2

# 기울기 공식의 분자
dividend = 0
for i in range(len(x)) :
  dividend += (x[i] - mx) * (y[i] - my)

print("분모:", divisor)
print("분자:", dividend)

# 기울기 a와 y 절편 b 구하기
a = dividend / divisor
b = my - (mx*a)

# 기울기 a와 y 절편 b 출력(확인)
print("기울기 a =", a)
print("y 절편 b =", b)

# 예측에 활용하기
xp = 14.25
y = a * xp + b

print("예측값 =", y)