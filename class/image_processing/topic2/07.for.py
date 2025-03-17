kor = [70, 80 ,90, 40, 50]							# 리스트 선언
eng = [90, 80 ,70, 70, 60]
sum1, sum2, sum3, sum4 = 0, 0, 0, 0				    # 누적 변수 초기화

for i in range(0, 5):								# range() 함수로 범위지정
    sum1 = sum1 + kor[i] + eng[i]

for k in kor:										# 리스트 원소 순회
    sum2 = sum2 + k
for e in eng:
    sum2 = sum2 + e

for i, k in enumerate(kor):							# 리스트의 인덱스와 원소로 순회
    sum3 = sum3 + k + eng[i]

for k, e in zip(kor, eng):							# 여러 객체 동시 순회
    sum4 = sum4 + k + e

print ('sum1=', sum1), print ('sum2=', sum2)
print ('sum3=', sum3), print ('sum4=', sum4)