# 2-dim regression model에서 a, b를 찾는 것을 브루트포스로 구현.
# 오차 발생 가능성이 있음 (간격 변화량 0.01)

'''
    1) 시계방향으로 선형함수가 회전하면서 각 요소마다의 거리를 측정
        - 이때 선형함수는 y가 최대인 점과 y가 최소인 점 사이의 중점을 지난다고 가정
    2) 거리의 합이 최소가 되는 a 선정 -> b 도출
    3) formula로 구한 a와 b의 오차를 출력
'''

import sys

print ("--- 2-dim regression model : 브루트포스 알고리즘으로 a, b 구현 ---")
N = int(input("tuple의 갯수 작성 : "))

data = []

print ("아래에 공백을 기준으로 변수 x, y 작성")
for i in range (N) :
    x, y = map(int, sys.stdin.readline().split())
    data.append ([x, y])
    
# data를 y의 오름차순으로 정렬
data = sorted(data, key = lambda x : (x[1]))
print ("y의 오름차순으로 data 정렬")
print (":", data)

# y의 최대와 y의 최소의 중점을 기준으로 선형함수를 회전
mid = [(data[0][0] + data[-1][0] / 2), (data[0][1] + data[-1][1]) / 2]

# 오차 리스트
diff = []

# 초기 a == 수평직선 (y = c 상수함수 형태)
a = 0

# 브루트포스 알고리즘을 이용해 시뮬레이션으로 a, b 구하기
while True :
    sum_diff = 0
    b = mid[1] - a * mid[0]
    
    for compo in data :
        sum_diff += abs(compo[1] - compo[0] * a + b)
    
    if a == 0 :
        diff.append (sum_diff)
        
    else :
        if diff[-1] <= sum_diff :
            break
        
        else :
            diff.append (sum_diff)
        
    a += 0.01 # 간격 변화량
    
    print (f"a, b, diff = {a, b, sum_diff}")
    
# 기본 formula로 a와 b 구하기
mean_x = sum([data[i][0] for i in range (N)]) / N
mean_y = sum([data[i][1] for i in range (N)]) / N

x_mean_x = [data[i][0] - mean_x for i in range (N)]
y_mean_y = [data[i][1] - mean_y for i in range (N)]

sum_times = sum([x_mean_x[i] * y_mean_y[i] for i in range (N)])
sum_square = sum([x_mean_x[i] ** 2 for i in range (N)])

formula_a = sum_times / sum_square
formula_b = mean_y - formula_a * mean_x

print ("--------------")
print (f"최종 a, b = {a, b}")
print (f"formula를 이용한 a, b = {formula_a, formula_b}")
print (f"오차 : {a - formula_a, b - formula_b}")