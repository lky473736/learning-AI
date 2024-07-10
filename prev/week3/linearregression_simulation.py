# 2-dim 우상향 regression model에서 a, b를 찾는 것을 브루트포스로 구현.
# 오차 발생 가능성이 있음 (간격 변화량 0.01)

'''
    1) 시계방향으로 선형함수가 회전하면서 각 요소마다의 거리를 측정
        # - 이때 선형함수는 y가 최대인 점과 y가 최소인 점 사이의 중점을 지난다고 가정
        - ver 2. b 또한 구하기
    2) 거리의 합이 최소가 되는 a 선정 -> b 도출
    3) formula로 구한 a와 b의 오차를 출력
'''

import sys
import time

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
# mid = [(data[0][0] + data[-1][0] / 2), (data[0][1] + data[-1][1]) / 2]

# 오차 리스트 (모든 경우의 수)
diff_all = []

# nested loopstation으로 a에 따른 b의 변화에 따라 오차를 계산할 수 있게 하자
# a의 범위 : 0.1 ~ 100 / 간격 : 0.1
# b의 범위 : 0.1 ~ 100 / 간격 : 0.1

# a와 b의 범위 및 간격 설정
a_range = [i * 0.1 for i in range(0, 1001)]  # 0.1부터 100까지의 범위
b_range = [i * 0.1 for i in range(0, 1001)]  # 0.1부터 100까지의 범위

# 브루트포스 알고리즘을 이용해 a와 b 구하기
for a in a_range :
    for b in b_range :
        sum_diff = sum(abs(compo[1] - compo[0] * a + b) for compo in data)
        diff_all.append((a, b, sum_diff))

'''a = 1

while True :
    diff = []
    b = 1
    
    if a > 100 :
        break
    
    while True :
        sum_diff = 0
        
        for compo in data :
            sum_diff += abs(compo[1] - compo[0] * a + b)
        
        if a == 1 :
            diff.append (sum_diff)
            
        else :
            if diff[-1] <= sum_diff or b > 100 :
                break
            
            else :
                diff.append (sum_diff)
         
        b += 1
    
    print ([a, b, diff[-1]])
    time.sleep(1)
    diff_all.append ([a, b, diff[-1]])
    a += 1   '''
    
# diff가 최소가 되는 a, b 찾기
min_diff = min(diff_all, key=lambda x: x[2])
a, b, _ = min_diff
    
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

# 아래는 실패한 코드 (b -> a)

# while True :
#     diff = []
#     b += 0.01 # b의 변화량
    
#     # 초기 a == 수평직선 (y = c 상수함수 형태)
#     a = 0  
     
#     # 브루트포스 알고리즘을 이용해 시뮬레이션으로 a, b 구하기
#     while True :
#         sum_diff = 0
        
#         for compo in data :
#             sum_diff += abs(compo[1] - compo[0] * a + b)
        
#         if a == 0 :
#             diff.append (sum_diff)
            
#         else :
#             if diff[-1] <= sum_diff or a > 100 : # 기울기의 한도를 100이라고 fix
#                 # diff가 최소인 a, b 넣기
#                 diff_all.append ([a, b, diff[-1]])
#                 break
            
#             else :
#                 diff.append (sum_diff)
            
#         a += 0.01 # a의 변화량  
    
#     print ("---")
#     print (diff_all)
#     time.sleep(1)
    
#     if b >= 0.03 and (diff_all[-1][2] > diff_all[-2][2] or b > 100) : # y절편의 한도를 100이라고 fix
#         diff_all.pop()
#         break