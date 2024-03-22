# 2dim K-NN을 수학적으로 구현

'''
    K값을 정하지 않고 2씩 점점 증가하여 통계낼 수 있도록 하자 
    (초기 K == 1, 2씩 증가하여 K가 항상 홀수이게 함)
    euclidean distance 사용하여 classification
'''

import sys
import math

print ("--- 2-dim K-NN model : euclidean distance를 이용하여 변수 K에 대한 data 정리 ---")
print ("공백을 기준으로 object tuple 작성 : ")
obj_x, obj_y = map(float, sys.stdin.readline().split())

N = int(input("tuple의 갯수 작성 : "))

# data를 해시로 받음 (x, y 좌표에 따른 분류를 빠른 속도로 찾기 위함)
data = dict()

print ("아래에 공백을 기준으로 변수 x, y, sorting 작성 (중복 허용 X)")
for i in range (N) :
    x, y, sorting = map(str, sys.stdin.readline().split())
    
    x = float(x)
    y = float(y)
    
    data[f"[{x}, {y}]"] = sorting;
    
positions = list(data.keys())
    
# object tuple과 data의 distance 구하기
distances = []

for i in range (N) :
    distance = [str(positions[i]), math.sqrt((positions[i][0] - obj_x)**2 + (positions[i][1] - obj_y)**2)]
    distances.append (distance)
    
# distances를 euclidean distance를 기준으로 오름차순으로 정렬하기
# euclidean distance가 작다 == object tuple과 가깝다
distance.sort (key = lambda x : (x[1]))
print (distances)