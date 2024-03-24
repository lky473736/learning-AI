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

print ("아래에 공백을 기준으로 변수 x, y, sorting 작성")
for i in range (N) :
    x, y, sorting = map(str, sys.stdin.readline().split())
    
    x = int(x)
    y = int(y)
    
    data[i] = [x, y, sorting];

print (data)
    
# object tuple과 data의 distance 구하기
distances = []

for i in range (N) :
    distance = [i, math.sqrt((data[i][0] - obj_x)**2 + (data[i][1] - obj_y)**2)]
    distances.append (distance)
    
print ("euclidean distance :", distances)
    
# distances를 euclidean distance를 기준으로 오름차순으로 정렬하기
# euclidean distance가 작다 == object tuple과 가깝다
distances.sort (key = lambda x : (x[1]))
print ("오름차순 정렬 :", distances)

# K에 따라 각 sorting의 count가 몇인지 측정 및 K에 따른 object의 sorting 출력
K = 1

print ("------------------------------")
while True :
    if K > N : # K가 N을 넘으면
        break
    
    proc = distances[:K] # 0부터 K-1까지의 원소들
    sorting_dict = {}
    
    for i in range (K) :
        try :
            sorting_dict[data[proc[i][0]][2]] += 1
                
        except KeyError :
            sorting_dict[data[proc[i][0]][2]] = 1
            
    # 완전 탐색을 통해 object sorting 구하기
    object_sorting = ['null', 0]
    
    for key in sorting_dict.keys() :
        if sorting_dict[key] > object_sorting[1] :
            object_sorting[0] = key
            object_sorting[1] = sorting_dict[key]
                
    print ("K, sorting_dict :", K, sorting_dict)
    print ("object sorting :", object_sorting[0])
    
    K += 2
    
    print ("------------------------------")
    
exit()