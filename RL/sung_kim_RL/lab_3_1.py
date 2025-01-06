import sys
import math
import random

# implementation optimal Q learning 
# Gyuyeon Lim, lky473736

'''
[[], [], [], []], 
[[], [], [], []],
[[], [], [], []],
[[], [], [], []]
'''

cursor = [0, 0]
hole = set()
goal = [3, 3]

print ("<Q-learning simulation : frozenlake 4 X 4>")

print ("Enter the hole's number : ")
holes_N = int(sys.stdin.readline())

print ("Enter the holes' coordinates : ")
for _ in range (holes_N) :
    coor_hole = tuple(map(int, sys.stdin.readline().split()))
    hole.add(coor_hole)
    
print ("Enter the epoch : ")
N = int(sys.stdin.readline()) # epoch

Q = [[[0 for k in range (4)] for j in range (4)] for i in range (4)]
'''
in arr Q

0 : up
1 : right
2 : bottom
3 : left
'''

for _ in range (N) : 
    directions = []
    if cursor[0]-1 >= 0 : 
        directions.append(0)
    if cursor[0]+1 <= 3 : 
        directions.append(2)
    if cursor[1]+1 <= 3 : 
        directions.append(1)
    if cursor[1]-1 >= 0 :
        directions.append(3)
    
    direction = random.choice(directions)
    reward = 0
    token_hole = 0
    
    next_cursor = [0, 0]
    match (direction) : 
        case 0 : 
            next_cursor[0] = cursor[0]-1    
            next_cursor[1] = cursor[1]
        case 1 :
            next_cursor[0] = cursor[0]
            next_cursor[1] = cursor[1]+1
        case 2 : 
            next_cursor[0] = cursor[0]+1
            next_cursor[1] = cursor[1]
        case 3 : 
            next_cursor[0] = cursor[0]
            next_cursor[1] = cursor[1]
    
    if next_cursor == goal : 
        reward = 1
    if tuple(next_cursor) in hole : 
        reward = -1
        token_hole = 1
    
    max_value = max(Q[next_cursor[0]][next_cursor[1]])
    
    Q[cursor[0]][cursor[1]][direction] = reward + max_value
    
    if token_hole == 1 : 
        cursor = [0, 0] # hole -> go starting point
        continue
    
    cursor = next_cursor

print ()
for i in range (4) : 
    print (f'layer {i} : ')
    for j in range (4) : 
        print (Q[i][j])
        
print ("--------------------------")

def print_q_values(Q):
    grid = [[' '*5 for _ in range(12)] for _ in range(12)]
    
    for i in range(4):
        for j in range(4):
            row = i * 3
            col = j * 3
            
            grid[row][col+1] = f"{Q[i][j][0]:5d}"
            grid[row+1][col] = f"{Q[i][j][3]:5d}"
            grid[row+1][col+2] = f"{Q[i][j][1]:5d}"
            grid[row+2][col+1] = f"{Q[i][j][2]:5d}"
    
    for row in grid:
        print(" ".join(row))
        
print_q_values(Q) 