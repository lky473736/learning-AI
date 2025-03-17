a = [1.5, 2, 3, 4, 5]
b = map(float, a)
c = divmod(5, 3)

print('최댓값:', max(a) ,'최솟값:' , min(a) )
print( '몫과 나머지:' , c )
print( 'c의 자료형:', type(c), type(c[0]), type(c[1]) )

print('2의 4제곱:', pow(2, 4) )
print('절댓값:', abs(-4) )