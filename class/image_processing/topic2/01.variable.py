'''# 01.variable.py - 변수의 자료형'''
variable1 = 100                             # 정수 변수 선언
variable2 = 3.14                            # 실수 변수 선언
variable3 = -200                            # 정수 변수 선언
variable4 = 1.2 + 3.4j                      # 복소수 변수 선언
variable5 = 'This is Python'                # 문자열 변수 선언

variable6 = True                            # bool 변수 선언
variable7 = float(variable1)                # 자료형 변경
variable8 = int(variable2)                  # 자료형 변경

print('variable1 =' , variable1, type(variable1))   # 변수의 값과 자료형 출력
print('variable2 =' , variable2, type(variable2))
print('variable3 =' , variable3, type(variable3))
print('variable4 =' , variable4, type(variable4))
print('variable5 =' , variable5, type(variable5))
print('variable6 =' , variable6, type(variable6))
print('variable7 =' , variable7, type(variable7))   # 실수로 변경되어 소수점 표시됨
print('variable8 =' , variable8, type(variable8))   # 정수로 변경되어 소수점이하 소실