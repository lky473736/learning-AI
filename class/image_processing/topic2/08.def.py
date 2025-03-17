def calc_area(type, a, b, c = None):
    if type == 1 :                      # 사격형
        result = a * b
        msg = '사각형'
    elif type == 2:                     # 삼각형
        result = a * b / 2
        msg = '삼각형'
    elif type == 3:                     # 평행사변형
        result = (a + b) * c / 2
        msg = '평행사변형'
    return result, msg                  # 반환값 – 2개 반환하면 튜플 반환

def say():
    print ('넓이를 구해요')

def write(result, msg):
    print ( msg,' 넓이는 ', result , '㎡ 입니다.')

say()                                   # 함수 호출 – 인수&반환값 없음
ret = calc_area(type=1, a=5, b=5)		# 함수 호출 – 튜플 반환
area, msg = calc_area(2, 5, 5)			# 함수 호출 – 튜플을 각 원소별로 반환
area2, _ = calc_area(3, 10, 7, 5)		# 함수 호출 – 반환 받을 원소만 지정

print(type(ret))
print(type(area),type(msg) )
write(ret[0], ret[1])
write(area, msg)
write(area2, '평행사변형' )

