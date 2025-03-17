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
    return result, msg

def say():
    print ('넓이를 구해요')

def write(result, msg):
    print ( msg,' 넓이는 ', result , ' ㎡ 입니다.')