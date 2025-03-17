import chap03.header_area as mod
from chap03.header_area import write

mod.say()                                       # 함수 호출 – 인수없음, 반환값 없음
ret = mod.calc_area(type=1, a=5, b=5)			# 함수 호출 – 튜플 반환
write(ret[0], ret[1])


