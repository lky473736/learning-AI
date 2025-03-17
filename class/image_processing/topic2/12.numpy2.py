import numpy as np

a = np.zeros((2,5), np.int32)
b = np.ones((3,1), np.uint8)
c = np.empty((1,5), np.float64)
d = np.full(5, 15, np.float32)

print(type(a), type(a[0]), type(a[0][0]), a.dtype)
print(type(b), type(b[0]), type(b[0][0]), b.dtype)
print(type(c), type(c[0]), type(c[0][0]), c.dtype)
print(type(d), type(d[0]) , d.dtype)
print('c 형태:', c.shape, '   d 형태:', d.shape)
print(a), print(b)
print(c), print(d)