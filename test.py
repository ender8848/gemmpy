import Interval
import numpy as np
from sys import getsizeof

# size test
i = Interval.Interval(1, 2) 
j = float(1)
# get size of i
print(getsizeof(i))
print(getsizeof(j))


a = np.ones(1, dtype=np.float32)
b = np.ones(2, dtype=np.float32)
c = np.ones(4, dtype=np.float32)

print(getsizeof(a))
print(getsizeof(b))
print(getsizeof(c))

d = np.eye(4, dtype=np.float32)
print()


## cp test

