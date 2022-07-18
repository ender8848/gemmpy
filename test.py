from ctypes import sizeof
from sys import getsizeof
import Interval

i = Interval.Interval(1, 2) 
# get size of i
print(getsizeof(i))


