import numpy as np
from calculation_apis import *
from Interval import print_2d_array

def can_convert_real_numbered_array_to_Interval_array():
    a = np.ones((3,4))
    b = to_interval_array_np(a)
    assert(b.dtype == object)
    assert(b.shape == a.shape)
    assert(b[0,0] == Interval(1,1))

def can_convert_Interval_array_to_upper():
    a = np.array([[Interval(1,1), Interval(1,2)], [Interval(1,3), Interval(1,4)]])
    b = get_upper(a)
    assert(b.dtype == np.float32)
    assert(b.shape == a.shape)
    assert(b[1,1] == 4)

def can_convert_Interval_array_to_lower():
    a = np.array([[Interval(1,1), Interval(1,2)], [Interval(1,3), Interval(1,4)]])
    b = get_lower(a)
    assert(b.dtype == np.float32)
    assert(b.shape == a.shape)
    assert(b[1,1] == 1)

if __name__ == '__main__':
    can_convert_real_numbered_array_to_Interval_array()
    can_convert_Interval_array_to_upper()
    can_convert_Interval_array_to_lower()
