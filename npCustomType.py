from ast import In
import numpy as np
from customType import Number

Intv_f = np.dtype([('lower', np.float32), ('upper', np.float32)])

itv1 = [0., 1.]
itv2 = [0., -1.]
itv_m1 = np.array([(0.0, 1.0), (0.0, 1.0)], dtype=Intv_f)
itv_m2 = np.array([(0.0, 2.0), (0.0, 2.0)], dtype=Intv_f)


def __add__(a, b):
    assert isinstance(a, Intv_f) and isinstance(b, Intv_f)
    return [a.lower + b.lower, a.upper + b.upper]


# def __mult__(a, b):
#     assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
#     assert (a.dtype == Intv_f and b.dtype == Intv_f)
#     assert (a.size > 0 and b.size > 0)



num_m1 = np.array([0., 0.], dtype = Number)
num_m2 = np.array([0., -1.], dtype = Number)
print(num_m1.dtype)

print(num_m1 + num_m2)