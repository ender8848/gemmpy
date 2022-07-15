import numpy as np

# numpy custom interval type with lower and upper 
Intv = np.dtype([('lower', np.float64), ('upper', np.float64)])

np_intv_1 = np.ones(1, dtype = Intv) # array([(1., 1.)])
np_intv_2 = np.ones(1, dtype = Intv) # array([(1., 1.)])

print(np_intv_1 + np_intv_2) 
# numpy.core._exceptions.UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types


# 现在的问题：
# Interval class，需要模拟出real number的接口
# 如何解决一个等号就从real-valued 转化为 Interval
# 需要有一种直接指定size，就初始化Interval Array的方法，可能得用到
# 难道尼玛真的得去改numpy的源码？或者加一个源码补丁之类的？这也太逆天了吧