import math
import numpy as np

def print_2d_array(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            print(array[i][j], end=" ")
        print()


class Interval():
    # constructor
    def __init__(self,*kwargs):
        if (len(kwargs) == 0):
            self.lower = float(0)
            self.upper = float(0)
        elif (len(kwargs) == 1):
            self.lower = float(kwargs[0])
            self.upper = float(kwargs[0])
        elif (len(kwargs) == 2):
            self.lower = float(kwargs[0])
            self.upper = float(kwargs[1])
        else:
            raise ValueError("Invalid constructor parameters: expected 0-2 parameters")
    
    # actually false implementation
    def __add__(self, other: 'Interval'):
        if  not isinstance(other, Interval):
            other = Interval(other)
        return Interval(
            math.nextafter(self.lower + other.lower, -math.inf),
            math.nextafter(self.upper + other.upper, math.inf))

    def __iadd__(self, other: 'Interval'):
        return self + other
    
    def __sub__(self, other: 'Interval'):
        if  not isinstance(other, Interval):
            other = Interval(other)
        return Interval(
            math.nextafter(self.lower - other.upper, -math.inf),
            math.nextafter(self.upper - other.lower, math.inf))
    
    def __isub__(self, other: 'Interval'):
        return self - other
    
    def __mul__(self, other: 'Interval'):
        if  not isinstance(other, Interval):
            other = Interval(other)
        l = math.nextafter(
            min(self.lower * other.lower, self.upper * other.upper, self.lower * other.upper, self.upper * other.lower)
            , -math.inf)
        u = math.nextafter(
            max(self.lower * other.lower, self.upper * other.upper, self.lower * other.upper, self.upper * other.lower)
            , math.inf)
        return Interval(l, u)

    def __imul__(self, other: 'Interval'):
        return self * other

    def __truediv__(self, other: int):
        if  not isinstance(other, Interval):
            other = Interval(other)
        l = math.nextafter(
            min(self.lower / other.lower, self.upper / other.upper, self.lower / other.upper, self.upper / other.lower)
            , -math.inf)
        u = math.nextafter(
            max(self.lower / other.lower, self.upper / other.upper, self.lower / other.upper, self.upper / other.lower)
            , math.inf)
        return Interval(l, u)
    
    def __itruediv__(self, other: 'Interval'):
        return self / other

    def __str__(self):
        return f"[{self.lower:.4}, {self.upper:.4}]"

    def __eq__(self, other):
        if not isinstance(other, Interval):
            other = Interval(other)
        return self.lower == other.lower and self.upper == other.upper
    
if __name__ == '__main__':
    # add test
    a = Interval(1,2)
    b = Interval(3,4)
    print(f"a+b: {a+b}")
    a += b
    print(f"a+=b:{a}")

    # minus test
    a = Interval(1,2)
    b = Interval(3,4)
    print(f"a-b: {a-b}")
    a -= b
    print(f"a-=b:{a}")

    # mul test
    a = Interval(1,2)
    b = Interval(3,4)
    print(f"a*b: {a*b}")
    a *= b
    print(f"a*=b:{a}")

    # div test
    a = Interval(1,2)
    b = Interval(3,4)
    print(f"a/b: {a/b}")
    a /= b
    print(f"a/=b:{a}")

    # test vectorize
    a = np.array([[Interval(1,1), Interval(1,1)], [Interval(1,1), Interval(1,1)]], dtype = Interval)
    b = np.array([[Interval(1,1), Interval(1,1)], [Interval(1,1), Interval(1,1)]], dtype = Interval)
    c = a @ b
    print("array a:")
    print_2d_array(a)
    print("array b:")
    print_2d_array(b)
    print("array c = a @ b:")
    print_2d_array(c)

    # test soundness
    a = np.array([[Interval(1,1), Interval(1,1)], [Interval(1,1), Interval(1,1)]]) # no need to specify dtype, python will veiw it as object
    for i in range (45): # propogate error 2^45 times (estimated)
        a = a @ a / 2
    print("array a:")
    print_2d_array(a)

    # other properties
    print("cannot use APIs like ones, zeros as it will change the type")
    a = np.ones((3,4))
    print(a.dtype)
    a = np.ones((3,4)).astype(Interval)
    print(a.dtype) # 直接把int变成了object，因为不能识别Interval类


    

    # 好吧就这个思路
    # 定义好interval的operator overloading，直接用np初始化
    # 支持对整个array直接+，*，@
    # 但是不能用一些方便的初始化（或者批量操作），例如ones，zeros，eye，
    # 因为这种做法相当于先用默认类型（例如float32）创建一个数组，在astype进行类型转换，但是astype的时候只会变成object，而不是Interval
    # 因此这个数组的创建，要么靠自己输入，要么靠一个自定义函数来转换
    # vectorize之后，numpy的的接口能直接算是因为@只用到一些__add__等接口，而这些是是基类object自带的，
    # 由于python用的都是dynamic binding，Interval类只是override了一下，因此能被直接调用。
    # 最后还需要2个快捷的函数，一个是由float/int/double数组转成interval类型的ndarray
    # 另一个是由interval的ndarray快速取得upper和lower数组


    # 关于最终的计算，差不多这样
    # if (!interval) 直接用python默认接口就行
    # else if (cpu) 计算直接代理给np，转换用自己的接口
    # else if (gpu) 计算和转换用自己的接口
    # 这种逻辑最好用积累和子类实现，也就是说还是得定义好自己的接口，不然比较麻烦
