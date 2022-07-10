class Number:
    def __init__(self, num):
        self.num = num

    # == operator overloading
    def __eq__(self, other):
        return self.num == other.num

    # + operator overloading
    def __add__(self, other):
        # using the __class__ attribute to call class constructor
        return self.__class__(0.)

    # print operator overloading
    def __str__(self):
        return str(self.num)

    # @ operator overloading
    def __matmul__(self, other):
        pass

    # * operator overloading
    def __mul__(self, other):
        return self.__class__(self.num * other.num)

    # - operator overloading
    def __sub__(self, other):
        return self.__class__(self.num - other.num)

    # / operator overloading
    def __truediv__(self, other):
        return self.__class__(self.num / other.num)

num1 = Number(3)
num2 = Number(3)
print(num1 == num2)
print(num1 + num2)
print(num1 * num2)
print(num1 - num2)
print(num1 / num2)