class Number(object): 
    def __init__(self, num):
        self.num = float(num)

    # override division operator
    def __truediv__(self, other: 'Number'):
        return self.num / other.num


if __name__ == '__main__':
    a = Number(1)
    b = Number(2)
    print(a / b)

