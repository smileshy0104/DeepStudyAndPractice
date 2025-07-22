# 属性方法命名
# 单下划线、双下划线、头尾双下划三种分别是：
#
# _foo（单下划线）： 表示被保护的（protected）类型的变量，只能本身与子类访问，不能用于 from module import *
# __foo（双下划线）： 私有类型(private) 变量, 只允许这个类本身访问
# __foo__（头尾双下划）：特殊方法，一般是系统内置的通用属性和方法名，如 __init__()
# foo_（单后置下划线，单右下划线）：用于避免与关键词冲突，也用于初始化参数中表示不需要用户传入的变量，通常会定义初始值，如 love_ = None
# 注：以上属性（变量）和方法（函数）均适用。


class Student(object):
    """这是一个学生类"""

    def __init__(self, name):
        self.name = name  # 实例属性

    def say(self):
        print(f'我的名字是：{self.name}')  # 实例方法

    def add(self, x, y):
        print(f'这个加法我会，等于{x + y}')  # 实例方法


# 实例化学生类
tom = Student('Tom')  # 实例化
print(tom.name)  # 输出：Tom
tom.say()  # 让他说句话
# 输出：我的名字是：Tom
tom.add(1, 1)  # 让他计算加法
# 输出：这个加法我会，等于2

# 修改实例属性
tom.name = 'Tome'  # 修改姓名
tom.math = 88  # 增加一个新的属性并赋值

# TODO 删除实例属性
# del tom.math
# print(tom.math)

# 内置属性
print(tom.__doc__)  # 输出：'这是一个学生类'
print(tom.__dict__)  # 查看类的属性，是一个字典
print(tom.__class__)  # 所属类
print(tom.__class__.__name__)  # 类名
print(tom.__module__)  # 类定义所在的模块


class Car(object):
    __price = 50  # 私有变量
    speed = 120  # 公开变量

    def sell(self):
        return self.__price - 10  # 公开方法，返回价格减去10

    def get_price(self):
        return self.__price  # 获取私有变量

    def set_price(self, price):
        if price >= 0:  # 可以添加逻辑检查
            self.__price = price  # 设置私有变量


# 实例化汽车类
c = Car()  # 实例化
print(c.speed)  # 输出：120

# 下面的代码会引发错误，因为 __price 是私有的
# print(c.__price)  # AttributeError: 'Car' object has no attribute '__price'

print(c.sell())  # 输出：40

# 访问私有变量
print(c._Car__price)  # 输出：50

# 使用 getter 和 setter 方法
print(c.get_price())  # 输出：50
c.set_price(60)  # 设置新的价格
print(c.get_price())  # 输出：60
