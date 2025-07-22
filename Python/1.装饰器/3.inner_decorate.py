# 使用 @staticmethod 装饰器定义一个静态方法
class MathOperations:
    @staticmethod
    def add(x, y):
        """
        计算两个数的和。

        参数:
        x -- 第一个加数
        y -- 第二个加数

        返回值:
        两个数的和
        """
        return x + y

# 使用静态方法
result = MathOperations.add(5, 3)
print(result)  # 输出: 8


# 使用 @classmethod 装饰器定义一个类方法
class Counter:
    count = 0

    @classmethod
    def increment(cls):
        """
        增加类的计数器。

        参数:
        cls -- 类本身，允许访问类属性和方法
        """
        cls.count += 1

# 使用类方法
Counter.increment()
print(Counter.count)  # 输出: 1


# 使用 @property 装饰器将方法转换为属性
class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        """
        计算圆的面积。

        返回值:
        圆的面积
        """
        return 3.14 * self.radius ** 2

# 使用属性
circle = Circle(5)
print(circle.area)  # 输出: 78.5

# 使用 @functools.wraps 装饰器以保留函数元数据
import functools

def my_decorator(func):
    """
    一个简单的装饰器，打印一条消息，然后调用原始函数。

    参数:
    func -- 被装饰的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        执行被装饰函数前的额外操作，然后调用被装饰的函数。

        参数:
        *args -- 位置参数
        **kwargs -- 关键字参数
        """
        print("调用函数前")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def say_hello():
    """打印问候语"""
    print("你好!")

# 调用装饰过的函数
say_hello()
print(say_hello.__name__)  # 输出: say_hello
print(say_hello.__doc__)   # 输出: 打印问候语
