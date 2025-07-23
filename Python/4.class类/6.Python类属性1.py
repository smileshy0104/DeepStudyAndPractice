# TODO 关于商品打折的应用，通过类属性可以设置统一的折扣：
class Product:
    discount = 0.5  # 类属性，所有实例共享

    def __init__(self, name, price):
        """
        初始化商品实例。

        :param name: 商品名称
        :param price: 商品价格
        """
        self.name = name  # 实例属性
        self.price = price  # 实例属性

    def get_price_after_discount(self):
        """
        计算打折后的价格。

        :return: 打折后的价格
        """
        return self.price * (1 - Product.discount)

# 创建实例
p1 = Product('Laptop', 1000)
p2 = Product('Phone', 500)

# 输出打折后的价格
print(p1.get_price_after_discount())  # 输出: 500.0
print(p2.get_price_after_discount())  # 输出: 250.0

print("\n计数器")

# TODO 计数器
class Counter:
    instance_count = 0

    def __init__(self):
        """
        初始化计数器实例，并增加类属性中的实例计数。
        """
        Counter.instance_count += 1

    @classmethod
    def get_instance_count(cls):
        """
        获取当前已创建的实例数量。

        :return: 实例数量
        """
        return cls.instance_count

c1 = Counter()
c2 = Counter()
print(Counter.get_instance_count())  # 输出: 2

print("\n配置和常量")

# TODO 配置和常量
# 类属性可以用于存储与类相关的常量或配置信息，例如数据库连接信息、默认配置参数等。
class Config:
    DATABASE_URI = "sqlite:///:memory:"
    DEBUG = True

print(Config.DATABASE_URI)  # 输出: sqlite:///:memory:
print(Config.DEBUG)         # 输出: True

print("\n共享状态")

# TODO 共享状态
# 当多个实例需要共享状态或数据时，类属性非常有用。
class Player:
    total_score = 0

    def __init__(self, score):
        """
        初始化玩家实例，并更新总分。

        :param score: 玩家得分
        """
        self.score = score
        Player.total_score += score

    @classmethod
    def get_total_score(cls):
        """
        获取所有玩家的总分。

        :return: 所有玩家的总分
        """
        return cls.total_score

p1 = Player(10)
p2 = Player(20)
print(Player.get_total_score())  # 输出: 30


print("\n缓存和共享资源")

# TODO 缓存和共享资源
# 类属性可以用来缓存计算结果或共享资源，以便所有实例都能访问，而无需每次都重新计算或创建。
class MathOperations:
    _factorial_cache = {}

    @classmethod
    def factorial(cls, n):
        """
        计算阶乘并缓存结果。

        :param n: 要计算阶乘的整数
        :return: n 的阶乘
        """
        if n in cls._factorial_cache:
            return cls._factorial_cache[n]
        if n == 0:
            result = 1
        else:
            result = n * cls.factorial(n-1)
        cls._factorial_cache[n] = result
        return result

print(MathOperations.factorial(5))  # 输出: 120
print(MathOperations._factorial_cache) # 输出: {5: 120, 4: 24, 3: 6, 2: 2, 1: 1, 0: 1}

print("\n类方法和静态方法")

# TODO 类方法和静态方法
# 类属性可以与类方法（使用 @classmethod 装饰器）或静态方法（使用 @staticmethod 装饰器）结合使用，以实现一些与类相关的操作或逻辑。

class MyClass:
    class_attribute = 0

    @classmethod
    def increment_class_attribute(cls):
        """
        增加类属性的值。
        """
        cls.class_attribute += 1

    @staticmethod
    def static_method_example():
        """
        返回一个静态方法示例字符串。

        :return: 示例字符串
        """
        return "This is a static method."

MyClass.increment_class_attribute()
print(MyClass.class_attribute)  # 输出: 1
print(MyClass.static_method_example())
# 输出: This is a static method.
