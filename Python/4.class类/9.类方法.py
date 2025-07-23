# TODO 类方法：它是绑定到类而不是绑定到实例上的。
#  类方法可以通过类本身调用，也可以通过类的实例调用。
#  类方法使用 @classmethod 装饰器来定义，第一个参数约定为 cls，用于指代类本身。

# 类方法的用途：
# 修改类属性：类方法可以方便地修改类属性，而不必依赖实例。
# 工厂方法：类方法可以用来创建类的实例，并返回这些实例。
# 实现多态：在继承体系中，类方法可以实现子类的多态行为。

class MyClass:
    class_attribute = 0

    def __init__(self, value):
        self.instance_attribute = value

    @classmethod  # 装饰器，定义类方法
    def set_class_attribute(cls, value):
        cls.class_attribute = value

    @classmethod
    def get_class_attribute(cls):
        return cls.class_attribute

# 通过类调用类方法
MyClass.set_class_attribute(10)
print(MyClass.get_class_attribute())  # 输出: 10

# 通过实例调用类方法
obj = MyClass(20)
obj.set_class_attribute(30)
print(obj.get_class_attribute())  # 输出: 30
print(MyClass.get_class_attribute())  # 输出: 30

# TODO 修改类属性
class Counter:
    _count = 0

    def __init__(self):
        Counter._count += 1

    @classmethod
    def get_count(cls):
        return cls._count

# 创建实例
c1 = Counter()
c2 = Counter()
print(Counter.get_count())  # 输出: 2

# TODO 工厂方法
class Pizza:
    """
    Pizza类用于创建不同类型的披萨对象

    Attributes:
        ingredients (list): 披萨的配料列表
    """

    def __init__(self, ingredients):
        """
        初始化Pizza对象

        Args:
            ingredients (list): 披萨的配料列表
        """
        self.ingredients = ingredients

    @classmethod
    def margherita(cls):
        """
        工厂方法，创建玛格丽特披萨

        Returns:
            Pizza: 包含番茄和马苏里拉奶酪的玛格丽特披萨对象
        """
        return cls(['mozzarella', 'tomatoes'])

    @classmethod
    def pepperoni(cls):
        """
        工厂方法，创建意大利辣香肠披萨

        Returns:
            Pizza: 包含番茄、马苏里拉奶酪和意大利辣香肠的披萨对象
        """
        return cls(['mozzarella', 'tomatoes', 'pepperoni'])

# 创建不同类型的 Pizza
pizza1 = Pizza.margherita()
pizza2 = Pizza.pepperoni()

print(pizza1.ingredients)
# 输出: ['mozzarella', 'tomatoes']
print(pizza2.ingredients)
# 输出: ['mozzarella', 'tomatoes', 'pepperoni']


# TODO 实现多态
class Animal:
    """
    动物基类，定义了动物的基本行为接口

    这是一个抽象基类，要求所有子类必须实现speak方法
    """
    @classmethod
    def speak(cls):
        """
        动物发声方法

        Args:
            cls: 类对象本身

        Returns:
            str: 动物的叫声

        Raises:
            NotImplementedError: 当子类没有实现此方法时抛出异常
        """
        raise NotImplementedError("Subclasses must implement this method")

class Dog(Animal):
    """
    狗类，继承自动物基类

    实现了狗的特定行为，包括发声
    """
    @classmethod
    def speak(cls):
        """
        狗的发声方法

        Args:
            cls: 类对象本身

        Returns:
            str: 狗的叫声"Woof!"
        """
        return "Woof!"

class Cat(Animal):
    """
    猫类，继承自动物基类

    实现了猫的特定行为，包括发声
    """
    @classmethod
    def speak(cls):
        """
        猫的发声方法

        Args:
            cls: 类对象本身

        Returns:
            str: 猫的叫声"Meow!"
        """
        return "Meow!"

# 调用各类的发声方法并打印结果
print(Dog.speak())  # 输出: Woof!
print(Cat.speak())  # 输出: Meow!
