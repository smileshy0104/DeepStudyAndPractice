# 1. 简单的消息
class SimpleMeta(type):
    """
    自定义元类，用于在创建类时打印一条消息。

    :param cls: 元类本身
    :param clsname: 要创建的类的名称
    :param bases: 类的基类元组
    :param dct: 类的属性和方法字典
    :return: 新创建的类对象
    """
    def __new__(cls, clsname, bases, dct):
        print(f"Creating class: {clsname}")
        return super().__new__(cls, clsname, bases, dct)

# 使用自定义的元类来创建类
class MyClass(metaclass=SimpleMeta):
    pass

# 输出: Creating class: MyClass

# 2. 修改类的定义
class MyMeta(type):
    """
    自定义元类，在类创建时为其添加一个方法。

    :param cls: 元类本身
    :param name: 要创建的类的名称
    :param bases: 类的基类元组
    :param dct: 类的属性和方法字典
    :return: 新创建的类对象
    """
    def __new__(cls, name, bases, dct):
        # 在类的字典中添加一个新的方法
        dct['added_method'] = lambda self: "This is an added method"
        return super().__new__(cls, name, bases, dct)

class MyClassWithMethod(metaclass=MyMeta):
    pass

instance = MyClassWithMethod()
print(instance.added_method())  # 输出: This is an added method

# 3. 验证类的定义
class ValidationMeta(type):
    """
    自定义元类，用于验证类是否定义了特定的方法。

    :param cls: 元类本身
    :param name: 要创建的类的名称
    :param bases: 类的基类元组
    :param dct: 类的属性和方法字典
    :return: 新创建的类对象
    :raises TypeError: 如果类没有定义 required_method 方法
    """
    def __new__(cls, name, bases, dct):
        # 检查类是否定义了 required_method 方法
        if 'required_method' not in dct:
            raise TypeError(f"{name} must define required_method")
        return super().__new__(cls, name, bases, dct)

class ValidClass(metaclass=ValidationMeta):
    def required_method(self):
        pass

# 这个类将会引发 TypeError
# class InvalidClass(metaclass=ValidationMeta):
#     pass

# 4. 自动注册类
# 用于存储已注册类的字典
registry = {}

class RegistrationMeta(type):
    """
    自定义元类，用于自动将创建的类注册到全局字典中。

    :param cls: 元类本身
    :param name: 要创建的类的名称
    :param bases: 类的基类元组
    :param dct: 类的属性和方法字典
    :return: 新创建的类对象
    """
    def __new__(cls, name, bases, dct):
        # 创建新的类对象
        new_class = super().__new__(cls, name, bases, dct)
        # 将新创建的类注册到全局字典中
        registry[name] = new_class
        return new_class

class RegisteredClass(metaclass=RegistrationMeta):
    pass

class AnotherRegisteredClass(metaclass=RegistrationMeta):
    pass

print(registry)
# 输出: {'RegisteredClass': <class '__main__.RegisteredClass'>,
#         'AnotherRegisteredClass': <class '__main__.AnotherRegisteredClass'>}

