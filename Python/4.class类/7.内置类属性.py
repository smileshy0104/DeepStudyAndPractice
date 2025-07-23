# TODO 先调用 new 函数再调用 init 函
#  __new__ 会创建对象，相当于构造器，起创建一个类实例的作用，__init__ 作为初始化器，负责对象的初始化。

class Student(object):
    """
    这是一个学生类，用于表示一个学生对象。

    该类支持实例化时传入姓名和年龄，并提供了字符串表示、可调用等特殊方法。
    """

    def __new__(cls, *args, **kwargs):
        """
        创建Student类的新实例。

        :param cls: 类本身
        :param args: 位置参数
        :param kwargs: 关键字参数
        :return: 新创建的实例
        """
        print("Student.__new__ called")
        return super(Student, cls).__new__(cls)

    def __init__(self, name, age):
        """
        初始化Student实例。

        :param name: 学生姓名（字符串）
        :param age: 学生年龄（整数）
        """
        print("Student.__init__ called")
        self.name = name
        self.age = age

    def __call__(self):
        """
        使Student实例可以像函数一样被调用。

        每次调用会使学生的年龄增加1岁，并打印提示信息。
        """
        self.age += 1
        print('我能执行了')

    def __str__(self):
        """
        返回Student实例的非正式字符串表示，适用于用户阅读。

        :return: 格式化的字符串，包含姓名和年龄
        """
        return f'姓名：{self.name}，年龄：{self.age}'

    def __repr__(self):
        """
        返回Student实例的官方字符串表示，适用于开发者调试。

        :return: 可以用来重新构建该对象的字符串表达式
        """
        return f'Student(name={self.name}, age={self.age})'


# 实例化Student对象
lily = Student('lily', 18)

# 查看文档字符串
print(lily.__doc__)  # 输出: 这是一个学生类

# 查看类的名称
print(Student.__name__)  # 输出: Student

# 查看实例的属性和方法
print(dir(lily))
# 输出: ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__',
#         '__eq__', '__format__', '__ge__', '__getattribute__',
#         '__gt__', '__hash__', '__init__', '__init_subclass__',
#         '__le__', '__lt__', '__module__', '__ne__', '__new__',
#         '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',
#         '__sizeof__', '__str__', '__subclasshook__', '__weakref__',
#         'age', 'name']

# 调用 __call__ 方法
print(callable(lily))  # 输出: True
lily()  # 输出: 我能执行了
print(lily.age)  # 输出: 19

# 打印实例的字符串表示
print(lily)  # 输出: 姓名：lily，年龄：19
print(repr(lily))  # 输出: Student(name=lily, age=19)
