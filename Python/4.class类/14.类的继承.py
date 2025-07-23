# 继承是面向对象编程中的一个重要特性，它允许新定义的类（子类）继承已有类（父类）的属性和方法，从而实现代码的重用和扩展。
# class ClassName(Base1, Base2, Base3):
#     pass

# TODO 基本继承
class Student(object):
    """这是一个学生类"""
    def __init__(self, name):
        self.name = name

    def say(self):
        print(f'我的名字是：{self.name}')

    def add(self, x, y):
        print(f'这个加法我会，等于{x + y}')


class CollegeStudent(Student):
    def practice(self):
        print(f'我是{self.name}, 在世界500强实习。')

# 实例化 CollegeStudent 类
lily = CollegeStudent('lily')  # 实例化
lily.say()  # Student 的方法
# 输出: 我的名字是：lily
lily.practice()  # 调用自己的方法
# 输出: 我是lily, 在世界500强实习。

# TODO 方法重写
class CollegeStudent(Student):
    def say(self):
        print(f'大家好！我的名字是：{self.name}')

# 实例化 CollegeStudent 类
lily = CollegeStudent('lily')
lily.say()  # 调用重写的方法
# 输出: 大家好！我的名字是：lily

# TODO 使用 super() 函数
# super() 函数用于调用父类的方法，特别是在重写父类方法时，有时需要保留父类的功能。
class CollegeStudent(Student):
    def practice(self):
        super().say()  # 调用父类的 say 方法
        super(CollegeStudent, self).add(1, 5)  # 调用父类的 add 方法

# 实例化 CollegeStudent 类
lily = CollegeStudent('lily')
lily.practice()
'''
输出:
我的名字是：lily
这个加法我会，等于6
'''

# TODO 多继承
class Base1:
    def method(self):
        print("Base1 method")

class Base2:
    def method(self):
        print("Base2 method")

class Derived(Base1, Base2):
    pass

# 实例化 Derived 类
# Derived 类继承了 Base1 和 Base2 的方法，但由于 MRO 的原因，调用 method 时————优先调用 Base1 的实现。
obj = Derived()
obj.method()  # 输出: Base1 method

# TODO 私有属性和方法
# 如果希望某些属性和方法————不被子类继承，可以————使用双下划线前缀将其设为私有。
class Father:
    def __init__(self, money, house):
        self.money = money
        self.house = house
        # 私有属性
        self.__girl_friend = "Cuihua"

    def operating_company(self):
        print("李氏集团业绩平稳上升")

    # 私有方法
    def __love(self):
        print(f"父亲年轻时与{self.__girl_friend}谈恋爱")

# 尝试继承 Father 类
class Son(Father):
    def __init__(self, money, house):
        super().__init__(money, house)

    def show_info(self):
        print(f"我有{self.money}元和一栋{self.house}的房子。")

# 实例化 Son 类
son = Son(1000000, "别墅")
son.show_info()  # 输出: 我有1000000元和一栋别墅的房子。

# 尝试访问私有属性和方法
# print(son.__girl_friend)  # 会引发 AttributeError
# son.__love()  # 会引发 AttributeError


# TODO 内置方法重载
class CustomClass:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f'CustomClass with name: {self.name}'

# 实例化 CustomClass
obj = CustomClass('Example')
print(obj)  # 输出: CustomClass with name: Example
