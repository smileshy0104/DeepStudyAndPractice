
class MyClass:
    def __init__(self):
        self.__private_var = 42 # 私有属性

    def __private_method(self): # 私有方法
        print("This is a private method")

    def get_private_var(self):
        return self.__private_var

# TODO: 私有变量的初衷是防止外部访问，但在 Python 中并非真正的私有
#  私有变量的名称实际上会被改写为 _ClassName__variableName，以此来防止直接访问
obj = MyClass()
# print(obj.__private_var)  # 这行代码会导致 AttributeError

# 通过名称改写访问私有变量
print(obj._MyClass__private_var)  # 输出: 42

# 通过公共方法访问私有变量
print(obj.get_private_var())  # 输出: 42

# 私有方法也可以通过名称改写访问：
print(obj._MyClass__private_method())  # 输出: This is a private method


class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner
        self.__balance = balance  # 私有变量

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
        else:
            print("Deposit amount must be positive")

    def __calculate_interest(self):  # 私有方法
        return self.__balance * 0.05

    def add_interest(self):
        interest = self.__calculate_interest()
        self.__balance += interest
        print(f"Interest added: {interest}")

    def get_balance(self):
        return self.__balance

account = BankAccount("Alice", 1000)
account.deposit(500)
print(account.get_balance())  # 输出: 1500
account.add_interest()        # 输出: Interest added: 75.0
print(account.get_balance())  # 输出: 1575



class Car:
    def __init__(self, make, model, color):
        self.make = make  # 公有属性
        self.model = model  # 公有属性
        self.color = color  # 公有属性
        self.__fuel_capacity = 50  # 私有属性，燃油容量 (单位: 升)
        self.__fuel_level = 0  # 私有属性，当前燃油水平 (单位: 升)

    def __update_fuel_level(self, amount):
        # 私有方法，更新燃油水平
        if amount > 0:
            self.__fuel_level += amount
            if self.__fuel_level > self.__fuel_capacity:
                self.__fuel_level = self.__fuel_capacity

    def drive(self, distance):
        # 公有方法，驾驶汽车
        fuel_needed = distance / 10  # 假设每行驶10公里消耗1升燃油
        if fuel_needed <= self.__fuel_level:
            print(f"开 {distance} 公里.")
            self.__fuel_level -= fuel_needed
        else:
            print("没有足够的燃料开那么远。")

    def refuel(self, amount):
        # 公有方法，加油
        self.__update_fuel_level(amount)
        print(f"加油后的油位: {self.__fuel_level}/{self.__fuel_capacity} 升.")

    def get_fuel_level(self):
        # 公有方法，获取当前燃油水平
        return self.__fuel_level

    def __str__(self):
        # 公有方法，返回汽车的描述信息
        return f"{self.color} {self.make} {self.model}"

# 使用Car类
my_car = Car("Toyota", "Camry", "Blue")
print(my_car) # Blue Toyota Camry

my_car.refuel(30)
my_car.drive(150)
my_car.get_fuel_level()


# TODO 数据封装
# 私有变量有助于将类的内部实现细节隐藏起来，只暴露公共接口给外部。
# 这样可以防止外部代码直接访问和修改类的内部状态，确保类的实现细节对外界是不可见的。
class Person:
    def __init__(self, name, age):
        self.__name = name  # 私有变量
        self.__age = age    # 私有变量

    def get_name(self):
        return self.__name

    def get_age(self):
        return self.__age

    def set_age(self, age):
        if age > 0:
            self.__age = age
        else:
            print("Invalid age")

p = Person("Alice", 30)
print(p.get_name())  # 输出: Alice
print(p.get_age())   # 输出: 30
p.set_age(35)
print(p.get_age())   # 输出: 35


# TODO 防止意外修改
class BankAccount:
    def __init__(self, initial_balance):
        self.__balance = initial_balance  # 私有变量

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
        else:
            print("Deposit amount must be positive")

    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
        else:
            print("Invalid withdrawal amount")

    def get_balance(self):
        return self.__balance

account = BankAccount(1000)
account.deposit(500)
print(account.get_balance())  # 输出: 1500
account.withdraw(200)
print(account.get_balance())  # 输出: 1300

# TODO 限制访问权限
class Example:
    def __init__(self):
        self.__private_var = 42  # 私有变量

    def get_private_var(self):
        return self.__private_var

e = Example()
print(e.get_private_var())  # 输出: 42

# 尝试直接访问私有变量
try:
    print(e.__private_var)
except AttributeError as error:
    print(error)  # 输出: 'Example' object has no attribute '__private_var'

# 通过名称改写机制访问（不推荐）
print(e._Example__private_var)  # 输出: 42


# TODO 提供控制访问的方法
class Employee:
    def __init__(self, name, salary):
        self.__name = name  # 私有变量
        self.__salary = salary  # 私有变量

    def get_salary(self):
        return self.__salary

    def set_salary(self, salary):
        if salary > 0:
            self.__salary = salary
        else:
            print("Salary must be positive")

emp = Employee("John", 5000)
print(emp.get_salary())  # 输出: 5000
emp.set_salary(6000)
print(emp.get_salary())  # 输出: 6000