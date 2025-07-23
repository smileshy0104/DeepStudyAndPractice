# TODO 使用包中的模块
# use_package.py
from mypackage import module1
from mypackage.subpackage import module3

greeting = module1.greet("Alice")
welcome_message = module3.welcome("Bob")

print(greeting)          # 输出 "Hello, Alice!"
print(welcome_message)   # 输出 "Welcome, Bob!"
