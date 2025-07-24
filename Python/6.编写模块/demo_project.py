# TODO 在没有 __init__.py 的情况下，确保 project_a 和 project_b 的路径都在 PYTHONPATH 中：
import mypackage.module1
import mypackage.module2

mypackage.module1.func1()  # 输出: Function 1 from project A
mypackage.module2.func2()  # 输出: Function 2 from project B