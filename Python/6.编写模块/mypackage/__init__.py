# 初始化包
print("mypackage 已加载")

# TODO 包初始化：__init__.py 可以包含包的初始化代码，这些代码会在包被导入时执行。
#  例如，可以在这里初始化包级别的变量、导入子模块等。
# 导入子模块中的函数
from .module1 import func1
from .module2 import func2

__all__ = ['func1', 'func2']