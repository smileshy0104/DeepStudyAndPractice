# 完整的异常处理示例

# 捕获特定异常
try:
    num = int(input("请输入一个整数: "))
    result = 10 / num
except ValueError:
    print("输入无效，请输入一个整数。")
except ZeroDivisionError:
    print("错误：不能除以零。")
else:
    print("结果:", result)

# 捕获所有异常
try:
    num = int(input("请输入一个整数: "))
    result = 10 / num
    print("结果:", result)
except Exception as e:
    print("发生错误:", e)

# 使用 finally
try:
    file = open("example.txt", "r")
    content = file.read()
    print(content)
except FileNotFoundError:
    print("文件未找到。")
finally:
    if 'file' in locals():
        file.close()
        print("文件已关闭。")

# 自定义异常
class CustomError(Exception):
    pass

try:
    raise CustomError("这是一个自定义异常！")
except CustomError as e:
    print(e)

# 使用 else 语句
try:
    num = int(input("请输入一个整数: "))
    result = 10 / num
except ValueError:
    print("输入无效，请输入一个整数。")
except ZeroDivisionError:
    print("错误：不能除以零。")
else:
    print("结果:", result)
