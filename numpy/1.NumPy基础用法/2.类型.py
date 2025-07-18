import numpy as np

# 1. 有符号 64 位 int 类型
int64_type = np.int64
print("np.int64:", int64_type)  # 输出: <class 'numpy.int64'>

# 2. 标准双精度浮点类型
float32_type = np.float32
print("np.float32:", float32_type)  # 输出: <class 'numpy.float32'>

# 3. 由128位的浮点数组成的复数类型
complex_type = np.complex  # np.complex 已被弃用，使用 np.complex128 代替
print("np.complex:", complex_type)  # 输出: <class 'numpy.complex128'>

# 4. TRUE 和 FALSE 的 bool 类型
bool_type = np.bool_
print("np.bool:", bool_type)  # 输出: <class 'numpy.bool_'>

# 5. Python 中的 object 类型
object_type = np.object_
print("np.object:", object_type)  # 输出: <class 'numpy.object_'>

# 6. TODO 固定长度的 string 类型 输出的是 bytes
string_type = np.string_
print("np.string:", string_type)  # 输出: <class 'numpy.bytes_'>

# 7. TODO 固定长度的 unicode 类型 输出的是 str
unicode_type = np.unicode_
print("np.unicode:", unicode_type)  # 输出: <class 'numpy.str_'>

# 8. np.float 的子类型
nan_type = np.NaN
print("np.NaN:", nan_type)  # 输出: nan

# 9. np.nan
nan_value = np.nan
print("np.nan:", nan_value)  # 输出: nan

# 10. np 所有数据类型
all_data_types = np.sctypeDict
print("np.sctypeDict:", all_data_types)  # 输出: 包含所有数据类型的字典
