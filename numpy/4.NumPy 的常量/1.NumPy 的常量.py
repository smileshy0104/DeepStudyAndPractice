import numpy as np

# 打印常量值
print("e的值:", np.e)
print("e的值type:", type(np.e))
print("π的值:", np.pi)
print("π的值type:", type(np.pi))
print("正无穷大:", np.inf)
print("正无穷大type:", type(np.inf))
print("NaN的值:", np.nan)
print("NaN的值type:", type(np.nan))
print("负无穷大:", np.NINF)
print("负无穷大type:", type(np.NINF))
print("正零:", np.PZERO)
print("正零type:", type(np.PZERO))
print("负零:", np.NZERO)
print("负零type:", type(np.NZERO))
print("欧拉常数:", np.euler_gamma)
print("欧拉常数type:", type(np.euler_gamma))

# 使用 np.newaxis 扩展数组维度
array = np.array([1, 2, 3])
new_array = array[:, np.newaxis]
print("原数组:", array)
print("增加维度后的数组:", new_array)


a = np.array([np.inf, -np.inf, 1])

# 显示哪些元素是正无穷大或负无穷大
print(np.isinf(a)) # array([ True,  True, False])
# 显示哪些元素为正无穷大
print(np.isposinf(a)) # array([ True, False, False])
# 显示哪些元素不是数字
print(np.isnan(a)) # array([False, False, False])
# 显示哪些元素为负无穷大
print(np.isneginf(a)) # array([False,  True, False])
# 显示哪些元素是有限的
print(np.isfinite(a)) # array([False, False,  True])
