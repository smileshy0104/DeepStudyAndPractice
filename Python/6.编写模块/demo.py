import cal
from cal import sub

# from package.module_a import m1
# from package import module_b
# import module_x

# 加
cal.add(1, 1)

# 减
sub(2, 1)


# demo.py

from package.sphere import circle_area, Sphere

# 计算圆的面积
area = circle_area(5)
print(f"圆的面积 (r=5): {area}")

# 创建球体对象并计算体积
sphere = Sphere(3)
volume = sphere.volume()
print(f"球体体积 (r=3): {volume}")
