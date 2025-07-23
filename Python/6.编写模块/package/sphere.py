# sphere.py

"""
This is a sample module to demonstrate
module creation in Python.
It contains a function, a constant, and
a class related to geometrical calculations.
"""

import math

__all__ = ['circle_area', 'PI', 'Sphere']

PI = 3.14159


def circle_area(radius):
    """Calculate the area of a circle."""
    return PI * radius ** 2


class Sphere:
    """A class representing a sphere."""

    def __init__(self, radius):
        self.radius = radius

    def volume(self):
        """Calculate the volume of the sphere. V=4/3 πr³"""
        return (4 / 3) * PI * self.radius ** 3

# TODO 使用 if __name__ == "__main__" 可以使模块在被直接运行时执行特定代码，而在被导入时不执行
if __name__ == "__main__":
    print(f"Circle area (r=5): {circle_area(5)}")
    s = Sphere(3)
    print(f"Sphere volume (r=3): {s.volume()}")
