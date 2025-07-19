import numpy as np

# 1. 32-bit integer
dtype_int32 = np.dtype(np.int32)
print("32-bit integer:", dtype_int32)  # 输出: dtype('int32')

# 2. 128-bit complex floating-point number
dtype_complex128 = np.dtype(np.complex128)
print("128-bit complex floating-point number:", dtype_complex128)  # 输出: dtype('complex128')

# 3. Python-compatible floating-point number
dtype_float = np.dtype(float)
print("Python-compatible floating-point number:", dtype_float)  # 输出: dtype('float64')

# 4. Python-compatible integer
dtype_int = np.dtype(int)
print("Python-compatible integer:", dtype_int)  # 输出: dtype('int64')

# 5. Python object
dtype_object = np.dtype(object)
print("Python object:", dtype_object)  # 输出: dtype('O')

# 6. Byte, native byte order
dtype_byte = np.dtype('b')
print("Byte, native byte order:", dtype_byte)  # 输出: dtype('S1')

# 7. Big-endian unsigned short
dtype_big_endian_ushort = np.dtype('>H')
print("Big-endian unsigned short:", dtype_big_endian_ushort)  # 输出: dtype('>u2')

# 8. Little-endian single-precision float
dtype_little_endian_float = np.dtype('<f')
print("Little-endian single-precision float:", dtype_little_endian_float)  # 输出: dtype('<f4')

# 9. Double-precision floating-point number
dtype_double = np.dtype('d')
print("Double-precision floating-point number:", dtype_double)  # 输出: dtype('float64')

# 10. 32-bit signed integer
dtype_signed_int32 = np.dtype('i4')
print("32-bit signed integer:", dtype_signed_int32)  # 输出: dtype('int32')

# 11. 64-bit floating-point number
dtype_float64 = np.dtype('f8')
print("64-bit floating-point number:", dtype_float64)  # 输出: dtype('float64')

# 12. 128-bit complex floating-point number
dtype_complex = np.dtype('c16')
print("128-bit complex floating-point number:", dtype_complex)  # 输出: dtype('complex128')

# 13. 25-length zero-terminated bytes
dtype_bytes = np.dtype('a25')
print("25-length zero-terminated bytes:", dtype_bytes)  # 输出: dtype('S25')

# 14. 25-character string
dtype_string = np.dtype('U25')
print("25-character string:", dtype_string)  # 输出: dtype('<U25')

# 15. 32-bit unsigned integer
dtype_uint32 = np.dtype('uint32')
print("32-bit unsigned integer:", dtype_uint32)  # 输出: dtype('uint32')

# 16. 64-bit floating-point number (重复)
dtype_float64_repeat = np.dtype('float64')
print("64-bit floating-point number (重复):", dtype_float64_repeat)  # 输出: dtype('float64')

# 17. 10-byte wide data block
dtype_void = np.dtype((np.void, 10))
print("10-byte wide data block:", dtype_void)  # 输出: dtype('V10')

# 18. 10-character unicode string
dtype_unicode = np.dtype(('U', 10))
print("10-character unicode string:", dtype_unicode)  # 输出: dtype('<U10')

# 19. 2 x 2 integer sub-array
dtype_subarray_2x2 = np.dtype((np.int32, (2, 2)))
print("2 x 2 integer sub-array:", dtype_subarray_2x2)  # 输出: dtype((numpy.int32, (2, 2)))

# 20. 2 x 3 structured sub-array
dtype_structured_subarray = np.dtype(('i4, (2,3)f8', (2, 3)))
print("2 x 3 structured sub-array:", dtype_structured_subarray)  # 输出: dtype((numpy.int32, (2, 3)))

# 21. 结构化类型，包含 big 和 little 字段
dtype_big_little = np.dtype([('big', '>i4'), ('little', '<i4')])
print("Big-endian和little-endian结构化类型:", dtype_big_little)  # 输出: dtype([('big', '>i4'), ('little', '<i4')])

# 22. RGBA 颜色结构化类型
dtype_rgba = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')])
print("RGBA 颜色结构化类型:", dtype_rgba)  # 输出: dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')])

# 23. 使用字典定义 RGBA 颜色
dtype_rgba_dict = np.dtype({'names': ['r', 'g', 'b', 'a'],
                             'formats': [np.uint8, np.uint8, np.uint8, np.uint8]})
print("使用字典定义的RGBA颜色:", dtype_rgba_dict)  # 输出: dtype([('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')])

# 24. 带偏移量和标题的结构化类型
dtype_offset_title = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'],
                                'offsets': [0, 2],
                                'titles': ['Red pixel', 'Blue pixel']})
print("带偏移量和标题的结构化类型:", dtype_offset_title)  # 输出: dtype([('r', 'u1'), ('b', 'u1')])

# 25. 复杂结构化类型
dtype_complex_struct = np.dtype({'col1': ('U10', 0), 'col2': (np.float32, 10),
                                  'col3': (int, 14)})
print("复杂结构化类型:", dtype_complex_struct)  # 输出: dtype({'col1': ('U10', 0), 'col2': ('f4', 10), 'col3': ('i8', 14)})

# 26. 32-bit integer with real and imaginary parts
dtype_real_imag = np.dtype((np.int32, {'real': (np.int16, 0), 'imag': (np.int16, 2)}))
print("32-bit integer with real and imaginary parts:", dtype_real_imag)  # 输出: dtype((numpy.int32, [('real', 'i2'), ('imag', 'i2')]))

# 27. 32-bit integer interpreted as a sub-array of 8-bit integers
dtype_subarray_int8 = np.dtype((np.int32, (np.int8, 4)))
print("32-bit integer interpreted as sub-array of 8-bit integers:", dtype_subarray_int8)  # 输出: dtype((numpy.int32, (numpy.int8, 4)))

# 28. 32-bit integer with RGBA fields
dtype_int_rgba = np.dtype(('i4', [('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')]))
print("32-bit integer with RGBA fields:", dtype_int_rgba)  # 输出: dtype((numpy.int32, [('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')]))
