import numpy as np

# TODO 1.最小标题类型 np.min_scalar_type
np.min_scalar_type(10)
# dtype('uint8')
np.min_scalar_type(-260)
# dtype('int16')
np.min_scalar_type(3.1)
# dtype('float16')
np.min_scalar_type(1e50)
# dtype('float64')
np.min_scalar_type(np.arange(4,dtype='f8'))
# dtype('float64')