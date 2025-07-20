import pandas as pd
import numpy as np

# TODO 1.DataFrame
print('————————DataFrame————————')
df = pd.DataFrame({'国家': ['中国', '美国', '日本'],
                   '地区': ['亚洲', '北美', '亚洲'],
                   '人口': [14.33, 3.29, 1.26],
                   'GDP': [14.22, 21.34, 5.18],
                  })
print(df)

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})
print(df2)

# TODO 2.Series
print('————————Series————————')
print(df['人口'])

gdp = pd.Series([14.22, 21.34, 5.18], name='gdp')
print(gdp)
print('————————df————————')
print(df.describe())
print('————————gdp————————')
print(gdp.describe())
print(df.max())
print(gdp.max())
