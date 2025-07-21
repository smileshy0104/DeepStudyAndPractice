import pandas as pd
import numpy as np

# 从 Excel 文件读取数据
df = pd.read_excel('https://gairuo.com/file/data/team.xlsx')

# 基础统计
print("基础统计信息:")
print("描述性统计:\n", df.describe())
print("\n随机抽取样本:\n", df.sample(3))
print("\n唯一值计数:\n", df['Q1'].value_counts())
print("\n唯一值的数量:", df['Q1'].nunique())
print("\n非缺失值的数量:\n", df.count())

# 统计数据
print("\n统计数据:")
print("众数:\n", df.mode())
print("求和:\n", df.sum())
print("平均值:\n", df.mean())
print("中位数:\n", df.median())
print("方差:\n", df.var())
print("标准差:\n", df.std())
print("标准误差:\n", df.sem())
print("峰度:\n", df.kurt())
print("偏度:\n", df.skew())
print("乘积:\n", df.prod())

# 数据特征
print("\n数据特征:")
print("相关系数:\n", df.corr())
print("自相关系数 (Q1):", df['Q1'].autocorr())
print("与 Q2 的相关系数:\n", df['Q1'].corr(df['Q2']))
print("协方差矩阵:\n", df.cov())
print("相邻变化百分比 (Q1):\n", df['Q1'].pct_change())
print("四分位数:\n", df.quantile([0.25, 0.5, 0.75]))
print("排名 (Q1):\n", df['Q1'].rank())
print("最大值 (Q1):", df['Q1'].max())
print("最小值 (Q1):", df['Q1'].min())
print("绝对值 (Q1):\n", df['Q1'].abs())

# 累积计算
print("\n累积计算:")
print("累积最大值 (Q1):\n", df['Q1'].cummax())
print("累积最小值 (Q1):\n", df['Q1'].cummin())
print("累积乘积 (Q1):\n", df['Q1'].cumprod())
print("累积和 (Q1):\n", df['Q1'].cumsum())

# 自定义相关性函数示例
def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v

# 创建一个新的 DataFrame 用于相关性计算
df_corr = pd.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
                       columns=['dogs', 'cats'])
print("\n自定义相关性计算结果:\n", df_corr.corr(method=histogram_intersection))
