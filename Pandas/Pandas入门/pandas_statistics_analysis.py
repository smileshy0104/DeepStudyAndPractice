#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pandas统计分析完全演示
基于《Pandas高级操作完全指南》的"统计分析"章节

作者: AI Assistant
日期: 2024
"""

import pandas as pd
import numpy as np

def main():
    print("=== Pandas 统计分析完全演示 ===\n")

    # 创建员工数据
    df = pd.DataFrame({
        '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八'],
        '年龄': [25, 30, 35, 28, 32, 27],
        '部门': ['技术', '销售', '技术', '市场', '技术', '销售'],
        '工资': [8000, 12000, 15000, 10000, 13000, 11000],
        '城市': ['北京', '上海', '广州', '深圳', '杭州', '成都']
    }, index=['emp1', 'emp2', 'emp3', 'emp4', 'emp5', 'emp6'])

    print("员工数据:")
    print(df)
    print(f"\n数据形状: {df.shape}")
    print(f"数据类型:\n{df.dtypes}")

    # ==================== 4.1 基础统计计算 ====================
    print("\n" + "="*60)
    print("4.1 基础统计计算")
    print("="*60)

    # 单列统计
    print("\n--- 单列统计 ---")

    print("1. 年龄统计:")
    age_stats = df['年龄'].describe()
    print(age_stats)

    print("\n2. 工资统计:")
    salary_stats = df['工资'].describe()
    print(salary_stats)

    # 手动计算各项统计指标
    print("\n--- 手动计算统计指标 ---")

    age_series = df['年龄']
    print("年龄 - 手动统计:")
    print(f"  计数: {age_series.count()}")
    print(f"  均值: {age_series.mean():.2f}")
    print(f"  中位数: {age_series.median():.2f}")
    print(f"  标准差: {age_series.std():.2f}")
    print(f"  方差: {age_series.var():.2f}")
    print(f"  最小值: {age_series.min()}")
    print(f"  最大值: {age_series.max()}")
    print(f"  极差: {age_series.max() - age_series.min()}")
    print(f"  25%分位数: {age_series.quantile(0.25):.2f}")
    print(f"  75%分位数: {age_series.quantile(0.75):.2f}")
    print(f"  四分位距: {age_series.quantile(0.75) - age_series.quantile(0.25):.2f}")

    # 自定义统计函数
    print("\n--- 自定义统计函数 ---")

    def custom_stats(series, name="数据"):
        """自定义统计函数"""
        stats_dict = {
            '计数': series.count(),
            '总和': series.sum(),
            '均值': series.mean(),
            '中位数': series.median(),
            '众数': series.mode().iloc[0] if not series.mode().empty else np.nan,
            '标准差': series.std(),
            '方差': series.var(),
            '最小值': series.min(),
            '最大值': series.max(),
            '极差': series.max() - series.min(),
            '四分位距': series.quantile(0.75) - series.quantile(0.25),
            '变异系数': series.std() / series.mean() if series.mean() != 0 else 0,
            '偏度': series.skew(),
            '峰度': series.kurtosis()
        }
        return pd.Series(stats_dict, name=name)

    print("年龄自定义统计:")
    print(custom_stats(df['年龄'], "年龄"))
    print("\n工资自定义统计:")
    print(custom_stats(df['工资'], "工资"))

    # 多列同时统计
    print("\n--- 多列同时统计 ---")

    # 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number])
    print("所有数值列统计:")
    print(numeric_cols.describe())

    # 自定义多列统计
    print("\n多列自定义统计:")
    multi_stats = pd.DataFrame({
        '年龄': custom_stats(df['年龄']),
        '工资': custom_stats(df['工资'])
    })
    print(multi_stats.round(2))

    # ==================== 4.2 分组统计 ====================
    print("\n" + "="*60)
    print("4.2 分组统计")
    print("="*60)

    # 按部门分组统计
    print("\n--- 按部门分组统计 ---")

    dept_stats = df.groupby('部门').agg({
        '年龄': ['count', 'mean', 'std', 'min', 'max'],
        '工资': ['mean', 'median', 'min', 'max', 'std']
    }).round(2)

    print("按部门统计的年龄和工资:")
    print(dept_stats)

    # 扁平化多级列名
    print("\n扁平化列名:")
    dept_stats.columns = ['_'.join(col).strip() for col in dept_stats.columns]
    print(dept_stats)

    # 按城市分组统计
    print("\n--- 按城市分组统计 ---")

    city_stats = df.groupby('城市').agg({
        '姓名': 'count',
        '年龄': ['mean', 'min', 'max'],
        '工资': ['mean', 'sum']
    }).round(2)

    city_stats.columns = ['员工数', '平均年龄', '最小年龄', '最大年龄', '平均工资', '总工资']
    print(city_stats)

    # 多级分组统计
    print("\n--- 多级分组统计 ---")

    # 添加年龄段列
    df['年龄段'] = pd.cut(df['年龄'], bins=[20, 25, 30, 35, 40], labels=['20-25', '25-30', '30-35', '35-40'])

    multi_group = df.groupby(['部门', '年龄段']).agg({
        '姓名': 'count',
        '工资': ['mean', 'median']
    }).round(2)

    multi_group.columns = ['员工数', '平均工资', '工资中位数']
    print("部门 + 年龄段分组统计:")
    print(multi_group)

    # 透视表统计
    print("\n--- 透视表统计 ---")

    pivot_table = pd.pivot_table(df,
                                values='工资',
                                index='部门',
                                columns='城市',
                                aggfunc='mean',
                                fill_value=0,
                                margins=True,
                                margins_name='总计')

    print("部门-城市工资透视表:")
    print(pivot_table.round(2))

    # 交叉表
    print("\n--- 交叉表统计 ---")

    cross_tab = pd.crosstab(df['部门'], df['年龄段'], margins=True)
    print("部门-年龄段交叉表:")
    print(cross_tab)

    # 高级分组操作
    print("\n--- 高级分组操作 ---")

    # 自定义聚合函数
    def salary_range(series):
        return f"{series.min()}-{series.max()}"

    custom_group = df.groupby('部门').agg({
        '工资': [salary_range, 'mean', 'std'],
        '年龄': ['count', lambda x: x.mean().round(1)]
    })
    custom_group.columns = ['工资范围', '平均工资', '工资标准差', '员工数', '平均年龄']
    print("自定义聚合函数结果:")
    print(custom_group)

    # ==================== 4.3 高级统计分析 ====================
    print("\n" + "="*60)
    print("4.3 高级统计分析")
    print("="*60)

    # 创建更多示例数据
    print("\n创建扩展数据集...")
    np.random.seed(42)
    extended_df = pd.DataFrame({
        '产品类别': np.random.choice(['A', 'B', 'C'], 100),
        '销售额': np.random.normal(1000, 200, 100),
        '成本': np.random.normal(600, 100, 100),
        '客户满意度': np.random.uniform(3.0, 5.0, 100),
        '员工ID': np.random.choice(['E001', 'E002', 'E003', 'E004'], 100)
    })

    # 计算利润率
    extended_df['利润率'] = (extended_df['销售额'] - extended_df['成本']) / extended_df['销售额'] * 100
    extended_df['利润'] = extended_df['销售额'] - extended_df['成本']

    print("扩展数据样本 (前10行):")
    print(extended_df.head(10))

    # 按产品类别的详细统计
    print("\n--- 按产品类别的详细统计 ---")

    detailed_stats = extended_df.groupby('产品类别').agg({
        '销售额': ['count', 'mean', 'std', 'min', 'max'],
        '成本': ['mean', 'std'],
        '利润率': ['mean', 'std'],
        '客户满意度': ['mean', 'std']
    }).round(2)

    print("产品类别详细统计:")
    print(detailed_stats)

    # 相关性分析
    print("\n--- 相关性分析 ---")

    numeric_columns = ['销售额', '成本', '利润率', '客户满意度', '利润']
    correlation_matrix = extended_df[numeric_columns].corr()

    print("数值变量相关系数矩阵:")
    print(correlation_matrix.round(3))

    # 找出最强的相关关系
    print("\n最强相关关系:")
    max_corr = 0
    max_pair = ('', '')
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > abs(max_corr):
                max_corr = corr_val
                max_pair = (correlation_matrix.columns[i], correlation_matrix.columns[j])

    print(f"最高相关性: {max_pair[0]} 和 {max_pair[1]}, 相关系数: {max_corr:.3f}")

    # 置信区间计算
    print("\n--- 置信区间计算 ---")

    def confidence_interval(series, confidence=0.95):
        """计算置信区间"""
        try:
            from scipy import stats
            n = len(series)
            mean = series.mean()
            std_err = stats.sem(series)
            h = std_err * stats.t.ppf((1 + confidence) / 2, n-1)
            return (mean - h, mean + h)
        except ImportError:
            # 如果没有scipy，使用正态分布近似
            n = len(series)
            mean = series.mean()
            std_err = series.std() / np.sqrt(n)
            h = std_err * 1.96  # 95%置信区间对应的Z值
            return (mean - h, mean + h)

    print("95%置信区间计算:")
    for category in extended_df['产品类别'].unique():
        sales_data = extended_df[extended_df['产品类别'] == category]['销售额']
        ci_lower, ci_upper = confidence_interval(sales_data)
        print(f"{category} 销售额 95% 置信区间: ({ci_lower:.2f}, {ci_upper:.2f})")

    # 异常值检测
    print("\n--- 异常值检测 ---")

    def detect_outliers_iqr(series):
        """使用IQR方法检测异常值"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return outliers, lower_bound, upper_bound

    def detect_outliers_zscore(series, threshold=3):
        """使用Z-score方法检测异常值"""
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = series[z_scores > threshold]
        return outliers

    print("异常值检测 (IQR 方法):")
    for category in extended_df['产品类别'].unique():
        category_data = extended_df[extended_df['产品类别'] == category]
        outliers, lower, upper = detect_outliers_iqr(category_data['销售额'])
        print(f"{category} 销售额:")
        print(f"  正常范围: ({lower:.2f}, {upper:.2f})")
        print(f"  异常值数量: {len(outliers)}")
        if len(outliers) > 0:
            print(f"  异常值: {outliers.values}")

    print("\n异常值检测 (Z-score 方法):")
    for category in extended_df['产品类别'].unique():
        category_data = extended_df[extended_df['产品类别'] == category]
        outliers = detect_outliers_zscore(category_data['销售额'])
        print(f"{category} 销售额异常值 (Z-score > 3): {len(outliers)} 个")

    # 分布统计
    print("\n--- 分布统计 ---")

    def distribution_stats(series, name="数据"):
        """计算分布统计"""
        stats_dict = {
            '均值': series.mean(),
            '中位数': series.median(),
            '众数': series.mode().iloc[0] if not series.mode().empty else np.nan,
            '标准差': series.std(),
            '偏度': series.skew(),
            '峰度': series.kurtosis(),
            '变异系数': series.std() / series.mean() if series.mean() != 0 else 0,
            '范围': series.max() - series.min(),
            '四分位距': series.quantile(0.75) - series.quantile(0.25)
        }

        # 分布类型判断
        skewness = stats_dict['偏度']
        if abs(skewness) < 0.5:
            distribution = "近似正态分布"
        elif skewness > 0.5:
            distribution = "右偏分布"
        else:
            distribution = "左偏分布"

        stats_dict['分布类型'] = distribution
        return pd.Series(stats_dict, name=name)

    for category in extended_df['产品类别'].unique():
        category_sales = extended_df[extended_df['产品类别'] == category]['销售额']
        print(f"\n{category} 销售额分布统计:")
        print(distribution_stats(category_sales, f"{category}销售额").round(3))

    # 员工效率分析
    print("\n" + "="*60)
    print("员工效率分析")
    print("="*60)

    # 员工绩效统计
    print("\n--- 员工绩效统计 ---")

    # 计算工资效率指标
    df['工资等级'] = pd.cut(df['工资'],
                           bins=[0, 8000, 12000, 16000],
                           labels=['初级', '中级', '高级'])

    performance_stats = df.groupby(['部门', '工资等级']).agg({
        '年龄': ['count', 'mean'],
        '工资': ['mean', 'std']
    }).round(2)

    performance_stats.columns = ['员工数', '平均年龄', '平均工资', '工资标准差']
    print("部门-工资等级绩效统计:")
    print(performance_stats)

    # 工资分布分析
    print("\n--- 工资分布分析 ---")

    print("整体工资分布:")
    salary_dist = distribution_stats(df['工资'], "工资")
    print(salary_dist.round(2))

    print("\n各部门工资分布:")
    for dept in df['部门'].unique():
        dept_salary = df[df['部门'] == dept]['工资']
        dept_stats = distribution_stats(dept_salary, f"{dept}部门工资")
        print(dept_stats.round(2))

    # 年龄与工资关系分析
    print("\n--- 年龄与工资关系分析 ---")

    # 计算相关系数
    age_salary_corr = df['年龄'].corr(df['工资'])
    print(f"年龄与工资相关系数: {age_salary_corr:.3f}")

    # 按年龄段分析工资
    age_groups = df.groupby('年龄段').agg({
        '工资': ['count', 'mean', 'std', 'min', 'max'],
        '年龄': ['mean', 'min', 'max']
    }).round(2)
    age_groups.columns = ['人数', '平均工资', '工资标准差', '最低工资', '最高工资', '平均年龄', '最小年龄', '最大年龄']
    print("\n年龄段工资分析:")
    print(age_groups)

    # ==================== 实用统计分析工具 ====================
    print("\n" + "="*60)
    print("实用统计分析工具")
    print("="*60)

    # 自动化统计报告函数
    def generate_stats_report(dataframe, group_col=None, value_cols=None):
        """生成统计报告"""
        if value_cols is None:
            value_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()

        report = {}

        if group_col:
            # 分组统计
            for group in dataframe[group_col].unique():
                group_data = dataframe[dataframe[group_col] == group]
                report[group] = {}
                for col in value_cols:
                    if col in group_data.columns:
                        report[group][col] = distribution_stats(group_data[col], col)
        else:
            # 整体统计
            for col in value_cols:
                if col in dataframe.columns:
                    report[col] = distribution_stats(dataframe[col], col)

        return report

    # 生成员工统计报告
    print("\n--- 自动化统计报告 ---")
    employee_report = generate_stats_report(df, group_col='部门', value_cols=['年龄', '工资'])

    for dept, stats in employee_report.items():
        print(f"\n{dept}部门统计:")
        for metric, values in stats.items():
            print(f"  {metric}: 均值={values['均值']:.2f}, 标准差={values['标准差']:.2f}, 分布={values['分布类型']}")

    # 数据质量评估
    print("\n--- 数据质量评估 ---")

    def data_quality_assessment(dataframe):
        """数据质量评估"""
        quality_report = {
            '总行数': len(dataframe),
            '总列数': len(dataframe.columns),
            '缺失值': dataframe.isnull().sum().sum(),
            '重复行': dataframe.duplicated().sum(),
            '数值列数': len(dataframe.select_dtypes(include=[np.number]).columns),
            '文本列数': len(dataframe.select_dtypes(include=['object']).columns),
            '日期列数': len(dataframe.select_dtypes(include=['datetime64']).columns)
        }

        # 计算每列的数据质量
        column_quality = {}
        for col in dataframe.columns:
            col_quality = {
                '缺失值数': dataframe[col].isnull().sum(),
                '缺失值比例': dataframe[col].isnull().sum() / len(dataframe) * 100,
                '唯一值数': dataframe[col].nunique(),
                '数据类型': str(dataframe[col].dtype)
            }
            column_quality[col] = col_quality

        quality_report['列质量'] = column_quality
        return quality_report

    quality_report = data_quality_assessment(df)
    print("数据质量报告:")
    for key, value in quality_report.items():
        if key != '列质量':
            print(f"  {key}: {value}")

    print("\n列质量详情:")
    for col, quality in quality_report['列质量'].items():
        print(f"  {col}: 类型={quality['数据类型']}, 缺失值={quality['缺失值数']}({quality['缺失值比例']:.1f}%), 唯一值={quality['唯一值数']}")

    print("\n" + "="*60)
    print("统计分析演示完成!")
    print("="*60)

    # 总结
    print("\n【统计分析总结】")
    print("✓ 基础统计: describe, 自定义统计函数")
    print("✓ 分组统计: groupby, 透视表, 交叉表")
    print("✓ 高级分析: 相关性, 置信区间, 异常值检测")
    print("✓ 分布分析: 偏度, 峰度, 分布类型判断")
    print("✓ 实用工具: 自动化报告, 数据质量评估")
    print("✓ 可视化准备: 为数据可视化准备统计数据")

    print(f"\n原始员工数据: {df.shape}")
    print(f"扩展产品数据: {extended_df.shape}")
    print("所有统计分析演示均成功完成! 📊")

if __name__ == "__main__":
    main()