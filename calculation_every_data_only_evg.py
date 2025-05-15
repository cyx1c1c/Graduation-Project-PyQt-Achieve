#  计算每个算法在所有图像上的平均指标值
import pandas as pd

# 读取你刚生成的Excel文件
df = pd.read_excel("data/ground_truth_only100/多算法PScore评估结果.xlsx")

# 分组求平均
mean_df = df.groupby("Algorithm").mean(numeric_only=True)

# 打印查看结果
print(mean_df)

# 如果你想导出到新的 Excel 文件方便引用：
mean_df.to_excel("data/ground_truth_only100/各算法平均指标.xlsx")