# 03数据标准化验证
import pandas as pd

df_scaled = pd.read_excel("data/02data_processed.xlsx")
print("均值验证:", df_scaled.mean().round(2))  # 应接近 [0, 0, 0, 0, 0, 0]
print("标准差验证:", df_scaled.std().round(2))  # 应接近 [1, 1, 1, 1, 1, 1]