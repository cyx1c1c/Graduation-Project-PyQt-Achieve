# 02数据标准化（修正版）
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize_data(input_path, output_path):
    # 读取数据
    df = pd.read_excel(input_path)

    # 提取数值列（确保排除非数值列）
    numeric_cols = ['PSNR', 'SSIM', 'LPIPS', 'MAE', 'STD', 'GCF']
    X = df[numeric_cols]

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 方向调整（LPIPS和MAE越小越好 → 取反）
    X_scaled[:, 2] = -X_scaled[:, 2]  # LPIPS
    X_scaled[:, 3] = -X_scaled[:, 3]  # MAE

    # 构建标准化数据框
    df_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)

    # 增强验证逻辑
    mean_check = np.allclose(df_scaled.mean(), 0, atol=0.05)
    std_check = np.allclose(df_scaled.std(), 1, atol=0.1)

    if not (mean_check and std_check):
        print("标准化验证结果:")
        print("均值:", df_scaled.mean().round(2))
        print("标准差:", df_scaled.std().round(2))
        if abs(df_scaled.mean()).max() > 0.1:
            raise ValueError("标准化均值偏差过大，请检查原始数据分布")
    else:
        print("标准化验证通过")

    # 保存结果
    df_scaled.to_excel(output_path, index=False)
    return df_scaled


if __name__ == "__main__":
    input_file = "data/realistic_enhancement_data_250430.xlsx"
    output_file = "data/02data_processed_250430.xlsx"
    df_scaled = standardize_data(input_file, output_file)