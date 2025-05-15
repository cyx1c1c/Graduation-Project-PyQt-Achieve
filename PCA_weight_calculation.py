# 03运行PCA并计算各指标的权重分配（修正版）
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def pca_weight_calculation(input_path, var_threshold=0.8):
    # 1. 加载标准化后的数据（假设已完成方向调整）
    try:
        df_scaled = pd.read_excel(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {input_path} 未找到，请检查路径")

    # 2. 提取数值列（确保排除非数值列）
    required_columns = ['PSNR', 'SSIM', 'LPIPS', 'MAE', 'STD', 'GCF']
    missing_cols = [col for col in required_columns if col not in df_scaled.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少必要列: {missing_cols}")

    X = df_scaled[required_columns]

    # 3. 验证标准化结果（可选）
    """
    if not np.allclose(X.mean(axis=0), 0, atol=1e-2) or not np.allclose(X.std(axis=0), 1, atol=1e-2):
        print("警告：数据未正确标准化！建议检查预处理步骤")
    """

    # 4. 运行PCA
    pca = PCA()
    pca.fit(X)

    # 动态选择主成分
    cumulative_var = pca.explained_variance_ratio_.cumsum()
    n_components = np.argmax(cumulative_var >= var_threshold) + 1

    # 计算科学权重
    loadings = pca.components_.T
    weighted_loadings = loadings[:, :n_components] * pca.explained_variance_ratio_[:n_components]
    weights = np.abs(weighted_loadings.sum(axis=1))
    weights = 100 * weights / weights.sum()

    # 构建结果
    weights_df = pd.DataFrame({
        '指标': df_scaled.columns,
        '权重 (%)': weights.round(1)
    }).sort_values('权重 (%)', ascending=False)

    # 输出分析报告
    print(f"\n主成分选择: 前{n_components}个成分（累计解释率 {cumulative_var[n_components - 1]:.1%})")
    print("载荷矩阵摘要:")
    print(pd.DataFrame(pca.components_.T, columns=[f'PC{i + 1}' for i in range(6)],
                       index=df_scaled.columns).round(3))

    return weights_df


if __name__ == "__main__":
    # 输入输出路径配置
    input_file = "data/02data_processed.xlsx"  # 标准化后数据路径
    output_file = "data/03pca_weights.xlsx"  # 权重结果保存路径

    # 执行分析
    weights_df = pca_weight_calculation(input_file)
    print("\n最终权重分配:")
    print(weights_df)
    weights_df.to_excel(output_file, index=False)