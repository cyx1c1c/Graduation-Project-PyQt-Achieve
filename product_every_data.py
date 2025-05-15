# 用于写客观指标对比时对各个模型算法的图像计算其图像质量指标（PSNR、SSIM、LPIPS、GCF）
import os
import numpy as np
import torch
import lpips
from skimage import io
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import pandas as pd

# 加载 LPIPS 评估模型
lpips_model = lpips.LPIPS(net='alex')

# 定义每个指标的权重（你提供的）
weights = {
    'LPIPS': 31.7,
    'SSIM': 26.4,
    'GCF': 19.2,
    'PSNR': 14.6,
    'MAE': 4.9,
    'STD': 3.2
}

def calculate_metrics(gt_path, pred_path):
    """ 计算单张图像的所有指标 """
    # 读入图像并归一化
    gt = io.imread(gt_path) / 255.0
    pred = io.imread(pred_path) / 255.0

    if gt.shape[-1] == 4:
        gt = gt[..., :3]
    if pred.shape[-1] == 4:
        pred = pred[..., :3]
    if gt.shape != pred.shape:
        raise ValueError(f"尺寸不一致: {gt.shape} vs {pred.shape}")

    # PSNR
    psnr_val = compare_psnr(gt, pred, data_range=1.0)

    # SSIM
    ssim_val = compare_ssim(gt, pred, data_range=1.0, channel_axis=-1)

    # LPIPS
    gt_tensor = torch.tensor(gt).permute(2,0,1).unsqueeze(0).float()
    pred_tensor = torch.tensor(pred).permute(2,0,1).unsqueeze(0).float()
    lpips_val = lpips_model(gt_tensor, pred_tensor).item()

    # MAE（平均绝对误差）
    mae_val = np.mean(np.abs(gt - pred))

    # STD（标准差）
    std_val = np.std(pred)

    # GCF（全局对比度因子）
    gcf_val = np.mean(np.abs(pred - np.mean(pred)))

    return psnr_val, ssim_val, lpips_val, gcf_val, mae_val, std_val

def compute_p_score(row):
    """ 根据给定权重计算单张图像的P-Score """
    # 注意：LPIPS 和 MAE 是越小越好，需要用 (1-值) 处理
    # SSIM, GCF, PSNR, STD 是越大越好，直接用
    p_score = (
        (1 - row['LPIPS']) * weights['LPIPS'] +
        row['SSIM'] * weights['SSIM'] +
        row['GCF'] * weights['GCF'] +
        row['PSNR'] * weights['PSNR'] +
        (1 - row['MAE']) * weights['MAE'] +
        row['STD'] * weights['STD']
    )
    return p_score

# ==== 批量处理 ====

# 设置路径
gt_dir = 'data/ground_truth_only100/本章算法'
pred_dir = 'data/predicted_images'

# 图片文件列表
image_list = os.listdir(gt_dir)

# 保存所有结果
results = []

for img_name in image_list:
    gt_path = os.path.join(gt_dir, img_name)
    pred_path = os.path.join(pred_dir, img_name)
    try:
        psnr_val, ssim_val, lpips_val, gcf_val, mae_val, std_val = calculate_metrics(gt_path, pred_path)
        results.append({
            'Image': img_name,
            'PSNR': psnr_val,
            'SSIM': ssim_val,
            'LPIPS': lpips_val,
            'GCF': gcf_val,
            'MAE': mae_val,
            'STD': std_val
        })
    except Exception as e:
        print(f"处理 {img_name} 时出错：{e}")

# 转为DataFrame
df = pd.DataFrame(results)

# 计算 P-Score 综合分
df['P-Score'] = df.apply(compute_p_score, axis=1)

# 打印结果
print(df)

# 保存结果
df.to_excel(gt_dir + '/增强算法评价指标带PScore.xlsx', index=False)
print('已保存为 增强算法评价指标带PScore.xlsx')