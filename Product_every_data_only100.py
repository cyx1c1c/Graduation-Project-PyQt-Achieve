import os
import numpy as np
import torch
import lpips
from skimage import io
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import pandas as pd

# 评估模型
lpips_model = lpips.LPIPS(net='alex')

# 指标权重（可调）
weights = {
    'LPIPS': 31.7,
    'SSIM': 26.4,
    'GCF': 19.2,
    'PSNR': 14.6,
    'MAE': 4.9,
    'STD': 3.2
}

def calculate_metrics(gt_path, pred_path):
    gt = io.imread(gt_path) / 255.0
    pred = io.imread(pred_path) / 255.0
    if gt.shape[-1] == 4:
        gt = gt[..., :3]
    if pred.shape[-1] == 4:
        pred = pred[..., :3]
    if gt.shape != pred.shape:
        raise ValueError(f"尺寸不一致: {gt.shape} vs {pred.shape}")

    psnr_val = compare_psnr(gt, pred, data_range=1.0)
    ssim_val = compare_ssim(gt, pred, data_range=1.0, channel_axis=-1)

    gt_tensor = torch.tensor(gt).permute(2,0,1).unsqueeze(0).float()
    pred_tensor = torch.tensor(pred).permute(2,0,1).unsqueeze(0).float()
    lpips_val = lpips_model(gt_tensor, pred_tensor).item()

    mae_val = np.mean(np.abs(gt - pred))
    std_val = np.std(pred)
    gcf_val = np.mean(np.abs(pred - np.mean(pred)))

    return psnr_val, ssim_val, lpips_val, gcf_val, mae_val, std_val

def compute_p_score(row):
    return (
        (1 - row['LPIPS']) * weights['LPIPS'] +
        row['SSIM'] * weights['SSIM'] +
        row['GCF'] * weights['GCF'] +
        row['PSNR'] * weights['PSNR'] +
        (1 - row['MAE']) * weights['MAE'] +
        row['STD'] * weights['STD']
    )

# ==== 设置路径 ====
base_dir = 'data/ground_truth_only100'
low_dir = os.path.join(base_dir, 'Enlighten-GAN/images_A')  # 所有算法使用统一 low 图来源

algorithms = {
    'Enlighten-GAN': os.path.join(base_dir, 'Enlighten-GAN/images_B'),
    'LIME': os.path.join(base_dir, 'LIME'),
    'Retinex-Net': os.path.join(base_dir, 'Retinex-Net'),
    '本章算法': os.path.join(base_dir, '本章算法')
}

# ==== 开始遍历每张图，逐算法比较 ====
results = []

for file in os.listdir(low_dir):
    if not file.endswith('_real_A.png'):
        continue

    image_id = file.replace('_real_A.png', '')  # 提取前缀
    gt_path = os.path.join(low_dir, file)  # 原图

    for alg, enh_dir in algorithms.items():
        # 构造增强图路径（命名规则不同）
        if alg == 'Enlighten-GAN':
            enh_name = image_id + '_fake_B.png'
        elif alg == 'LIME':
            enh_name = 'enhanced_' + image_id + '_real_A.png'
        elif alg ==  'Retinex-Net':
            enh_name = image_id + '_real_A.jpg'  # 特殊处理
        else:
            enh_name = image_id + '_real_A.png'

        enh_path = os.path.join(enh_dir, enh_name)

        if not os.path.exists(enh_path):
            print(f"缺失增强图: {enh_path}")
            continue

        try:
            psnr_val, ssim_val, lpips_val, gcf_val, mae_val, std_val = calculate_metrics(gt_path, enh_path)
            p_score = compute_p_score({
                'PSNR': psnr_val,
                'SSIM': ssim_val,
                'LPIPS': lpips_val,
                'GCF': gcf_val,
                'MAE': mae_val,
                'STD': std_val
            })
            results.append({
                'Image': image_id,
                'Algorithm': alg,
                'PSNR': psnr_val,
                'SSIM': ssim_val,
                'LPIPS': lpips_val,
                'GCF': gcf_val,
                'MAE': mae_val,
                'STD': std_val,
                'P-Score': p_score
            })
        except Exception as e:
            print(f"[错误] 处理 {alg} - {image_id}: {e}")

# ==== 保存结果 ====
df = pd.DataFrame(results)
df = df.sort_values(by=['Image', 'P-Score'], ascending=[True, False])
print(df)
df.to_excel(base_dir + '/多算法PScore评估结果.xlsx', index=False)
print('✅ 评估完成，已保存：多算法PScore评估结果.xlsx')