# 改进：在10次实验模型基础上
# 添加【边缘感知损失函数（Sobel边缘差异）】
import os
import time
import random

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchvision import models

lpips_model = lpips.LPIPS(net='alex')  # 选择 AlexNet 作为 LPIPS 计算模型


class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[
            0]  # [B, 3, H, W]-->[B, 1, H, W]  在 RGB 通道维度（dim=1）取最大值，得到单通道的亮度图
        input_img = torch.cat((input_max, input_im), dim=1)  # [B, 1, H, W]-->[B, 4, H, W]  将亮度图与原图拼接，形成 4 通道输入
        feats0 = self.net1_conv0(input_img)  # [B, 4, H, W] --> [B, 64, H, W]  浅层特征提取
        featss = self.net1_convs(feats0)  # [B, 64, H, W] --> [B, 64, H, W]  深层特征处理
        outs = self.net1_recon(featss)  # [B, 64, H, W] --> [B, 4, H, W]  分解结果重建
        R = torch.sigmoid(outs[:, 0:3, :, :])  # [B, 4, H, W] --> [B, 3, H, W]
        L = torch.sigmoid(outs[:, 3:4, :, :])  # [B, 4, H, W] --> [B, 1, H, W]
        return R, L


class PerceptualLoss(nn.Module):
    """ 计算 VGG19 特征层损失 """

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16]  # 仅使用前 16 层
        self.vgg = nn.Sequential(*vgg).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        return F.l1_loss(self.vgg(x), self.vgg(y))


class MSSSIMLoss(nn.Module):
    """ 多尺度结构相似性损失 """

    def __init__(self):
        super(MSSSIMLoss, self).__init__()

    def forward(self, x, y):
        return 1 - torch.mean(torch.clamp(F.mse_loss(x, y), min=1e-6))


class IlluminationSmoothnessLoss(nn.Module):
    """ 约束照度变化的梯度，使亮度变化平滑 """

    def __init__(self):
        super(IlluminationSmoothnessLoss, self).__init__()

    def forward(self, I):
        dx = torch.abs(I[:, :, :, :-1] - I[:, :, :, 1:])
        dy = torch.abs(I[:, :, :-1, :] - I[:, :, 1:, :])
        return torch.mean(dx) + torch.mean(dy)


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        sobel_kernel_x = torch.tensor([[[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[[-1, -2, -1],
                                        [0, 0, 0],
                                        [1, 2, 1]]], dtype=torch.float32).unsqueeze(0)

        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def forward(self, pred, target):
        # pred, target: [B, 1, H, W]
        grad_pred_x = self.sobel_x(pred)
        grad_pred_y = self.sobel_y(pred)
        grad_target_x = self.sobel_y(target)
        grad_target_y = self.sobel_y(target)
        grad_pred = torch.sqrt(grad_pred_x ** 2 + grad_pred_y ** 2)
        grad_target = torch.sqrt(grad_target_x ** 2 + grad_target_y ** 2)
        return F.l1_loss(grad_pred, grad_target)



# 新增CBAM模块（在model.py中添加）
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出形状: [B, C, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 输出形状: [B, C, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel)
        )
        # 空间注意力
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)  # 输入形状: [B, 2, H, W]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入x形状: [B, C, H, W]

        # ----------------- 通道注意力 -----------------
        avg_out = self.avg_pool(x)  # [B, C, 1, 1]
        avg_out = avg_out.view(avg_out.size(0), -1)  # [B, C] 显式保留批次维度
        avg_out = self.fc(avg_out)  # [B, C]

        max_out = self.max_pool(x)  # [B, C, 1, 1]
        max_out = max_out.view(max_out.size(0), -1)  # [B, C] 显式保留批次维度
        max_out = self.fc(max_out)  # [B, C]

        channel_att = self.sigmoid(avg_out + max_out)  # [B, C]
        channel_att = channel_att.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # ----------------- 空间注意力 -----------------
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_att = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        spatial_att = self.conv(spatial_att)  # [B, 1, H, W]
        spatial_att = self.sigmoid(spatial_att)  # [B, 1, H, W]

        # ----------------- 综合注意力 -----------------
        return x * channel_att * spatial_att  # 输出形状与输入x一致: [B, C, H, W]

# 修改HSBlock，替换SE为CBAM
class HSBlock(nn.Module):
    """ 改进 HSBlock，加入 Squeeze-and-Excitation (SE Block) """
    def __init__(self, channel):
        super(HSBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.cbam = CBAM(channel)  # 使用修复后的CBAM

    def forward(self, x):
        x1 = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x)))))  # 输入输出形状一致: [B, C, H, W]
        x1 = self.cbam(x1)  # 输入输出形状一致: [B, C, H, W]
        return x1


class CrossAttention(nn.Module):
    def __init__(self, channel):
        super(CrossAttention, self).__init__()
        # 新增金字塔池化层
        self.pyramid_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 输出形状：[B, C, 1, 1]
            nn.AdaptiveAvgPool2d(2),  # 输出形状：[B, C, 2, 2]
            nn.AdaptiveAvgPool2d(4)  # 输出形状：[B, C, 4, 4]
        )
        self.query_conv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(channel * 4, channel // 2, kernel_size=1)  # 输入通道从64→256
        self.value_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
        x1: input feature map 1  [B, C, H, W] (e.g. [16,64,12,12])
        x2: input feature map 2 (to interact with x1)  [B, C, H, W] (e.g. [16,64,12,12])
        """
        # Step 1: 多尺度池化
        pooled = []
        for pool in self.pyramid_pool:
            pooled_feat = pool(x2)  # 不同尺度池化
            pooled_feat = F.interpolate(pooled_feat, size=x2.shape[2:])  # 上采样回原尺寸
            pooled.append(pooled_feat)
        # 拼接后通道数：64 (x2) + 64×3 (pooled) = 256
        x2_pooled = torch.cat([x2] + pooled, dim=1)  # [B, 256, H, W] (e.g. [16,256,12,12])

        # Step 2: 调整后的卷积层处理
        batch_size, C, H, W = x1.shape
        query = self.query_conv(x1).view(batch_size, -1, H * W).permute(0, 2, 1)  # [B, HW, C//2]
        key = self.key_conv(x2_pooled).view(batch_size, -1, H * W)  # [B, C//2, HW] (输入256→输出32)
        value = self.value_conv(x2).view(batch_size, -1, H * W)  # [B, C, HW] (输入64→输出64)

        # 后续计算保持不变
        attention_map = self.softmax(torch.bmm(query, key))  # [B, HW, HW]
        out = torch.bmm(value, attention_map.permute(0, 2, 1)).view(batch_size, C, H, W)

        return out + x1  # 残差连接


# 新增SKNet模块（在model.py中添加）
class SKConv(nn.Module):
    def __init__(self, channel, reduction=16, M=2):
        super(SKConv, self).__init__()
        self.M = M
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel * M),
            nn.Softmax(dim=1)
        )
        self.convs = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1+i, dilation=1+i, groups=32)
            for i in range(M)])

    def forward(self, x):
        batch_size = x.size(0)
        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, x.size(1), x.size(2), x.size(3))
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U).view(batch_size, -1)
        feats_Z = self.fc(feats_S).view(batch_size, self.M, x.size(1), 1, 1)
        feats_V = torch.sum(feats * feats_Z, dim=1)
        return feats_V


class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        self.relu = nn.ReLU()

        # 初始卷积
        self.initial_conv = nn.Conv2d(4, channel, kernel_size, padding=1, padding_mode='replicate')  # [B,4,H,W] -> [B,64,H,W]
        self.cbam_0 = CBAM(channel)

        # 下采样块（Encoder）
        self.down1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            SKConv(channel),
            CBAM(channel)
        )  # [B,64,H,W] -> [B,64,H/2,W/2]

        self.down2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            SKConv(channel),
            CBAM(channel)
        )  # [B,64,H/2,W/2] -> [B,64,H/4,W/4]

        self.down3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            SKConv(channel),
            CBAM(channel)
        )  # [B,64,H/4,W/4] -> [B,64,H/8,W/8]

        # 深层交叉注意力模块（原CrossAttention + 金字塔池化）
        self.cross_att1 = CrossAttention(channel)
        self.cross_att2 = CrossAttention(channel)

        # Dilated卷积增强上下文
        self.dilated1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2, padding_mode='replicate')
        self.dilated2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3, padding_mode='replicate')

        # 上采样块（Decoder）
        self.up2 = nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1, padding_mode='replicate')  # skip from down2
        self.up1 = nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1, padding_mode='replicate')  # skip from down1
        self.up0 = nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1, padding_mode='replicate')  # skip from input

        # 多尺度融合
        self.fuse = nn.Conv2d(channel * 3, channel, kernel_size=1, padding=0)
        self.output_conv = nn.Conv2d(channel, 1, kernel_size=3, padding=1)

    def forward(self, input_L, input_R):
        """
        input_L: [B, 1, H, W] 光照图
        input_R: [B, 3, H, W] 反射图
        return: [B, 1, H, W] 增强后光照图
        """
        input = torch.cat([input_R, input_L], dim=1)  # [B,4,H,W]
        x0 = self.cbam_0(self.relu(self.initial_conv(input)))  # [B,64,H,W]

        x1 = self.down1(x0)  # [B,64,H/2,W/2]
        x2 = self.down2(x1)  # [B,64,H/4,W/4]
        x3 = self.down3(x2)  # [B,64,H/8,W/8]

        # 深层特征增强
        x3_1 = self.relu(self.dilated1(x3))  # [B,64,H/8,W/8]
        x3_2 = self.relu(self.dilated2(x3))  # [B,64,H/8,W/8]
        att3 = self.cross_att1(x3_1, x3_2)  # [B,64,H/8,W/8]

        # 上采样阶段
        up2 = F.interpolate(att3, size=x2.shape[2:], mode='bilinear')  # [B,64,H/4,W/4]
        up2 = self.relu(self.up2(torch.cat([up2, x2], dim=1)))  # [B,128,H/4,W/4] -> [B,64,H/4,W/4]

        up1 = F.interpolate(up2, size=x1.shape[2:], mode='bilinear')
        up1 = self.relu(self.up1(torch.cat([up1, x1], dim=1)))  # [B,128,H/2,W/2] -> [B,64,H/2,W/2]

        up0 = F.interpolate(up1, size=x0.shape[2:], mode='bilinear')
        up0 = self.relu(self.up0(torch.cat([up0, x0], dim=1)))  # [B,128,H,W] -> [B,64,H,W]

        # 多尺度特征融合
        up2_rs = F.interpolate(up2, size=x0.shape[2:], mode='bilinear')  # [B,64,H,W]
        up1_rs = F.interpolate(up1, size=x0.shape[2:], mode='bilinear')  # [B,64,H,W]
        concat_all = torch.cat([up0, up1_rs, up2_rs], dim=1)  # [B,192,H,W]
        fuse = self.fuse(concat_all)  # [B,64,H,W]

        out = self.output_conv(fuse)  # [B,1,H,W]
        return out


class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()

        self.DecomNet = DecomNet()
        self.RelightNet = RelightNet()

        # 损失函数
        self.perceptual_loss = PerceptualLoss()
        self.illumination_loss = IlluminationSmoothnessLoss()
        self.ms_ssim_loss = MSSSIMLoss()
        self.edge_loss = EdgeLoss().cuda()

    def forward(self, input_low, input_high):
        # Forward DecompNet
        input_low = Variable(
            torch.FloatTensor(torch.from_numpy(input_low))).cuda()  # 将低光照输入的numpy数组转换为PyTorch张量，并转移到GPU
        input_high = Variable(torch.FloatTensor(torch.from_numpy(input_high))).cuda()  # 同上，处理高光照输入数据
        R_low, I_low = self.DecomNet(input_low)  # 使用分解网络（DecomNet）对低光输入进行分解
        # 返回：
        # R_low - 低光图像的反射分量（物体表面属性）  [B, 3, H, W]
        # I_low - 低光图像的照明分量（光照条件）  [B, 1, H, W]
        R_high, I_high = self.DecomNet(input_high)  # 对高光输入进行相同分解操作
        # 返回：
        # R_high - 高光图像的反射分量  [B, 3, H, W]
        # I_high - 高光图像的照明分量  [B, 1, H, W]

        # Forward RelightNet
        I_delta = self.RelightNet(I_low, R_low)  # 调用 RelightNet 对低光图像的照明分量进行 relighting 操作  [B, 1, H, W]

        # Other variables
        I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)  # 将单通道的 I_low 沿通道维度复制三次，生成三通道张量
        # 形状从 [B,1,H,W] → [B,3,H,W]
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)  # 将单通道的 I_high（高光图像的光照分量）复制为三通道
        # 形状 [B,1,H,W] → [B,3,H,W]
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)  # 将单通道的调整后光照 I_delta 复制为三通道
        # 形状 [B,1,H,W] → [B,3,H,W]

        # Compute losses
        self.recon_loss_low = F.l1_loss(R_low * I_low_3, input_low)  # 确保低光分解结果能准确重建原始输入
        self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high)  # 确保高光分解结果能准确重建原始输入
        self.recon_loss_mutal_low = F.l1_loss(R_high * I_low_3, input_low)  # 强制反射分量与光照分量解耦（跨图像重建）
        self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high)  # 同上，对称操作
        self.equal_R_loss = F.l1_loss(R_low, R_high.detach())  # 保证不同光照条件下同一场景的反射分量一致
        self.relight_loss = F.l1_loss(R_low * I_delta_3, input_high)  # 保证不同光照条件下同一场景的反射分量一致
        edge_preserve_loss = self.edge_loss(I_delta, I_high)  # 强化光照图像边缘清晰度

        # 额外损失 光照平滑损失
        self.Ismooth_loss_low = self.smooth(I_low, R_low)  # 确保光照条件不连续
        self.Ismooth_loss_high = self.smooth(I_high, R_high)  # 同上
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)  # 确保光照条件不连续

        # 感知损失与多尺度结构相似性损失
        perceptual_loss = self.perceptual_loss(R_low * I_delta_3, input_high)  # 确保反射分量与高光图像的 perceptual loss
        ms_ssim_loss = self.ms_ssim_loss(R_low * I_delta_3, input_high)  # 确保反射分量与高光图像的 perceptual loss

        # 综合DecomNet损失
        self.loss_Decom = self.recon_loss_low + \
                          self.recon_loss_high + \
                          0.001 * self.recon_loss_mutal_low + \
                          0.001 * self.recon_loss_mutal_high + \
                          0.1 * self.Ismooth_loss_low + \
                          0.1 * self.Ismooth_loss_high + \
                          0.01 * self.equal_R_loss
        # 设计逻辑：
        # 主损失：直接重建损失 (recon_loss_low 和 recon_loss_high).
        # 辅助损失：交叉重建损失权重较低 (0.001)，防止过度约束。
        # 平滑损失权重适中 (0.1)，平衡分解精度与光照合理性。
        # 反射一致性损失权重最低 (0.01)，作为软约束。

        # 综合RelightNet损失
        self.loss_Relight = self.relight_loss + \
                            3 * self.Ismooth_loss_delta + \
                            0.1 * perceptual_loss + \
                            0.1 * ms_ssim_loss + \
                            0.1 * edge_preserve_loss
        # 设计逻辑：
        # 主损失：光照调整 L1 损失 (relight_loss).
        # 强光照平滑约束 (权重 3)，确保调整后光照自然。
        # 感知损失与 MS-SSIM 损失权重较低 (0.1)，辅助提升视觉质量。

        self.output_R_low = R_low.detach().cuda()  # 保存低光图像的 反射分量（Reflectance）
        self.output_I_low = I_low_3.detach().cuda()  # 保存低光图像的 光照分量（Illumination）
        self.output_I_delta = I_delta_3.detach().cuda()  # 保存 调整后的光照分量（由 RelightNet 生成）
        self.output_S = R_low.detach().cpu() * I_delta_3.detach().cpu()  # 生成并保存 最终增强结果

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data_names, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data_names)):
            eval_low_img = Image.open(eval_low_data_names[idx])
            eval_low_img = np.array(eval_low_img, dtype="float32") / 255.0
            eval_low_img = np.transpose(eval_low_img, (2, 0, 1))
            input_low_eval = np.expand_dims(eval_low_img, axis=0)

            if train_phase == "Decom":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                input = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                cat_image = np.concatenate([input, result_1, result_2], axis=2)
            if train_phase == "Relight":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                result_3 = self.output_I_delta
                result_4 = self.output_S
                input = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                result_3 = np.squeeze(result_3)
                result_4 = np.squeeze(result_4)
                cat_image = np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = os.path.join(vis_dir, 'eval_%s_%d_%d.png' %
                                    (train_phase, idx + 1, epoch_num))
            im.save(filepath[:-4] + '.jpg')

    def save(self, iter_num, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        save_name = save_dir + '/' + str(iter_num) + '.tar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        elif self.train_phase == 'Relight':
            torch.save(self.RelightNet.state_dict(), save_name)

    def load(self, ckpt_dir):
        load_dir = ckpt_dir + '/' + self.train_phase + '/'
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)

            ipynb_checkpo_path = os.path.join(load_dir, ".ipynb_checkpo")
            # 如果存在则删除
            if os.path.exists(ipynb_checkpo_path):
                if os.path.isdir(ipynb_checkpo_path):
                    # 如果是目录，递归删除
                    import shutil
                    shutil.rmtree(ipynb_checkpo_path)
                else:
                    # 如果是文件，直接删除
                    os.remove(ipynb_checkpo_path)
                print(f"Deleted: {ipynb_checkpo_path}")
            else:
                print(f"No .ipynb_checkpoints found in {load_dir}")
            print(f"load_ckpts:{load_ckpts}")

            ipynb_checkpoints_path = os.path.join(load_dir, ".ipynb_checkpoints")
            # 如果存在则删除
            if os.path.exists(ipynb_checkpoints_path):
                if os.path.isdir(ipynb_checkpoints_path):
                    # 如果是目录，递归删除
                    import shutil
                    shutil.rmtree(ipynb_checkpoints_path)
                else:
                    # 如果是文件，直接删除
                    os.remove(ipynb_checkpoints_path)
                print(f"Deleted: {ipynb_checkpoints_path}")
            else:
                print(f"No .ipynb_checkpoints found in {load_dir}")
            print(f"load_ckpts:{load_ckpts}")

            if len(load_ckpts) > 0:
                load_ckpt = load_ckpts[-1]
                global_step = int(load_ckpt[:-4])
                ckpt_dict = torch.load(load_dir + load_ckpt, weights_only=True)
                if self.train_phase == 'Decom':
                    self.DecomNet.load_state_dict(ckpt_dict)
                elif self.train_phase == 'Relight':
                    self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0

    def train(self,
              train_low_data_names,
              train_high_data_names,
              eval_low_data_names,
              batch_size,
              patch_size, epoch,
              lr,
              vis_dir,
              ckpt_dir,
              eval_every_epoch,
              train_phase):
        assert len(train_low_data_names) == len(train_high_data_names)
        numBatch = len(train_low_data_names) // int(batch_size)

        # Create the optimizers
        self.train_op_Decom = optim.Adam(self.DecomNet.parameters(),
                                         lr=lr[0], betas=(0.9, 0.999))
        self.train_op_Relight = optim.Adam(self.RelightNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999))

        # Initialize a network if its checkpoint is available
        self.train_phase = train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
              (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0
        for epoch in range(start_epoch, epoch):
            self.lr = lr[epoch]
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.train_op_Relight.param_groups:
                param_group['lr'] = self.lr
            for batch_id in range(start_step, numBatch):
                # Generate training data for a batch
                batch_input_low = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                batch_input_high = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                for patch_id in range(batch_size):
                    # Load images
                    train_low_img = Image.open(train_low_data_names[image_id])
                    train_low_img = np.array(train_low_img, dtype='float32') / 255.0
                    train_high_img = Image.open(train_high_data_names[image_id])
                    train_high_img = np.array(train_high_img, dtype='float32') / 255.0
                    # Take random crops
                    h, w, _ = train_low_img.shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
                    train_low_img = train_low_img[x: x + patch_size, y: y + patch_size, :]
                    train_high_img = train_high_img[x: x + patch_size, y: y + patch_size, :]
                    # Data augmentation
                    if random.random() < 0.5:
                        train_low_img = np.flipud(train_low_img)
                        train_high_img = np.flipud(train_high_img)
                    if random.random() < 0.5:
                        train_low_img = np.fliplr(train_low_img)
                        train_high_img = np.fliplr(train_high_img)
                    rot_type = random.randint(1, 4)
                    if random.random() < 0.5:
                        train_low_img = np.rot90(train_low_img, rot_type)
                        train_high_img = np.rot90(train_high_img, rot_type)
                    # Permute the images to tensor format
                    train_low_img = np.transpose(train_low_img, (2, 0, 1))
                    train_high_img = np.transpose(train_high_img, (2, 0, 1))
                    # Prepare the batch
                    batch_input_low[patch_id, :, :, :] = train_low_img
                    batch_input_high[patch_id, :, :, :] = train_high_img
                    self.input_low = batch_input_low
                    self.input_high = batch_input_high

                    image_id = (image_id + 1) % len(train_low_data_names)
                    if image_id == 0:
                        tmp = list(zip(train_low_data_names, train_high_data_names))
                        random.shuffle(list(tmp))
                        train_low_data_names, train_high_data_names = zip(*tmp)

                # Feed-Forward to the network and obtain loss
                self.forward(self.input_low, self.input_high)
                if self.train_phase == "Decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    loss = self.loss_Decom.item()
                elif self.train_phase == "Relight":
                    self.train_op_Relight.zero_grad()
                    self.loss_Relight.backward()
                    self.train_op_Relight.step()
                    loss = self.loss_Relight.item()

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data_names,
                              vis_dir=vis_dir, train_phase=train_phase)
                self.save(iter_num, ckpt_dir)

        print("Finished training for phase %s." % train_phase)

    def compute_metrics(original, enhanced):
        # 计算 PSNR、SSIM、LPIPS、MAE、STD、GCF 评价指标
        original_np = np.array(original) / 255.0  # 归一化
        enhanced_np = np.array(enhanced) / 255.0  # 归一化

        # PSNR
        psnr_value = psnr(original_np, enhanced_np, data_range=1.0)

        # 确保 win_size 不超过图像的最小边长
        min_dim = min(original_np.shape[0], original_np.shape[1])
        win_size = min(7, min_dim)

        # SSIM
        ssim_value = ssim(original_np, enhanced_np, data_range=1.0, channel_axis=-1, win_size=win_size)

        # LPIPS
        lpips_value = lpips_model(
            torch.tensor(original_np).permute(2, 0, 1).unsqueeze(0).float(),
            torch.tensor(enhanced_np).permute(2, 0, 1).unsqueeze(0).float()
        ).item()

        # MAE
        mae_value = np.mean(np.abs(original_np - enhanced_np))

        # STD（标准差）
        std_value = np.std(enhanced_np)

        # GCF（全局对比度因子）
        gcf_value = np.mean(np.abs(enhanced_np - enhanced_np.mean()))

        return {
            "PSNR": psnr_value,
            "SSIM": ssim_value,
            "LPIPS": lpips_value,
            "MAE": mae_value,
            "STD": std_value,
            "GCF": gcf_value
        }

    def predict(self,
                test_low_data_names,
                res_dir,
                ckpt_dir):

        # Load the network with a pre-trained checkpoint
        self.train_phase = 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception
        self.train_phase = 'Relight'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = False
        save_only_one = True

        # Predict for the test images
        for idx in range(len(test_low_data_names)):
            test_img_path = test_low_data_names[idx]
            test_img_name = test_img_path.split('/')[-1]
            print('Processing ', test_img_name)
            test_low_img = Image.open(test_img_path)
            test_low_img = np.array(test_low_img, dtype="float32") / 255.0
            test_low_img = np.transpose(test_low_img, (2, 0, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)

            self.forward(input_low_test, input_low_test)
            result_1 = self.output_R_low
            result_2 = self.output_I_low
            result_3 = self.output_I_delta
            result_4 = self.output_S
            input = np.squeeze(input_low_test)
            result_1 = np.squeeze(result_1)
            result_2 = np.squeeze(result_2)
            result_3 = np.squeeze(result_3)
            result_4 = np.squeeze(result_4)

            # print(f"result_4:{result_4.shape}")
            if save_only_one:
                # cat_image = input
                cat_image = result_4.detach().cpu().numpy()
            else:
                if save_R_L:
                    cat_image = np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)
                else:
                    cat_image = np.concatenate([input, result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = os.path.join(res_dir, test_img_name)
            im.save(filepath)