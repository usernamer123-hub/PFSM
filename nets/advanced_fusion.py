"""
高级差分修复模块 - 解决偏振图像融合中的鬼影和细节丢失问题
作者：改进版
日期：2025-01-24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

# ===============================================================
# 1. 自适应引导滤波（解决固定参数问题）
# ===============================================================

class AdaptiveGuidedFilter(nn.Module):
    """
    自适应引导滤波器：
    - 根据特征图尺度动态调整半径r
    - 根据区域纹理密度动态调整eps
    - 为不同区域提供不同的滤波强度
    """
    def __init__(self, base_r=2, base_eps=1e-3):
        super(AdaptiveGuidedFilter, self).__init__()
        self.base_r = base_r
        self.base_eps = base_eps
        
        # 参数预测网络：输入差分特征，输出eps调整因子
        self.eps_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()  # 输出0-1，用于缩放eps
        )
        
    def box_filter(self, x, r):
        """快速盒式滤波"""
        ch = x.shape[1]
        k = 2 * r + 1
        weight = torch.ones((ch, 1, k, k), device=x.device) / (k * k)
        return F.conv2d(x, weight, padding=r, groups=ch)
    
    def forward(self, guide, source, diff_map):
        """
        guide: 引导特征 [B, C, H, W]
        source: 源特征 [B, C, H, W]
        diff_map: 差分图 [B, 1, H, W] 或 [B, C, H, W]
        """
        B, C, H, W = guide.shape
        
        # 1. 自适应调整eps（根据差分强度）
        if diff_map.size(1) != 1:
            diff_map = torch.mean(diff_map, dim=1, keepdim=True)
        eps_scale = self.eps_predictor(diff_map)  # [B, 1, 1, 1]
        eps = self.base_eps * (0.1 + 10 * eps_scale)  # 动态范围: [1e-4, 1e-2]
        
        # 2. 根据分辨率自适应调整r
        scale_factor = max(1, H // 128)
        r = self.base_r * scale_factor
        
        # 3. 引导滤波核心计算
        mean_I = self.box_filter(guide, r)
        mean_p = self.box_filter(source, r)
        mean_Ip = self.box_filter(guide * source, r)
        
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = self.box_filter(guide * guide, r)
        var_I = mean_II - mean_I * mean_I
        
        # 使用自适应eps
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = self.box_filter(a, r)
        mean_b = self.box_filter(b, r)
        
        output = mean_a * guide + mean_b
        
        return output

# ===============================================================
# 2. 物理感知差分模块（解决简单相减的问题）
# ===============================================================

class PhysicsAwareDifference(nn.Module):
    """
    物理感知差分：
    - 考虑S0（光强）和DoP（偏振度）的物理意义
    - 提取空域+频域双差分特征
    - 生成多尺度差分图
    """
    def __init__(self, in_channels):
        super(PhysicsAwareDifference, self).__init__()
        
        # 1. 空域差分（多尺度）
        self.spatial_diff = nn.ModuleList([
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=k, padding=k//2)
            for k in [1, 3, 5]  # 多尺度感受野
        ])
        
        # 2. 频域差分（Sobel梯度）
        self.register_buffer('sobel_x', torch.FloatTensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.FloatTensor([
            [1, 2, 1], [0, 0, 0], [-1, -2, -1]
        ]).view(1, 1, 3, 3))
        
        # 3. 差分特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, 1),  # 4 = 3个空域 + 1个频域
            nn.GroupNorm(4, in_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def get_gradient_diff(self, x1, x2):
        """计算梯度差异（保留通道信息）"""
        B, C, H, W = x1.shape
        
        # 对每个通道分别计算梯度
        x1_flat = x1.view(B * C, 1, H, W)
        x2_flat = x2.view(B * C, 1, H, W)
        
        grad1_x = F.conv2d(x1_flat, self.sobel_x, padding=1)
        grad1_y = F.conv2d(x1_flat, self.sobel_y, padding=1)
        grad1 = torch.sqrt(grad1_x ** 2 + grad1_y ** 2 + 1e-6)
        
        grad2_x = F.conv2d(x2_flat, self.sobel_x, padding=1)
        grad2_y = F.conv2d(x2_flat, self.sobel_y, padding=1)
        grad2 = torch.sqrt(grad2_x ** 2 + grad2_y ** 2 + 1e-6)
        
        grad_diff = torch.abs(grad1 - grad2).view(B, C, H, W)
        return grad_diff
    
    def forward(self, feat_guide, feat_source):
        """
        feat_guide: 低频特征 (例如S0) [B, C, H, W]
        feat_source: 高频特征 (例如DoP) [B, C, H, W]
        """
        # 1. 空域多尺度差分
        concat_feat = torch.cat([feat_guide, feat_source], dim=1)
        spatial_diffs = [conv(concat_feat) for conv in self.spatial_diff]
        
        # 2. 频域差分（梯度差异）
        freq_diff = self.get_gradient_diff(feat_guide, feat_source)
        
        # 3. 融合所有差分特征
        all_diffs = torch.cat(spatial_diffs + [freq_diff], dim=1)
        diff_feat = self.fusion(all_diffs)
        
        return diff_feat

# ===============================================================
# 3. 动态融合权重生成器（解决固定权重问题）
# ===============================================================

class DynamicFusionWeight(nn.Module):
    """
    动态融合权重生成器：
    - 根据区域特性（平坦/纹理）生成不同的融合策略
    - 支持"完全屏蔽"某一分支
    - 引入注意力机制增强判别能力
    """
    def __init__(self, in_channels):
        super(DynamicFusionWeight, self).__init__()
        
        # 1. 纹理密度检测器
        self.texture_detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 2. 全局-局部注意力
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 3, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.local_attn = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1),
            nn.GroupNorm(4, in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 3. 权重预测头（无下限约束）
        self.weight_head = nn.Sequential(
            nn.Conv2d(in_channels + 2, in_channels // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()  # 输出0-1，不强制下限
        )
        
    def forward(self, feat_guide, feat_source, diff_feat):
        """
        返回: alpha权重，范围[0, 1]
        fusion = feat_guide * alpha + feat_source * (1 - alpha)
        """
        B, C, H, W = feat_guide.shape
        
        # 1. 检测纹理密度
        texture_map = self.texture_detector(diff_feat)  # [B, 1, H, W]
        
        # 2. 全局-局部注意力
        combined = torch.cat([feat_guide, feat_source, diff_feat], dim=1)
        global_weight = self.global_attn(combined)  # [B, C, 1, 1]
        local_weight = self.local_attn(combined)     # [B, 1, H, W]
        
        # 3. 融合注意力特征
        weighted_diff = diff_feat * global_weight  # 广播
        weight_input = torch.cat([weighted_diff, texture_map, local_weight], dim=1)
        
        # 4. 生成最终权重（无下限）
        alpha = self.weight_head(weight_input)
        
        return alpha

# ===============================================================
# 4. 高级差分融合层（整合上述模块）
# ===============================================================

class AdvancedDiffusionFusionLayer(nn.Module):
    """
    高级差分融合层：
    - 集成自适应引导滤波
    - 物理感知差分
    - 动态融合权重
    - 残差连接增强稳定性
    """
    def __init__(self, in_channels_guide, in_channels_source, out_channels):
        super(AdvancedDiffusionFusionLayer, self).__init__()
        
        # 1. 投影对齐
        self.guide_proj = nn.Conv2d(in_channels_guide, in_channels_source, 1, bias=False)
        
        # 2. 物理感知差分
        self.diff_extractor = PhysicsAwareDifference(in_channels_source)
        
        # 3. 自适应引导滤波
        self.adaptive_gf = AdaptiveGuidedFilter(base_r=2, base_eps=1e-3)
        
        # 4. 动态融合权重
        self.fusion_weight = DynamicFusionWeight(in_channels_source)
        
        # 5. 最终投影
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels_source, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 6. 残差门控
        if in_channels_source != out_channels:
            self.residual_proj = nn.Conv2d(in_channels_source, out_channels, 1)
        else:
            self.residual_proj = nn.Identity()
        
    def forward(self, feat_guide, feat_source):
        """
        feat_guide: 低频特征 [B, C1, H, W]
        feat_source: 高频特征 [B, C2, H, W]
        返回: 融合后特征 [B, out_channels, H, W]
        """
        # 1. 投影对齐
        feat_guide_aligned = self.guide_proj(feat_guide)
        
        # 2. 提取物理感知差分特征
        diff_feat = self.diff_extractor(feat_guide_aligned, feat_source)
        
        # 3. 自适应引导滤波修复
        diff_map = torch.mean(torch.abs(diff_feat), dim=1, keepdim=True)
        feat_refined = self.adaptive_gf(feat_guide_aligned, feat_source, diff_map)
        
        # 4. 动态融合
        alpha = self.fusion_weight(feat_guide_aligned, feat_refined, diff_feat)
        feat_fused = feat_guide_aligned * alpha + feat_refined * (1 - alpha)
        
        # 5. 输出投影 + 残差
        out = self.out_conv(feat_fused)
        residual = self.residual_proj(feat_source)
        
        return out + residual * 0.2  # 弱残差连接，保留部分原始信息

# ===============================================================
# 5. 改进的MSFFN差分模块（用于mism_2d.py替换）
# ===============================================================

class ImprovedMSFFN(nn.Module):
    """
    改进的多尺度特征融合网络：
    - 使用物理感知差分
    - 动态融合权重
    - 多尺度残差连接
    """
    def __init__(self, in_channels):
        super(ImprovedMSFFN, self).__init__()
        self.dim = in_channels
        
        # Path 1: LKA (S0-like) 保持不变
        self.lka_conv0 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)
        self.lka_conv_spatial = nn.Conv2d(in_channels, in_channels, 7, stride=1, padding=9, groups=in_channels, dilation=3)
        self.lka_conv1 = nn.Conv2d(in_channels, in_channels, 1)
        
        # Path 2: Detail (DoP-like) 保持不变
        self.detail_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.GroupNorm(4, in_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Adaptive fusion of low- and high-frequency features
        self.diff_extractor = PhysicsAwareDifference(in_channels)
        self.fusion_weight = DynamicFusionWeight(in_channels)
        
        self.proj_in = nn.Conv2d(in_channels, in_channels, 1)
        self.gn = nn.GroupNorm(4, in_channels)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        shortcut = x
        x = self.gn(x)
        x = self.proj_in(x)
        
        # Path 1: LKA
        u = x.clone()
        attn_lka = self.lka_conv0(x)
        attn_lka = self.lka_conv_spatial(attn_lka)
        attn_lka = self.lka_conv1(attn_lka)
        feat_lka = u * attn_lka
        
        # Path 2: Detail
        feat_detail = self.detail_extractor(u)
        
        # *** 改进的融合策略 ***
        diff_feat = self.diff_extractor(feat_lka, feat_detail)
        alpha = self.fusion_weight(feat_lka, feat_detail, diff_feat)
        
        out = feat_lka * alpha + feat_detail * (1 - alpha)
        out = self.proj_out(out)
        
        return out + shortcut



