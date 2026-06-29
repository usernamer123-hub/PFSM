# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SFTM(nn.Module):
#     """
#     Enhanced Stokes Feature Transformation Module (E-SFTM)
#     改进点：
#     1. 引入 S0 (光强) 作为 Attention 的参考，防止高光误判。
#     2. 使用 5x5 卷积扩大感受野，生成更平滑的 Attention Map。
#     3. 保留了原始特征的残差连接。
#     """
#     def __init__(self, in_channels=12):
#         super(SFTM, self).__init__()
        
#         assert in_channels == 12, "Input must be 12 channels"
        
#         # --- 1. 物理特征提取层 ---
#         # 输入: S0(3) + S1(3) + S2(3) = 9 channels
#         # 把 S0 放进去，让网络学习 (S1/S0) 这种归一化关系
#         self.physics_encoder = nn.Sequential(
#             # 使用 5x5 卷积扩大感受野，捕捉区域偏振特征
#             nn.Conv2d(9, 12, kernel_size=5, padding=2, bias=False), 
#             nn.BatchNorm2d(12),
#             nn.ReLU(inplace=True)
#         )
        
#         # --- 2. Attention 生成层 ---
#         self.attention_gen = nn.Sequential(
#             nn.Conv2d(12, 12, kernel_size=1, bias=False),
#             nn.Sigmoid() # 输出 0~1 的权重
#         )

#         # --- 3. 特征融合与精炼 ---
#         self.refine = nn.Sequential(
#             nn.Conv2d(12, 12, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(12),
#             nn.LeakyReLU(0.2, inplace=True)
#         )

#     def forward(self, x):
#         # x: [B, 12, H, W] -> I0, I45, I90, I135
#         # 1. 物理切片
#         I0   = x[:, 0:3, :, :]
#         I45  = x[:, 3:6, :, :]
#         I90  = x[:, 6:9, :, :]
#         I135 = x[:, 9:12, :, :]

#         # 2. 构造 Stokes 参数 (近似)
#         # S0: 总光强 (重要参考！)
#         feat_s0 = (I0 + I90 + I45 + I135) / 2.0 
        
#         # S1, S2: 偏振差分
#         feat_s1 = I0 - I90
#         feat_s2 = I45 - I135
        
#         # 3. 拼接物理特征 [B, 9, H, W]
#         # 网络现在同时看到了：光强(S0) 和 偏振差异(S1, S2)
#         physics_feat = torch.cat([feat_s0, feat_s1, feat_s2], dim=1)

#         # 4. 生成 Attention
#         spatial_ctx = self.physics_encoder(physics_feat)
#         attn_map = self.attention_gen(spatial_ctx)

#         # 5. 施加 Attention (Residual)
#         # 增强包含显著物理特征的区域
#         out = x * attn_map + x
        
#         # 6. Refine
#         out = self.refine(out)
        
#         return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class SFTM(nn.Module):
    """
    Enhanced Stokes Feature Transformation Module (E-SFTM)
    [翻新升级版]：
    1. 引入 S0 (光强), DoLP(偏振度), AoP(偏振角) 扩充至 15 通道。
    2. 提取全局物理空间注意力的同时，将计算得到的 AoP 作为显式空间先验向后传递。
    """
    def __init__(self, in_channels=12):
        super(SFTM, self).__init__()
        
        assert in_channels == 12, "Input must be 12 channels"
        
        # --- 1. 物理特征提取层 (9通道 -> 15通道) ---
        self.physics_encoder = nn.Sequential(
            nn.Conv2d(15, 12, kernel_size=5, padding=2, bias=False), 
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )
        
        # --- 2. Attention 生成层 ---
        self.attention_gen = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=1, bias=False),
            nn.Sigmoid() 
        )

        # --- 3. 特征融合与精炼 ---
        self.refine = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # 1. 物理切片 [B, 12, H, W]
        I0   = x[:, 0:3, :, :]
        I45  = x[:, 3:6, :, :]
        I90  = x[:, 6:9, :, :]
        I135 = x[:, 9:12, :, :]

        # 2. 构造高阶 Stokes 参数
        eps = 1e-6
        feat_s0 = (I0 + I90 + I45 + I135) / 2.0 
        feat_s1 = I0 - I90
        feat_s2 = I45 - I135
        
        # 计算 DoLP (线偏振度)
        feat_dolp = torch.sqrt(feat_s1**2 + feat_s2**2 + eps) / (feat_s0 + eps)
        feat_dolp = torch.clamp(feat_dolp, 0.0, 1.0) 
        
        # 计算 AoP (偏振角), 归一化至 [-1, 1]
        feat_aop = 0.5 * torch.atan2(feat_s2, feat_s1 + eps)
        feat_aop = feat_aop / (3.14159265 / 2.0)
        
        # 3. 拼接 15 通道物理特征
        physics_feat = torch.cat([feat_s0, feat_s1, feat_s2, feat_dolp, feat_aop], dim=1)

        # 4. 生成 Attention 并施加残差
        spatial_ctx = self.physics_encoder(physics_feat)
        attn_map = self.attention_gen(spatial_ctx)

        out = x * attn_map + x
        out = self.refine(out)
        
        # [核心改动]：同时返回特征 out 和 物理先验 AoP
        return out, feat_aop
