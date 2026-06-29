import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math

# class Sobelxy(nn.Module):
#     """
#     计算图像梯度的模块 (用于 Texture/Edge Loss)
#     PFSM loss functions
#     """
#     def __init__(self):
#         super(Sobelxy, self).__init__()
#         kernelx = [[-1, 0, 1],
#                    [-2, 0, 2],
#                    [-1, 0, 1]]
#         kernely = [[1, 2, 1],
#                    [0, 0, 0],
#                    [-1, -2, -1]]
#         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#         kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        
#         # 设置为不可训练参数
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False)

#     def forward(self, x):
#         # x shape: [B, C, H, W]
#         # 为了处理多通道，我们将 batch 和 channel 合并，单独计算每一张图的梯度
#         b, c, h, w = x.shape
#         x_reshaped = x.view(b * c, 1, h, w)
        
#         sobelx = F.conv2d(x_reshaped, self.weightx, padding=1)
#         sobely = F.conv2d(x_reshaped, self.weighty, padding=1)
        
#         # 计算梯度幅值
#         grad = torch.abs(sobelx) + torch.abs(sobely)
        
#         return grad.view(b, c, h, w)
    
# class ContrastLoss(nn.Module):
#     """
#     对比度损失：强制生成的 S0 图像具有和 GT 相似的均值和标准差。
#     对于去雾任务至关重要，因为雾会导致对比度降低（方差变小）。
#     """
#     def __init__(self):
#         super(ContrastLoss, self).__init__()
#         self.l1 = nn.L1Loss()

#     def forward(self, x, y):
#         # x: Pred, y: GT
#         # 计算 Channel 维度的均值和标准差
#         mean_x = torch.mean(x, dim=[2, 3])
#         mean_y = torch.mean(y, dim=[2, 3])
        
#         std_x = torch.std(x, dim=[2, 3])
#         std_y = torch.std(y, dim=[2, 3])
        
#         return self.l1(mean_x, mean_y) + self.l1(std_x, std_y)
    
# class ColorLoss(nn.Module):
#     def __init__(self):
#         super(ColorLoss, self).__init__()
#         self.register_buffer("rgb2xyz_matrix", torch.tensor([
#             [0.412453, 0.357580, 0.180423],
#             [0.212671, 0.715160, 0.072169],
#             [0.019334, 0.119193, 0.950227]
#         ]).view(3, 3, 1, 1))

#     def rgb_to_xyz(self, x):
#         return F.conv2d(x, self.rgb2xyz_matrix)

#     def xyz_to_lab(self, xyz):
#         xn, yn, zn = 0.950456, 1.0, 1.088754
#         xyz_norm = xyz.clone()
#         xyz_norm[:, 0, :, :] = xyz_norm[:, 0, :, :] / xn
#         xyz_norm[:, 1, :, :] = xyz_norm[:, 1, :, :] / yn
#         xyz_norm[:, 2, :, :] = xyz_norm[:, 2, :, :] / zn
        
#         delta = 6/29
#         delta_cube = delta ** 3
#         mask = xyz_norm > delta_cube
#         f_xyz = torch.zeros_like(xyz_norm)
#         f_xyz[mask] = torch.pow(xyz_norm[mask], 1.0/3.0)
#         f_xyz[~mask] = (xyz_norm[~mask] / (3 * (delta**2))) + (4/29)

#         L = 116 * f_xyz[:, 1, :, :] - 16
#         a = 500 * (f_xyz[:, 0, :, :] - f_xyz[:, 1, :, :])
#         b = 200 * (f_xyz[:, 1, :, :] - f_xyz[:, 2, :, :])
#         return torch.stack([L, a, b], dim=1)

#     def forward(self, x, y):
#         # 假设输入是 [-1, 1]，转为 [0, 1]
#         x = (x + 1.0) / 2.0
#         y = (y + 1.0) / 2.0
#         x = torch.clamp(x, 1e-6, 1.0)
#         y = torch.clamp(y, 1e-6, 1.0)

#         lab_pred = self.xyz_to_lab(self.rgb_to_xyz(x))
#         lab_gt = self.xyz_to_lab(self.rgb_to_xyz(y))

#         # L亮度权重加倍，强迫提亮
#         loss_l = F.l1_loss(lab_pred[:, 0], lab_gt[:, 0]) * 1.5 
#         loss_a = F.l1_loss(lab_pred[:, 1], lab_gt[:, 1])
#         loss_b = F.l1_loss(lab_pred[:, 2], lab_gt[:, 2])

#         return (loss_l + loss_a + loss_b) / 3.0

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean() # 返回 1 - SSIM 作为 Loss
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)
    
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = sqrt(diff^2 + eps^2)
        loss = torch.sqrt((diff * diff) + (self.eps*self.eps))
        return torch.mean(loss)
 
 # 原有的损失函数可能出现冲突    
# class HybridPolarLoss(nn.Module):
#     """
#     混合偏振损失函数：
#     1. Intensity Loss (S0 L1) - 保证光强准确，防止过曝
#     2. DoP Loss (L1) - 保证偏振度准确
#     3. Gradient Loss - 保证纹理和边缘清晰 (Mamba 的强项)
#     4. SSIM Loss - 保证结构一致性，消除虚影
#     """
#     def __init__(self, w_s0=1.0, w_dolp=1.0, w_grad=1.0, w_cont=0.2, w_ssim=2.0, w_color=0.1):
#         super(HybridPolarLoss, self).__init__()
#         self.sobel = Sobelxy()
#         # self.l1 = nn.L1Loss()
#         self.char_loss = CharbonnierLoss(eps=1e-3)
#         self.contrast = ContrastLoss()
#         self.ssim_loss = SSIMLoss()
#         self.color_loss = ColorLoss()
        
#         self.w_s0 = w_s0
#         self.w_dolp = w_dolp
#         self.w_grad = w_grad
#         self.w_cont = w_cont
#         self.w_ssim = w_ssim
#         self.w_color = w_color

#     def forward(self, s0_pred, s0_gt, dolp_pred, dolp_gt):
#         # 1. 物理量 L1 Loss
#         # loss_s0 = self.l1(s0_pred, s0_gt)
#         # loss_dolp = self.l1(dolp_pred, dolp_gt)
#         loss_s0 = self.char_loss(s0_pred, s0_gt)
#         loss_dolp = self.char_loss(dolp_pred, dolp_gt)
#         # 2. 梯度 Loss (重点计算 S0 的梯度，因为 S0 包含主要轮廓)
#         # 将数据 detach 防止梯度爆炸（可选），但在训练生成器时通常需要传梯度
#         grad_pred = self.sobel(s0_pred)
#         grad_gt = self.sobel(s0_gt)
#         # loss_grad = self.l1(grad_pred, grad_gt)
#         loss_grad = self.char_loss(grad_pred, grad_gt)
#         loss_cont = self.contrast(s0_pred, s0_gt)
#         loss_ssim = self.ssim_loss(s0_pred, s0_gt)
#         loss_color = self.color_loss(s0_pred, s0_gt)
        
#         # 3. 组合
#         total_loss = (self.w_s0 * loss_s0) + (self.w_dolp * loss_dolp) + (self.w_grad * loss_grad) + (self.w_cont * loss_cont) + (self.w_ssim * loss_ssim) + (self.w_color * loss_color)
        
#         return total_loss, loss_grad, loss_cont, loss_color

class UnifiedMultiscalePolarLoss(nn.Module):
    """
    [借鉴自 PGGAFNet] 统一多尺度偏振结构损失
    通过 SSIM 与 Charbonnier 的联合，在 5 个尺度上逼迫网络完美对齐物理纹理。
    """
    def __init__(self, init_alpha=0.84, w_s0=1.0):
        super(UnifiedMultiscalePolarLoss, self).__init__()
        # self.alpha = alpha  # SSIM 与 像素损失的平衡系数
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        
        self.w_s0 = w_s0    
        self.char_loss = CharbonnierLoss(eps=1e-3)
        self.ssim_loss = SSIMLoss(window_size=11, size_average=True)
        # 多尺度监督权重：浅层（细粒度）权重大，深层（语义）权重小
        self.scale_weights = [1.0, 0.8, 0.6, 0.4, 0.2]

    def calc_single_scale_loss(self, pred, gt):
        """计算单一尺度的 SSIM + Charbonnier 联合损失"""
        # 对齐分辨率
        if pred.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(gt, size=pred.shape[-2:], mode='bilinear', align_corners=False)
            
        l_char = self.char_loss(pred, gt)
        l_ssim = self.ssim_loss(pred, gt)
        
        # 使用 torch.clamp 限制 alpha 在 0~1 之间，防止网络学崩溃
        current_alpha = torch.clamp(self.alpha, min=0.0, max=1.0)
        
        # 联合物理结构损失公式
        # return self.alpha * l_ssim + (1.0 - self.alpha) * l_char
        return current_alpha * l_ssim + (1.0 - current_alpha) * l_char

    def forward(self, preds_list, target_gt):
        """
        preds_list: PFSMNet outputs的 5 个尺度的列表 [out_256, out_128, out_64, out_32, out_16]
        target_gt: 原尺寸的真值
        """
        total_loss = 0.0
        
        # 遍历 5 个尺度的预测结果进行联合监督
        for i in range(len(preds_list)):
            level_loss = self.calc_single_scale_loss(preds_list[i], target_gt)
            total_loss += self.scale_weights[i] * level_loss
            
        return total_loss
