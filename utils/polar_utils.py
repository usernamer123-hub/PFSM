import torch

def calculate_s0_dop(input_tensor):
    """
    从 12 通道输出计算 S0 和 DoP
    Input: [B, 12, H, W] (顺序: I0, I45, I90, I135, 每种3通道)
    Output: 
        S0: [B, 3, H, W] (RGB 清晰图像)
        DoP: [B, 1, H, W] (灰度偏振度图)
    """
    # 1. 拆分通道
    # 假设顺序是 I0, I45, I90, I135 (每块 3 个通道)
    I0   = input_tensor[:, 0:3, :, :]
    I45  = input_tensor[:, 3:6, :, :]
    I90  = input_tensor[:, 6:9, :, :]
    I135 = input_tensor[:, 9:12, :, :]

    # 2. 计算斯托克斯矢量 (S0, S1, S2)
    # S0 = 总光强 (0.5 * (I0 + I90 + I45 + I135))
    S0 = 0.5 * (I0 + I90 + I45 + I135)
    
    # S1 = 水平/垂直差
    S1 = I0 - I90
    
    # S2 = 45/135度差
    S2 = I45 - I135

    # 3. 计算 DoP (Degree of Linear Polarization)
    # 公式: DoP = sqrt(S1^2 + S2^2) / S0
    # 为了计算单通道 DoP，我们先计算 Intensity (灰度)
    # 或者直接算 3 通道 DoP 然后求平均
    
    # 这里计算 3 通道的偏振强度
    P_int = torch.sqrt(S1**2 + S2**2)
    
    # 防止除以 0，加一个极小值 epsilon
    epsilon = 1e-6
    DoP_rgb = P_int / (S0 + epsilon)
    
    # 限制在 [0, 1] 范围内 (物理约束)
    DoP_rgb = torch.clamp(DoP_rgb, 0, 1)
    
    # 将 3 通道 DoP 转为 1 通道 (取平均)
    DoP_gray = torch.mean(DoP_rgb, dim=1, keepdim=True)

    return S0, DoP_gray