import cv2
import numpy as np
import torch
import torch.nn.functional as F

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order."""
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    if len(img.shape) == 3:
        if input_order == 'CHW':
            img = img.transpose(1, 2, 0)
    return img

def calculate_psnr(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)."""
    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
        
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    max_value = 255. # 假设输入已经是 0-255 范围
    return 20. * np.log10(max_value / np.sqrt(mse))

def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images."""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):
    """Calculate SSIM (structural similarity)."""
    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # 多通道 SSIM：对每个通道分别计算然后求平均
    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))

    return np.array(ssims).mean()

def tensor2img(tensor):
    """
    将 Tensor (B,C,H,W) 转换为 numpy 图像 (H,W,C)，范围 0-255，uint8
    假设输入 tensor 是经过 Normalize((0.5,...), (0.5,...)) 的，范围 [-1, 1]
    """
    # 1. 反归一化: [-1, 1] -> [0, 1]
    img = tensor.detach().cpu() * 0.5 + 0.5
    # 2. 限制范围
    img = torch.clamp(img, 0, 1)
    # 3. 转换为 numpy, [0, 255], uint8
    img = img.numpy().transpose(1, 2, 0) # CHW -> HWC
    img = (img * 255.0).round().astype(np.uint8)
    return img