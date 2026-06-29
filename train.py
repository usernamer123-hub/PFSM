"""
 > Training pipeline for PFSM underwater polarization image restoration model
 > Adapted for: 12-channel Input/Output -> S0/DoP Calculation
"""
import os
import sys
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.transforms import Resize

# Local libs
from nets.Dwt import DWTForward
from nets.commons import Weights_Normal, VGG19_PercepLoss
from nets.pfsm import PFSMNet
from utils.data_utils import GetTrainingPairs, GetValImage, AspectRatioResize
from utils.metrics import calculate_psnr, calculate_ssim, tensor2img

def remove_padding_and_resize(tensor, target_size=(512, 512), pad_threshold=0.01, edge_check_size=10):
    """
    移除AspectRatioResize添加的padding区域，然后resize到目标尺寸（不保持宽高比）
    Args:
        tensor: [B, C, H, W] 或 [C, H, W]，范围[0, 1]
        target_size: 目标尺寸 (H, W)
        pad_threshold: padding检测阈值（padding是灰色128，归一化后约0.004）
        edge_check_size: 检查边缘的像素数
    Returns:
        tensor: 裁剪并resize后的tensor，尺寸为target_size，不包含padding
    """
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False
    
    B, C, H, W = tensor.shape
    
    # 检测边缘的padding区域
    # padding是灰色128，归一化后约0.004，RGB三个通道值接近
    # 检查上下边缘
    top_edge = tensor[:, :, :edge_check_size, :].mean(dim=[1, 3])  # [B, edge_check_size]
    bottom_edge = tensor[:, :, -edge_check_size:, :].mean(dim=[1, 3])  # [B, edge_check_size]
    
    # 检查左右边缘
    left_edge = tensor[:, :, :, :edge_check_size].mean(dim=[1, 2])  # [B, edge_check_size]
    right_edge = tensor[:, :, :, -edge_check_size:].mean(dim=[1, 2])  # [B, edge_check_size]
    
    # 判断是否为padding（值接近0.004，且RGB通道值接近）
    # padding区域：值接近0.004，标准差小（RGB接近）
    top_is_pad = (top_edge.abs() < pad_threshold).all(dim=1)  # [B]
    bottom_is_pad = (bottom_edge.abs() < pad_threshold).all(dim=1)  # [B]
    left_is_pad = (left_edge.abs() < pad_threshold).all(dim=1)  # [B]
    right_is_pad = (right_edge.abs() < pad_threshold).all(dim=1)  # [B]
    
    # 计算裁剪边界（从边缘向内找到第一个非padding行/列）
    top = 0
    bottom = H
    left = 0
    right = W
    
    # 从上往下找
    if top_is_pad.any():
        for i in range(H):
            row_mean = tensor[:, :, i, :].mean(dim=[1, 2])  # [B]
            if (row_mean.abs() > pad_threshold).any():
                top = i
                break
    
    # 从下往上找
    if bottom_is_pad.any():
        for i in range(H-1, -1, -1):
            row_mean = tensor[:, :, i, :].mean(dim=[1, 2])  # [B]
            if (row_mean.abs() > pad_threshold).any():
                bottom = i + 1
                break
    
    # 从左往右找
    if left_is_pad.any():
        for i in range(W):
            col_mean = tensor[:, :, :, i].mean(dim=[1, 2])  # [B]
            if (col_mean.abs() > pad_threshold).any():
                left = i
                break
    
    # 从右往左找
    if right_is_pad.any():
        for i in range(W-1, -1, -1):
            col_mean = tensor[:, :, :, i].mean(dim=[1, 2])  # [B]
            if (col_mean.abs() > pad_threshold).any():
                right = i + 1
                break
    
    # 安全检查：确保裁剪区域有效
    if bottom <= top or right <= left:
        # 如果检测失败，返回原始tensor（不裁剪）
        tensor_cropped = tensor
    else:
        # 确保裁剪尺寸至少为1
        bottom = max(bottom, top + 1)
        right = max(right, left + 1)
        tensor_cropped = tensor[:, :, top:bottom, left:right]
    
    # 检查裁剪后的尺寸
    _, _, crop_h, crop_w = tensor_cropped.shape
    if crop_h < 1 or crop_w < 1:
        # 如果裁剪后尺寸无效，返回原始tensor
        tensor_cropped = tensor
        crop_h, crop_w = H, W
    
    # Resize到目标尺寸（不保持宽高比，直接拉伸到512×512）
    target_h, target_w = target_size
    
    # 安全检查：确保目标尺寸有效
    if target_h < 1 or target_w < 1:
        target_h, target_w = H, W
    
    # 如果已经是目标尺寸，直接返回
    if crop_h == target_h and crop_w == target_w:
        tensor_resized = tensor_cropped
    else:
        tensor_resized = F.interpolate(
            tensor_cropped, 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        )
    
    # 检查NaN
    if torch.isnan(tensor_resized).any():
        # 如果出现NaN，返回原始tensor
        print(f"Warning: NaN detected in remove_padding_and_resize, returning original tensor")
        tensor_resized = tensor
    
    if squeeze_back:
        tensor_resized = tensor_resized.squeeze(0)
    
    return tensor_resized
# 引入偏振计算工具
from utils.polar_utils import calculate_s0_dop
# from utils.loss_utils import HybridPolarLoss, Sobelxy
from utils.loss_utils import UnifiedMultiscalePolarLoss

# ==================== 辅助 Loss 类 ====================
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class Laplace(nn.Module):
    def __init__(self):
        super(Laplace, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        nn.init.constant_(self.conv1.weight, 1)
        nn.init.constant_(self.conv1.weight[0, 0, 1, 1], -8)
        nn.init.constant_(self.conv1.weight[0, 1, 1, 1], -8)
        nn.init.constant_(self.conv1.weight[0, 2, 1, 1], -8)

    def forward(self, x1):
        edge_map = self.conv1(x1)
        return edge_map

# ==================== 参数配置 ====================
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--dataset_path", type=str, default="./data/train/", help="path of train images")
parser.add_argument("--img_width", type=int, default=512, help="width of image")
parser.add_argument("--img_height", type=int, default=512, help="height of image")
parser.add_argument("--val_interval", type=int, default=10, help="how often to run validation") # 建议设小一点方便调试
parser.add_argument("--val_save_num", type=int, default=40, help="number of validation images to save")
parser.add_argument("--val_max_num", type=int, default=-1, help="max number of validation images for metrics calculation (-1 means all)")
parser.add_argument("--ckpt_interval", type=int, default=10, help="checkpoint interval")
parser.add_argument("--num_epochs", type=int, default=601, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches") # 12通道显存大，建议设小
parser.add_argument("--lr", type=float, default=0.0001,help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of 1st order momentum")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")

args = parser.parse_args()

# 路径设置
model_v = "PFSM"
exp_root = os.path.join("experiments", model_v)
os.makedirs(exp_root, exist_ok=True)
checkpoint_dir = os.path.join(exp_root, "checkpoint")
result_dir = os.path.join(exp_root, "results")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
log_path = os.path.join(exp_root, "train_log.txt")
if args.epoch == 0:
    with open(log_path, "w") as f:
        f.write(f"Start Training: {model_v}\n")
        # f.write("Epoch, Loss, Rec, VGG, Fre, Grad, Color\n")
        f.write("Epoch, Loss, Unified_Polar, VGG, Fre\n")
# checkpoint_dir = "checkpoints/%s/" % (model_v)
# os.makedirs(checkpoint_dir, exist_ok=True)
# os.makedirs(f"results/{model_v}/", exist_ok=True)

# ==================== 模型与 Loss 初始化 ====================
L2_G = torch.nn.MSELoss()
L1_G = torch.nn.SmoothL1Loss()
L_vgg = VGG19_PercepLoss() 
# L_hybrid = HybridPolarLoss(w_s0=1.0, w_dolp=1.0, w_grad=0.5)
L_unified = UnifiedMultiscalePolarLoss(init_alpha=0.84)

# DWT
dwt1 = DWTForward(J=1, wave='db1', mode='zero')
dwt2 = DWTForward(J=1, wave='db1', mode='zero')
dwt3 = DWTForward(J=1, wave='db1', mode='zero')
dwt4 = DWTForward(J=1, wave='db1', mode='zero')

# Initialize network
Generator = PFSMNet(in_channels=12, out_channels=12,img_size=args.img_width)
Lap = Laplace()

if torch.cuda.is_available():
    Generator = Generator.cuda()
    dwt1 = dwt1.cuda()
    dwt2 = dwt2.cuda()
    dwt3 = dwt3.cuda()
    dwt4 = dwt4.cuda()
    L2_G = L2_G.cuda()
    L1_G = L1_G.cuda()
    L_vgg = L_vgg.cuda()
    # L_hybrid = L_hybrid.cuda()
    L_unified = L_unified.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# # 加载权重
# if args.epoch == 0:
#     Generator.apply(Weights_Normal)
# else:
#     ckpt_path = os.path.join(checkpoint_dir, f"generator_{args.epoch}.pth")
#     if os.path.exists(ckpt_path):
#         Generator.load_state_dict(torch.load(ckpt_path))
#         print(f"Loaded model from {ckpt_path}")
#     else:
#         print(f"Warning: Checkpoint not found at {ckpt_path}")
# ==================== 智能权重加载 (支持跨模型架构) ====================
if args.epoch == 0:
    Generator.apply(Weights_Normal)
    print("Initializing weights from scratch...")
else:
    ckpt_path = os.path.join(checkpoint_dir, f"generator_{args.epoch}.pth")
    if os.path.exists(ckpt_path):
        print(f"-------- Loading Weights from Epoch {args.epoch} --------")
        
        # 1. 读取旧权重
        pretrained_dict = torch.load(ckpt_path)
        # 2. 获取新模型当前的权重字典
        model_dict = Generator.state_dict()
        
        # 3. 【核心步骤】筛选：只保留 名字相同 且 形状相同 的权重
        # 这样会自动过滤掉形状改变了的 gf_128.att_conv 等层
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
        
        # 4. 打印过滤掉的层（可选，让你知道哪些层被重置了）
        all_keys = set(model_dict.keys())
        loaded_keys = set(pretrained_dict.keys())
        missing_keys = all_keys - loaded_keys
        print(f"Skipped layers (Initialized from scratch): {len(missing_keys)}")
        # print(missing_keys) # 如果想看具体哪些层没加载，取消注释
        
        # 5. 更新模型字典并加载
        model_dict.update(pretrained_dict)
        Generator.load_state_dict(model_dict)
        print(f"Successfully loaded matching weights from {ckpt_path}")
        
    else:
        print(f"Error: Checkpoint not found at {ckpt_path}")
        # 如果找不到权重，不仅要报错，还可以选择是否初始化新模型
        # sys.exit()


# 优化器
optimizer_G = torch.optim.Adam(Generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# ==================== 数据加载 ====================
# transforms_ = [
#     transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize to [-1, 1]
# ]
transforms_ = [
    AspectRatioResize((args.img_height, args.img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# 使用你修改后的 data_utils (支持 I1-I4 读取)
dataloader = DataLoader(
    GetTrainingPairs(args.dataset_path, transforms_=transforms_),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
)

val_dataloader = DataLoader(
    GetValImage(args.dataset_path, transforms_=transforms_, sub_dir='validation'), 
    batch_size=1, 
    shuffle=False, 
    num_workers=1
)

# Resizers
resize_128 = Resize([args.img_height // 2, args.img_width // 2])
resize_64 = Resize([args.img_height // 4, args.img_width // 4])
resize_32 = Resize([args.img_height // 8, args.img_width // 8])
resize_16 = Resize([args.img_height // 16, args.img_width // 16])

# ==================== 训练循环 ====================
print(f"Start training from epoch {args.epoch}...")
print(f"Logs will be saved to: {log_path}")

for epoch in range(args.epoch, args.num_epochs):
    Generator.train()
    
    for i, batch in enumerate(dataloader):
        #  获取输入 (B, 12, H, W)
        imgs_distorted = Variable(batch["A"].type(Tensor))
        gt = Variable(batch["B"].type(Tensor))

        #  GT 多尺度处理 (DWT 和 Resize 都可以处理 12 通道)
        gll_128, _ = dwt1(gt)
        gll_64, _  = dwt2(gll_128)
        gll_32, _  = dwt3(gll_64)
        gll_16, _  = dwt4(gll_32)

        gt_128 = resize_128(gt)
        gt_64  = resize_64(gt)
        gt_32  = resize_32(gt)
        gt_16  = resize_16(gt)

        optimizer_G.zero_grad()

        # 前向传播
        out, ll = Generator(imgs_distorted)
        fake, fake_128, fake_64, fake_32, fake_16 = out[0], out[1], out[2], out[3], out[4]
        ll_128, ll_64, ll_32, ll_16 = ll[0], ll[1], ll[2], ll[3]

        # # 1重建损失 (在 12 通道上计算 L1)
        # loss_rec = 15 * (L1_G(fake, gt) + 
        #                  1/4 * L1_G(fake_128, gt_128) + 
        #                  1/8 * L1_G(fake_64, gt_64) + 
        #                  1/16 * L1_G(fake_32, gt_32) + 
        #                  1/32 * L1_G(fake_16, gt_16))

        # 2频域损失
        loss_fre = L1_G(ll_128, gll_128) + \
                   1/4 * L1_G(ll_64, gll_64) + \
                   1/8 * L1_G(ll_32, gll_32) + \
                   1/16 * L1_G(ll_16, gll_16)

        # 3.VGG 感知损失 
        fake_01 = (fake + 1.0) / 2.0
        gt_01   = (gt + 1.0) / 2.0
        # 计算 S0 (光强图)
        s0_fake, dop_fake = calculate_s0_dop(fake_01)
        s0_gt, dop_gt  = calculate_s0_dop(gt_01)  
        s0_fake_safe = (s0_fake / 2.0).clamp(0, 1) # 归一化到 [0, 1]
        s0_gt_safe   = (s0_gt / 2.0).clamp(0, 1)
        # 将 S0 转回 [-1, 1] 喂给 VGG (假设 VGGLoss 内部期望这个范围)
        s0_fake_norm = s0_fake_safe * 2.0 - 1.0
        s0_gt_norm   = s0_gt_safe * 2.0 - 1.0
        loss_vgg = 3 * L_vgg(s0_fake_norm, s0_gt_norm)
        
#         # 4.PFSM multi-scale restoration loss (物理一致性 + 梯度) ---
#         # 计算 S0 Loss, DoP Loss 和 Gradient Loss
#         loss_hybrid, loss_grad_val, loss_cont_val, loss_color_val = L_hybrid(s0_fake, s0_gt, dop_fake, dop_gt)
        loss_unified_val = L_unified(out, gt)
    
        # Total Loss
        # loss = loss_rec + loss_vgg + loss_fre + loss_hybrid
       # 权重可改
        loss = 15 * loss_unified_val + loss_vgg + loss_fre
        
        
        # 检查NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf detected in loss at epoch {epoch}, batch {i}, skipping this batch")
            optimizer_G.zero_grad()
            continue
        
        # 检查各个损失项
        # if torch.isnan(loss_rec) or torch.isnan(loss_vgg) or torch.isnan(loss_fre) or torch.isnan(loss_hybrid):
        #     print(f"Warning: NaN detected in loss components at epoch {epoch}, batch {i}")
        #     print(f"  loss_rec: {loss_rec.item()}, loss_vgg: {loss_vgg.item()}, loss_fre: {loss_fre.item()}, loss_hybrid: {loss_hybrid.item()}")
        if torch.isnan(loss_unified_val) or torch.isnan(loss_vgg) or torch.isnan(loss_fre):
            print(f"Warning: NaN detected in loss components at epoch {epoch}, batch {i}")
            print(f"  loss_unified: {loss_unified_val.item()}, loss_vgg: {loss_vgg.item()}, loss_fre: {loss_fre.item()}")
            optimizer_G.zero_grad()
            continue

        loss.backward()
        
        # 检查梯度
        total_grad_norm = 0
        for param in Generator.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"Warning: NaN/Inf detected in gradients at epoch {epoch}, batch {i}")
                    optimizer_G.zero_grad()
                    break
                total_grad_norm += param.grad.norm().item()
        
        # 梯度裁剪（更严格）
        torch.nn.utils.clip_grad_norm_(Generator.parameters(), max_norm=0.5)  # 从1.0降到0.5
        
        optimizer_G.step()

        # 日志输出
        # if i % 20 == 0:
        #     log_msg = "[Epoch %d/%d: batch %d/%d] [Loss: %.3f, Rec: %.3f, VGG: %.3f, Fre: %.3f, Grad: %.3f, Cont: %.3f, Color: %.3f]" % (
        #         epoch, args.num_epochs, i, len(dataloader),
        #         loss.item(), loss_rec.item(), loss_vgg.item(), loss_fre.item(), loss_grad_val.item(), loss_cont_val.item(), loss_color_val.item())
        if i % 20 == 0:
            log_msg = "[Epoch %d/%d: batch %d/%d] [Loss: %.3f, Unified: %.3f, VGG: %.3f, Fre: %.3f]" % (
                epoch, args.num_epochs, i, len(dataloader),
                loss.item(), loss_unified_val.item(), loss_vgg.item(), loss_fre.item())
            sys.stdout.write("\r" + log_msg)
            sys.stdout.flush()

            with open(log_path, "a") as f:
                f.write(log_msg + "\n")

    # ==================== 验证循环 ====================
# ==================== 验证循环 (修复维度报错版) ====================
  # ==================== 验证循环 (终极维度修复版) ====================
    if val_dataloader is not None and (epoch % args.val_interval == 0) and epoch > 0:
        Generator.eval()
        print(f"\n[Epoch {epoch}] Starting validation...")
        # val_save_dir = f"results/{model_v}/epoch_{epoch}"
        # os.makedirs(val_save_dir, exist_ok=True)
        val_save_dir = os.path.join(result_dir, f"epoch_{epoch}")
        os.makedirs(val_save_dir, exist_ok=True)

        total_psnr = 0.0
        total_ssim = 0.0
        count = 0

        with torch.no_grad():
            for j, val_batch in enumerate(val_dataloader):
                # 限制用于计算指标的最大图像数量
                if args.val_max_num > 0 and j >= args.val_max_num:
                    break
                
                val_in = Variable(val_batch["A"].type(Tensor))
                val_gt = Variable(val_batch["B"].type(Tensor))

                # 1. 推理
                val_out, _ = Generator(val_in)
                val_fake = val_out[0]

                # 2. 物理转换
                val_fake_01 = (val_fake + 1.0) / 2.0
                val_gt_01   = (val_gt + 1.0) / 2.0
                
                s0_fake, dop_fake = calculate_s0_dop(val_fake_01)
                s0_gt, dop_gt     = calculate_s0_dop(val_gt_01)

                # 3. 准备指标计算数据
                s0_fake_metric = (s0_fake / 2.0).clamp(0, 1)
                s0_gt_metric   = (s0_gt / 2.0).clamp(0, 1)

                # 【核心修复】防爆措施：确保 Tensor 是 4 维 [B, 3, H, W]
                # 如果只有 3 维 [3, H, W]，说明 Batch 维度丢失，手动加回来
                if s0_fake_metric.ndim == 3:
                    s0_fake_metric = s0_fake_metric.unsqueeze(0)
                if s0_gt_metric.ndim == 3:
                    s0_gt_metric = s0_gt_metric.unsqueeze(0)

                # 4. 遍历 Batch 计算指标
                for k in range(s0_fake_metric.size(0)):
                    # 移除padding并resize到512×512（与保存图像保持一致）
                    s0_fake_metric_clean = remove_padding_and_resize(s0_fake_metric[k:k+1], target_size=(512, 512))
                    s0_gt_metric_clean = remove_padding_and_resize(s0_gt_metric[k:k+1], target_size=(512, 512))
                    
                    # 此时 s0_fake_metric_clean[0] 必定是 [3, H, W]，tensor2img 不会再报错
                    img_fake_s0 = tensor2img(s0_fake_metric_clean[0])
                    img_gt_s0   = tensor2img(s0_gt_metric_clean[0])

                    # 计算指标（不需要crop_border，因为已经移除了padding）
                    psnr = calculate_psnr(img_fake_s0, img_gt_s0, crop_border=4)
                    ssim = calculate_ssim(img_fake_s0, img_gt_s0, crop_border=4)

                    total_psnr += psnr
                    total_ssim += ssim
                    count += 1

                # 5. 保存图片（使用可配置的数量）
                if j < args.val_save_num:
                    dop_fake_3c = dop_fake.repeat(1, 3, 1, 1)
                    dop_gt_3c   = dop_gt.repeat(1, 3, 1, 1)
                    
                    s0_fake_vis = (s0_fake / 2.0).clamp(0, 1)
                    s0_gt_vis   = (s0_gt / 2.0).clamp(0, 1)
                    
                    # 移除padding并resize到512×512（不包含padding）
                    s0_fake_vis = remove_padding_and_resize(s0_fake_vis, target_size=(512, 512))
                    s0_gt_vis = remove_padding_and_resize(s0_gt_vis, target_size=(512, 512))
                    dop_fake_3c = remove_padding_and_resize(dop_fake_3c, target_size=(512, 512))
                    dop_gt_3c = remove_padding_and_resize(dop_gt_3c, target_size=(512, 512))
                    
                    combined = torch.cat([s0_fake_vis, s0_gt_vis, dop_fake_3c, dop_gt_3c], dim=3)
                    save_image(combined, f"{val_save_dir}/val_{j}_psnr{psnr:.2f}.png")

        if count > 0:
            avg_psnr = total_psnr / count
            avg_ssim = total_ssim / count
            val_msg = f"[Epoch {epoch}] Val S0-PSNR: {avg_psnr:.4f}, S0-SSIM: {avg_ssim:.4f}"
        else:
            val_msg = f"[Epoch {epoch}] Warning: No validation images."
        
        print(val_msg)
        with open(log_path, "a") as f:
            f.write(val_msg + "\n")
    # 保存模型
    if (epoch % args.ckpt_interval == 0) and epoch > 0:
        # torch.save(Generator.state_dict(), "checkpoints/%s/generator_%d.pth" % (model_v, epoch))
        save_name = os.path.join(checkpoint_dir, f"generator_{epoch}.pth")
        torch.save(Generator.state_dict(), save_name)
        print(f"Saved model to {save_name}")
