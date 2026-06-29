import os
import argparse
import csv
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# 1. 导入数据工具与物理计算工具
from utils.data_utils import GetValImage
from utils.polar_utils import calculate_s0_dop

# 2. 导入 metrics
try:
    from utils.metrics import calculate_psnr, calculate_ssim, tensor2img
except ImportError:
    print("❌ 警告: 无法导入 metrics.py。请检查路径。")
    import sys; sys.exit()

# 3. 导入你最新训练的模型
from nets.pfsm import PFSMNet 


# ==================== 绘图专用：特征提取 Hook ====================
activation_dict = {}

def get_activation(name):
    """获取网络中间层特征图的 Hook 函数"""
    def hook(model, input, output):
        if isinstance(output, tuple):
            activation_dict[name] = [o.detach() for o in output]
        else:
            activation_dict[name] = output.detach()
    return hook

def save_feature_as_heatmap(tensor, save_path):
    """将高维张量转换为 Jet 伪彩色热力图并保存"""
    # 沿着通道维度求平均 (1, C, H, W) -> (1, 1, H, W)
    feat_mean = torch.mean(tensor, dim=1, keepdim=True)
    # 归一化到 0-1
    feat_mean = feat_mean - feat_mean.min()
    feat_mean = feat_mean / (feat_mean.max() + 1e-8)
    # 转换为 OpenCV 格式的伪彩色热力图
    feat_np = feat_mean.squeeze().cpu().numpy()
    feat_np = np.uint8(255 * feat_np)
    heatmap = cv2.applyColorMap(feat_np, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heatmap)
# ===============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="PFSM Test and Feature Extraction Script")
    parser.add_argument("--dataset_path", type=str, default="./data/test", help="测试集路径")
    parser.add_argument("--sub_dir", type=str, default="test", help="测试图片子文件夹名")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 .pth 权重路径")
    parser.add_argument("--save_dir", type=str, default="./results/pfsm_test", help="结果保存主目录")
    parser.add_argument("--img_width", type=int, default=512)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1, help="测试时必须保持 batch_size=1")
    parser.add_argument("--gpu_id", type=str, default="0")
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建所有需要的子目录 (为了论文拼图准备)
    s0_save_dir = os.path.join(args.save_dir, "S0_Pred")
    dop_save_dir = os.path.join(args.save_dir, "DoP_Pred")
    compare_save_dir = os.path.join(args.save_dir, "Compare_Visual")
    multiscale_save_dir = os.path.join(args.save_dir, "MultiScale_Outputs")
    feature_save_dir = os.path.join(args.save_dir, "Feature_Maps")
    
    for d in [s0_save_dir, dop_save_dir, compare_save_dir, multiscale_save_dir, feature_save_dir]:
        os.makedirs(d, exist_ok=True)

    # ==================== 1. 数据加载 ====================
    transforms_ = [
        transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    
    test_dataset = GetValImage(args.dataset_path, transforms_=transforms_, sub_dir=args.sub_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"✅ 数据集加载成功: 包含 {len(test_dataset)} 张测试图片。")

    # ==================== 2. 模型初始化与 Hook 注册 ====================
    print("✅ 正在初始化 PFSM 模型...")
    # ⚠️ 请确保这里的 in_channels / out_channels 与你最终版 train.py 严格一致
    model = PFSMNet(in_channels=12, out_channels=12, img_size=args.img_width).to(device)

    # 💡 挂载钩子提取中间特征图 (⚠️ 需根据你模型代码中真实的变量名修改!!!)
    try:
        # 提取 SFTM 模块输出的调制特征或 PAoP
        # model.sftm.register_forward_hook(get_activation('Stage1_SFTM')) 
        
        # 提取第一个降采样层级 PA-ADFL 的融合特征
        # model.encoders[0].pa_adfl.register_forward_hook(get_activation('Level_1_PA_ADFL'))
        print("💡 Hook 注册提示: 请在代码中取消注释并填入正确的模块变量名以提取热力图。")
    except AttributeError as e:
        print(f"⚠️ Hook 注册失败，请检查模型内部变量名称: {e}")

    # ==================== 3. 权重加载 ====================
    if os.path.exists(args.checkpoint):
        print(f"✅ 正在加载最优权重: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except Exception as e:
            print("⚠️ 严格加载失败，尝试非严格模式...")
            model.load_state_dict(new_state_dict, strict=False)
    else:
        raise FileNotFoundError("❌ 未找到权重文件！")

    # ==================== 4. 开始测试 ====================
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    # 准备写入 CSV 记录结果
    csv_file_path = os.path.join(args.save_dir, "test_metrics_results.csv")
    with open(csv_file_path, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Image_Name', 'PSNR', 'SSIM'])

        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
        
        with torch.no_grad():
            for i, batch in progress_bar:
                img_name = batch.get("name", [f"image_{i:04d}"])[0] 
                img_name = os.path.splitext(img_name)[0]

                val_in = batch["A"].to(device)
                val_gt = batch["B"].to(device) 
                
                # --- 前向推理 ---
                model_returns = model(val_in)
                if isinstance(model_returns, tuple) and len(model_returns) == 2:
                    outputs, _ = model_returns  # 拆包，拿出真正的图像列表 outputs
                else:
                    outputs = model_returns
                
                # --- 获取最终输出与多尺度保存 ---
                if isinstance(outputs, (list, tuple)):
                    val_fake = outputs[0] # 取列表最后一个作为全分辨率最优预测
                    
                    # 遍历保存 5 个尺度的预测图 (论文画图神器!)
                    for scale_idx, scale_fake in enumerate(outputs):
                        s_fake_01 = (scale_fake + 1.0) / 2.0
                        s0_scale, _ = calculate_s0_dop(s_fake_01)
                        s0_scale_metric = (s0_scale / 2.0).clamp(0, 1)
                        # 命名为 scale_1 到 scale_5，数字越小分辨率越低（对应你的降采样）
                        save_image(s0_scale_metric[0], os.path.join(multiscale_save_dir, f"{img_name}_scale_{len(outputs)-scale_idx}.png"))
                else:
                    val_fake = outputs
                
                # --- 物理处理与限幅 ---
                val_fake_01 = (val_fake + 1.0) / 2.0
                val_gt_01   = (val_gt + 1.0) / 2.0
                
                s0_fake, dop_fake = calculate_s0_dop(val_fake_01)
                s0_gt, dop_gt     = calculate_s0_dop(val_gt_01)

                s0_fake_metric = (s0_fake / 2.0).clamp(0, 1)
                s0_gt_metric   = (s0_gt / 2.0).clamp(0, 1)
                
                if s0_fake_metric.ndim == 3: s0_fake_metric = s0_fake_metric.unsqueeze(0)
                if s0_gt_metric.ndim == 3:   s0_gt_metric = s0_gt_metric.unsqueeze(0)
                
                # --- 计算评估指标 (标准做法) ---
                img_fake_np = tensor2img(s0_fake_metric[0])
                img_gt_np   = tensor2img(s0_gt_metric[0])
                
                # 💡 学术标准：放弃 80% 裁剪，采用 crop_border=4 去除卷积伪影边缘
                current_psnr = calculate_psnr(img_fake_np, img_gt_np, crop_border=4)
                current_ssim = calculate_ssim(img_fake_np, img_gt_np, crop_border=4)
                
                total_psnr += current_psnr
                total_ssim += current_ssim
                count += 1
                
                # --- 单独保存高清原图 (S0 与 DoP) ---
                save_image(s0_fake_metric[0], os.path.join(s0_save_dir, f"{img_name}_S0.png"))
                dop_fake_vis = dop_fake.repeat(1, 3, 1, 1).clamp(0, 1)
                save_image(dop_fake_vis[0], os.path.join(dop_save_dir, f"{img_name}_DoP.png"))
                
                # 保存快速对比拼图 (为了你自己查阅方便)
                dop_gt_vis = dop_gt.repeat(1, 3, 1, 1).clamp(0, 1)
                combined = torch.cat([s0_fake_metric, s0_gt_metric, dop_fake_vis, dop_gt_vis], dim=3)
                save_image(combined, os.path.join(compare_save_dir, f"{img_name}_compare.png"))
                
                # --- 保存中间层特征热力图 ---
                for layer_name, feat_tensors in activation_dict.items():
                    if isinstance(feat_tensors, list): # 处理类似 DWT 输出的 Tuple
                        for idx, feat in enumerate(feat_tensors):
                            feat_path = os.path.join(feature_save_dir, f"{img_name}_{layer_name}_part{idx}.png")
                            save_feature_as_heatmap(feat, feat_path)
                    else:
                        feat_path = os.path.join(feature_save_dir, f"{img_name}_{layer_name}.png")
                        save_feature_as_heatmap(feat_tensors, feat_path)

                # 记录到 CSV
                csv_writer.writerow([img_name, round(current_psnr, 4), round(current_ssim, 4)])
                progress_bar.set_postfix({"PSNR": f"{current_psnr:.2f}", "SSIM": f"{current_ssim:.4f}"})
                    
    # ==================== 5. 打印最终总结 ====================
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    
    print("\n" + "="*50)
    print(f" 🏆 测试圆满完成! ")
    print(f" 使用权重: {os.path.basename(args.checkpoint)}")
    print(f" 测试数量: {count} 张图片")
    print(f" 平均 PSNR: {avg_psnr:.4f} dB")
    print(f" 平均 SSIM: {avg_ssim:.4f}")
    print(f" 详细指标已保存至: {csv_file_path}")
    print(f" 所有绘图素材已分类保存在: {args.save_dir}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

