"""
  Modules for processing training/validation data
"""
import os
import torch
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AspectRatioResize:
    """
    保持宽高比的Resize
    解决固定尺寸resize导致的边界裁剪问题
    """
    def __init__(self, target_size=(512, 512), pad_color=(128, 128, 128)):
        """
        Args:
            target_size: 目标尺寸 (width, height)
            pad_color: padding颜色 (RGB)
        """
        self.target_size = target_size
        self.pad_color = pad_color
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image: 保持宽高比，padding到target_size
        """
        target_w, target_h = self.target_size
        orig_w, orig_h = img.size
        
        # 计算缩放比例（保持宽高比）
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize（保持宽高比，使用高质量插值）
        img_resized = img.resize((new_w, new_h), Image.BICUBIC)
        
        # 创建目标尺寸的空白图像（使用padding颜色）
        img_padded = Image.new(img.mode, (target_w, target_h), self.pad_color)
        
        # 计算padding位置（居中）
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        
        # 粘贴resize后的图像
        img_padded.paste(img_resized, (paste_x, paste_y))
        
        return img_padded

class ReflectivePaddingResize:
    """
    反射padding + resize
    在resize前先padding，确保边界信息不丢失
    """
    def __init__(self, target_size=(512, 512), pad_ratio=0.1):
        """
        Args:
            target_size: 目标尺寸 (width, height)
            pad_ratio: padding比例（相对于原始尺寸）
        """
        self.target_size = target_size
        self.pad_ratio = pad_ratio
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image: 反射padding后resize到target_size
        """
        target_w, target_h = self.target_size
        orig_w, orig_h = img.size
        
        # 计算padding大小
        pad_w = int(orig_w * self.pad_ratio)
        pad_h = int(orig_h * self.pad_ratio)
        
        # 反射padding
        img_array = np.array(img)
        img_padded = np.pad(img_array, 
                           ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                           mode='reflect')
        img_padded = Image.fromarray(img_padded.astype(np.uint8))
        
        # Resize到目标尺寸
        img_resized = img_padded.resize((target_w, target_h), Image.BICUBIC)
        
        return img_resized

class GetTrainingPairs(Dataset):
    """ Common data pipeline to organize and generate
         training pairs for various datasets
    """
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.root = root
        # self.filesA, self.filesB = self.get_file_paths(root)
        # self.len = min(len(self.filesA), len(self.filesB))
        self.dirs_A = ['I1', 'I2', 'I3', 'I4'] # 输入的4个文件夹
        self.dirs_B = ['T1',  'T2',  'T3',  'T4']  # 真值的4个文件夹
        
        self.files = self.get_file_paths(root)
        self.len = len(self.files)


    def __getitem__(self, index):
        # img_A = Image.open(self.filesA[index % self.len])
        # img_B = Image.open(self.filesB[index % self.len])
        # if np.random.random() < 0.5:
        #     img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
        #     img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        # img_A = self.transform(img_A)
        # img_B = self.transform(img_B)
        # return {"A": img_A, "B": img_B}
        # 获取基准文件名 (例如 "image_001.png")
        filename = os.path.basename(self.files[index % self.len])
        imgs_A = []
        imgs_B = []
        
        # 随机翻转决策：生成一个随机数，所有 8 张图共用
        flip = np.random.random() < 0.5
        
        # 1. 读取 4 个输入角度 (I1-I4)
        for d in self.dirs_A:
            path = os.path.join(self.root, d, filename)
            img = Image.open(path).convert("RGB")
            if flip:
                img = Image.fromarray(np.array(img)[:, ::-1, :], "RGB")
            imgs_A.append(self.transform(img))
            
        # 2. 读取 4 个真值角度 (T1-T4)
        for d in self.dirs_B:
            path = os.path.join(self.root, d, filename)
            img = Image.open(path).convert("RGB")
            if flip:
                img = Image.fromarray(np.array(img)[:, ::-1, :], "RGB")
            imgs_B.append(self.transform(img))
            
        # 3. 拼接: List[3, H, W] * 4 -  Tensor[12, H, W]
        tensor_A = torch.cat(imgs_A, dim=0) 
        tensor_B = torch.cat(imgs_B, dim=0)
        
        return {"A": tensor_A, "B": tensor_B}

    def __len__(self):
        return self.len

#     def get_file_paths(self, root):
#         filesA, filesB = [], []
#         filesA += sorted(glob.glob(os.path.join(root, 'trainA') + "/*.*"))
#         filesB += sorted(glob.glob(os.path.join(root, 'trainB') + "/*.*"))

#         print("Train_dataset", len(filesA))
#         return filesA, filesB
    def get_file_paths(self, root):
        # 只要 I1 里有的图，默认其他 7 个文件夹里也有
        ref_dir = os.path.join(root, self.dirs_A[0])  # 即 .../I1
        # 读取所有图片文件
        files = sorted(glob.glob(os.path.join(ref_dir, "*.*")))
        print(f"Polar Dataset: Found {len(files)} scenes (indexed by {self.dirs_A[0]})")
        # 我们只需要返回这一个列表，不需要返回 filesA, filesB 了
        return files

# class GetValImage(Dataset):
#     """ Common data pipeline to organize and generate
#          vaditaion samples for various datasets
#     """
#     def __init__(self, root, transforms_=None, sub_dir='validation'):
#         self.transform = transforms.Compose(transforms_)
#         self.files = self.get_file_paths(root)
#         self.len = len(self.files)

#     def __getitem__(self, index):
#         img_val = Image.open(self.files[index % self.len])
#         img_val = self.transform(img_val)
#         return {"val": img_val}

#     def __len__(self):
#         return self.len

#     def get_file_paths(self, root):
#         files = []
#         files += sorted(glob.glob(os.path.join(root, 'trainB') + "/*.*"))
#         #print("validation", len(files))
#         return files

class GetValImage(Dataset):
    """ Common data pipeline to organize and generate
         validation samples for various datasets (Paired Version)
    """
#     def __init__(self, root, transforms_=None, sub_dir='validation'):
#         self.transform = transforms.Compose(transforms_)
#         # 自动拼接路径：root + sub_dir (例如 UIEB/validation)
#         if sub_dir:
#             self.root = os.path.join(root, sub_dir)
#         else:
#             self.root = root
#         # self.filesA, self.filesB = self.get_file_paths(self.root)
#         # self.len = min(len(self.filesA), len(self.filesB))
    
#         self.dirs_A = ['I1', 'I2', 'I3', 'I4']
#         self.dirs_B = ['T1', 'T2', 'T3', 'T4']
#         base_dir = os.path.join(self.root, self.dirs_A[0])
#         if os.path.exists(base_dir):
#             self.files = sorted(glob.glob(os.path.join(base_dir, "*.*")))
#         else:
#             print(f"Warning: Validation folder {base_dir} not found!")
#             self.files = []
            
#         self.len = len(self.files)
#         print(f"Polar Val Dataset: Found {self.len} pairs in {self.root}")   
  
    def __init__(self, root, transforms_=None, sub_dir='validation'):
        self.transform = transforms.Compose(transforms_)
        
        # 路径拼接
        if sub_dir:
            self.root = os.path.join(root, sub_dir)
        else:
            self.root = root
            
        # [修改点5] 文件夹名称需与训练集保持一致
        self.dirs_A = ['I1', 'I2', 'I3', 'I4']
        self.dirs_B = ['T1', 'T2', 'T3', 'T4']
        
        # 获取文件列表
        self.files = self.get_file_paths(self.root)
        self.len = len(self.files)
      
    def __getitem__(self, index):
#         img_A = Image.open(self.filesA[index % self.len])
#         img_B = Image.open(self.filesB[index % self.len])
        
#         #验证集取消随机翻转！  
#         img_A = self.transform(img_A)
#         img_B = self.transform(img_B)  
#         # 返回字典，键名与训练集保持一致，方便代码复用
#         return {"A": img_A, "B": img_B}
        filename = os.path.basename(self.files[index % self.len])
        imgs_A = []
        imgs_B = []
        
        # 验证集直接读取，不翻转
        for d in self.dirs_A:
            path = os.path.join(self.root, d, filename)
            img = Image.open(path).convert("RGB")
            imgs_A.append(self.transform(img))
            
        for d in self.dirs_B:
            path = os.path.join(self.root, d, filename)
            img = Image.open(path).convert("RGB")
            imgs_B.append(self.transform(img))
            
        tensor_A = torch.cat(imgs_A, dim=0) 
        tensor_B = torch.cat(imgs_B, dim=0)
        
        return {"A": tensor_A, "B": tensor_B}

    def __len__(self):
        return self.len

    def get_file_paths(self, root):

        ref_dir = os.path.join(root, self.dirs_A[0])
        
        if os.path.exists(ref_dir):
            files = sorted(glob.glob(os.path.join(ref_dir, "*.*")))
        else:
            print(f"Warning: Validation folder {ref_dir} not found!")
            files = []
            
        print(f"Polar Val Dataset: Found {len(files)} pairs in {root}")
        return files


