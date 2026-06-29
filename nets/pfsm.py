"""PFSM network for underwater polarization image restoration."""


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import matplotlib.pyplot as plt
from .Dwt import DWTForward
import math
import numpy as np
from torchvision.utils import save_image
from .physics_module import SFTM
from .mism_2d import PolarMISMBlock
from .advanced_fusion import AdvancedDiffusionFusionLayer


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        # 将4个分量(LL, LH, HL, HH)拼接后融合
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            # 如果 batch_size 较小，建议用 GroupNorm 替代 BatchNorm2d
            nn.GroupNorm(4, out_ch), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        # 【核心】打包所有频率信息，防止丢失
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

class GELU(nn.Module):

    def forward(self, x):
        return F.gelu(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        #self.drop = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.drop(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
        """
        reference: wzlxjtu/PositionalEncoding2D

        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        return x

class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, in_channel = 3, out_channel = 64, depth = 1, head = 4, drop_path = 0.2):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.query = MultiHeadDense(out_channel, bias=False)
        self.key = MultiHeadDense(out_channel, bias=False)
        self.value = MultiHeadDense(out_channel, bias=False)
        self.qkv = nn.Linear(out_channel, out_channel * 3, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.pe = PositionalEncodingPermute2D(out_channel)
        self.Conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel*3, out_channels=out_channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),)
        self.depth = depth
        self.norm1 = nn.LayerNorm(out_channel)
        self.norm2 = nn.LayerNorm(out_channel)
        self.drop_path = DropPath(drop_path)
        self.mlp = Mlp(in_features=out_channel, hidden_features=out_channel*2, act_layer=GELU)

    def forward(self, x):
        B, C, _, H, W = x.shape
        HL = torch.chunk(x, dim=2, chunks=3)
        lh = HL[0].view(B, C, H, W)
        hh = HL[1].view(B, C, H, W)
        hl = HL[2].view(B, C, H, W)
        # print("...",lh.shape)
        contact = torch.cat((lh, hh, hl), 1)
        #print("...", contact.shape)
        x = self.Conv0(contact)
        #print("...", x.shape)
        b, c, h, w = x.shape
        input = x
        pe = self.pe(input)
        input = input + pe

        for i in range(0, self.depth):

            input = input.reshape(b, c, h * w).permute(0, 2, 1)  #[b, h*w, c
            #print("...:", Q.shape, A.shape)
            qkv = self.norm1(input)
            Q = self.query(qkv)
            K = self.key(qkv)
            A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(c))  #[b, h*w, h*w]
            V = self.value(qkv)
            #print("...:", A.shape, V.shape)
            attn = torch.bmm(A, V)

            #FFN
            attn = input + self.drop_path(attn)
            input = attn + self.drop_path(self.mlp(self.norm2(attn)))
            input = input.permute(0, 2, 1).reshape(b, c, h, w)
            x = input
        #print("sucessful")
        return x

class CALayer(nn.Module):
    def __init__(self, channel, reduction=8, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CABlock(nn.Module):
    """ Residual attention block
    """
    # [修改] 增加 out_c 参数，默认 3
    def __init__(self, in_size=3, out_size=64, out_c=3):
        super(CABlock, self).__init__()

        self.Conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.CA = CALayer(out_size)

        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_size*3, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # [修改] 这里使用传入的 out_c (12)，而不是写死 3
            nn.Conv2d(in_channels=out_size, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_size * 3, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )


    def forward(self, x):
        #print("...", x.shape)

        x1 = self.Conv0(x)
        x2 = self.Conv1(x1)
        x3 = self.CA(x2)
        contact = torch.cat((x1, x2, x3), 1)
        out = self.Conv2(contact)
        feature_map = self.Conv3(contact)
        return out, feature_map

#Dual-domain Fusion Block(DFB)
class DFB(nn.Module):
    def __init__(self, in_channel = 64, out_channel = 3, bias = False):
        super(DFB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel*3, in_channel, kernel_size=3, padding=1, stride=1, bias=bias),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, in_channel, kernel_size=3, padding=1, stride=1, bias=bias),
            nn.Sigmoid(),
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 8, kernel_size=3, padding=1, stride=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(in_channel // 8),
            nn.Conv2d(in_channel // 8, in_channel, kernel_size=3, padding=1, stride=1, bias=bias),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.Up_sample = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.conv_finnay = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=in_channel//4, out_channels = out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),)

    def forward(self, h_img, p_img, middle_img = None):
        if middle_img == None:
            middle_img = h_img

        img = torch.cat((h_img, p_img, middle_img), 1)
        x1 = self.conv1(img)
        out = self.conv_finnay(x1)

        x2 = self.conv2(out)
        x3 = self.conv3(x1)
        x4 = x3 * x2 + x1
        out_img = self.Up_sample(x4)

        return out, out_img

class Pixel_Restruction(nn.Module):
    """ Pixel_Restruction Module
    """

    # def __init__(self, out_size=64):
    #     super(Pixel_Restruction, self).__init__()
    # [修改] 增加 in_c 和 out_c 参数，默认设为 12
    def __init__(self, out_size=64, in_c=12, out_c=12):
        super(Pixel_Restruction, self).__init__()

        self.DFB1 = DFB(in_channel=64,out_channel=out_c)
        self.DFB2 = DFB(in_channel=64,out_channel=out_c)
        self.DFB3 = DFB(in_channel=64,out_channel=out_c)
        self.DFB4 = DFB(in_channel=64,out_channel=out_c)

        self.Conv256 = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=in_c, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.Conv128 = nn.Sequential(
            nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.Conv64 = nn.Sequential(
            nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.Conv32 = nn.Sequential(
            nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.Conv16 = nn.Sequential(
            nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.conv_finnay = nn.Sequential(
            nn.Conv2d(in_channels=out_size, out_channels=out_size//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size//4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(in_channels=out_size//4, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=out_size//4, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, D_128, D_64, D_32, D_16):
        '''Pixel Domain Module(PDM)'''
        P_256 = self.Conv256(x)
        P_128 = self.Conv128(P_256)
        P_64 = self.Conv64(P_128)
        P_32 = self.Conv32(P_64)
        P_16 = self.Conv16(P_32)

        '''Reconstruction Module(REM)'''
        Out_16, img_32 = self.DFB1(D_16, P_16)
        Out_32, img_64 = self.DFB2(D_32, P_32, img_32)
        Out_64, img_128 = self.DFB3(D_64, P_64, img_64)
        Out_128, img_256 = self.DFB4(D_128, P_128, img_128)


        img_256 = img_256 + P_256
        out_256 = self.conv_finnay(img_256)
        return out_256, Out_128, Out_64, Out_32, Out_16

# Adapt high-frequency DWT bands to the 2D state-space branch.
class PolarMambaBlock(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, depth=1, resolution=128):
            super().__init__()

            self.reduce_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channel*3, out_channels=out_channel, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(out_channel),
                nn.GroupNorm(num_groups=4, num_channels=out_channel),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

            self.mamba_core = PolarMISMBlock(in_channels=out_channel, out_channels=out_channel)
            # ===== [新增]: AoP 物理先验空间门控网络 =====
            self.aop_gate = nn.Sequential(
                nn.Conv2d(3, out_channel, kernel_size=3, padding=1, bias=False), # AoP 是3通道
                nn.GroupNorm(num_groups=4, num_channels=out_channel),
                nn.Sigmoid() # 输出 0~1 的空间掩码
            )
            
#     def forward(self, x ,condition=None):
#         # x shape: [B, C, 3, H, W] (来自 DWT 输出)
#         B, C, _, H, W = x.shape

#         # 1. 数据重组：把 3 个高频带拼起来
#         # split
#         high_bands = torch.chunk(x, dim=2, chunks=3)
#         lh = high_bands[0].view(B, C, H, W)
#         hh = high_bands[1].view(B, C, H, W)
#         hl = high_bands[2].view(B, C, H, W)

#         # concat -> [B, 3*C, H, W]
#         feature = torch.cat((lh, hh, hl), 1)

#         # 2. 降维 -> [B, out_channel, H, W]
#         feature = self.reduce_conv(feature)

#         # 3. 喂给 Mamba
#         out = self.mamba_core(feature, condition=condition)

#         return out
    def forward(self, x, condition=None, aop_prior=None):
        B, C, _, H, W = x.shape
        high_bands = torch.chunk(x, dim=2, chunks=3)
        lh = high_bands[0].view(B, C, H, W)
        hh = high_bands[1].view(B, C, H, W)
        hl = high_bands[2].view(B, C, H, W)

        # 降维
        feature = torch.cat((lh, hh, hl), 1)
        feature = self.reduce_conv(feature)

        # ===== [新增]: 施加 AoP 物理门控 =====
        if aop_prior is not None:
            # 动态下采样 AoP 先验至当前特征图分辨率
            aop_scaled = F.interpolate(aop_prior, size=(H, W), mode='bilinear', align_corners=False)
            gate = self.aop_gate(aop_scaled)
            feature = feature * gate  # 门控相乘：引导高频算力聚焦于偏振边缘

        out = self.mamba_core(feature, condition=condition)
        return out


class PFSMNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=12, embed_dim=64,img_size=512):
        super().__init__()
        
         # 初始化物理模块
        res_1 = img_size // 2   # 512 -> 256
        res_2 = img_size // 4   # 512 -> 128
        res_3 = img_size // 8   # 512 -> 64
        res_4 = img_size // 16  # 512 -> 32   
        self.sftm = SFTM(in_channels=in_channels)

        '''......Discrete wavelet transform......'''
        self.dwt1 = DWTForward(J=1, wave='db1', mode='zero')
        self.dwt2 = DWTForward(J=1, wave='db1', mode='zero')
        self.dwt3 = DWTForward(J=1, wave='db1', mode='zero')
        self.dwt4 = DWTForward(J=1, wave='db1', mode='zero')

        '''......[新增] Packed Wavelet Downsampling (用于主干分支)......'''
        # 注意：这里 out_ch 设为 in_channels，保持和原来 ll 的通道数一致，方便接入 L_128
        self.down_wt1 = Down_wt(in_channels, in_channels)
        self.down_wt2 = Down_wt(in_channels, in_channels)
        self.down_wt3 = Down_wt(in_channels, in_channels)
        self.down_wt4 = Down_wt(in_channels, in_channels)



        self.MHSA1 = PolarMambaBlock(in_channel=in_channels, out_channel=embed_dim, depth=1, resolution=res_1)
        self.MHSA2 = PolarMambaBlock(in_channel=in_channels, out_channel=embed_dim, depth=1, resolution=res_2)
        self.MHSA3 = PolarMambaBlock(in_channel=in_channels, out_channel=embed_dim, depth=1, resolution=res_3)
        self.MHSA4 = PolarMambaBlock(in_channel=in_channels, out_channel=embed_dim, depth=1, resolution=res_4)

        self.conv128 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.conv64 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv32 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        '''......Global average pooling......'''
        self.avg_pool = nn.AdaptiveAvgPool2d(1)


        self.L_128 = CABlock(in_channels, embed_dim,out_c=in_channels)
        self.L_64 = CABlock(in_channels, embed_dim,out_c=in_channels)
        self.L_32 = CABlock(in_channels, embed_dim,out_c=in_channels)
        self.L_16 = CABlock(in_channels, embed_dim,out_c=in_channels)
        
        
        '''......Dual-domain adaptive fusion layer......'''
        self.gf_128 = AdvancedDiffusionFusionLayer(in_channels_guide=embed_dim, in_channels_source=embed_dim, out_channels=embed_dim)
        self.gf_64  = AdvancedDiffusionFusionLayer(in_channels_guide=embed_dim, in_channels_source=embed_dim, out_channels=embed_dim)
        self.gf_32  = AdvancedDiffusionFusionLayer(in_channels_guide=embed_dim, in_channels_source=embed_dim, out_channels=embed_dim)
        self.gf_16  = AdvancedDiffusionFusionLayer(in_channels_guide=embed_dim, in_channels_source=embed_dim, out_channels=embed_dim)
        
        '''......Pixel-Reconstruction block......'''
        # self.pix_rec = Pixel_Restruction()
        self.pix_rec = Pixel_Restruction(out_size=embed_dim, in_c=in_channels, out_c=out_channels)

        self.l1 = torch.nn.SmoothL1Loss().cuda()  # similarity loss (l1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        
        
        x, aop_prior = self.sftm(x)

        # ================= Level 1 =================
        # 1. Mamba 分支需要的 High Freq (high-frequency branch)
        _, lh_128 = self.dwt1(x) 
        
        # 2. [新增] 主干分支需要的 Packed Features (全信息)
        x_packed_128 = self.down_wt1(x) 

        # 3. 处理
        # 把 x_packed_128 传给 L_128，而不是low-frequency features
        ll_128, l_128 = self.L_128(x_packed_128)  
        
        h_128 = self.MHSA1(lh_128[0], condition=l_128, aop_prior=aop_prior)
        D_128 = self.gf_128(l_128, h_128)

        # ================= Level 2 =================
        # 注意：下一层的输入应该是上一层的 "低频输出" (ll_128)
        # 这里的 ll_128 已经是经过 L_128 处理过的特征了
        
        _, lh_64 = self.dwt2(ll_128)          # Mamba 分支取高频
        x_packed_64 = self.down_wt2(ll_128)   # [新增] 主干分支取全信息

        ll_64, l_64 = self.L_64(x_packed_64)  # 传入全信息
        
        h_64 = self.MHSA2(lh_64[0], condition=l_64, aop_prior=aop_prior)
        D_64 = self.gf_64(l_64, h_64)

        # ================= Level 3 =================
        _, lh_32 = self.dwt3(ll_64)
        x_packed_32 = self.down_wt3(ll_64)    # [新增]

        ll_32, l_32 = self.L_32(x_packed_32)  # 传入全信息
        
        h_32 = self.MHSA3(lh_32[0], condition=l_32, aop_prior=aop_prior) 
        D_32 = self.gf_32(l_32, h_32)

        # ================= Level 4 =================
        _, lh_16 = self.dwt4(ll_32)
        x_packed_16 = self.down_wt4(ll_32)    # [新增]

        ll_16, l_16 = self.L_16(x_packed_16)  # 传入全信息
        
       
        h_16 = self.MHSA4(lh_16[0], condition=l_16, aop_prior=aop_prior) 
        D_16 = self.gf_16(l_16, h_16)

        # ================= Output =================
        out_256, out_128, out_64, out_32, out_16 = self.pix_rec(x, D_128, D_64, D_32, D_16)
        out = [out_256, out_128, out_64, out_32, out_16]
        ll = [ll_128, ll_64, ll_32, ll_16]
        return out, ll

    def forward(self, x):
        x = self.forward_features(x)
        return x


