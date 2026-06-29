import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nets.advanced_fusion import ImprovedMSFFN
from einops import repeat, rearrange

# 尝试导入 Mamba 及其底层算子
try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    print("!!! Warning: mamba_ssm not found. Mocking for debug. !!!")
    Mamba = nn.Identity
    selective_scan_fn = None

# ================================================================
# 调试工具
# ================================================================
def debug_check(x, name=""):
    if torch.isnan(x).any():
        print(f"!!! [CRITICAL] NaN detected in {name} !!!")
        return True
    if torch.isinf(x).any():
        print(f"!!! [CRITICAL] Inf detected in {name} !!!")
        return True
    return False

# ================================================================
# 1. 预处理减震模块 (Input Stabilizer)
# ================================================================
class InputStabilizer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gn = nn.GroupNorm(4, channels)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

# ================================================================
# [核心升级] SS2D: 4-Way Cross Scan (used in the PFSM state-space branch)
# ----------------------------------------------------------------
# 作用: four-directional scanning for [左->右, 右->左, 上->下, 下->上] 全向感知
# ================================================================
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) 
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) 
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) 
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) 

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A) 
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D) 
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) 
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) 
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) 
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) 
        
        # 核心算子调用
        out_y = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        # 输入: [B, H, W, C]
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
# ================================================================
# [集成版] PolarMISM Block (使用 SS2D 4路扫描 + SAF-MSFFN)
# ================================================================
class PolarMISMBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        dim = in_channels
        if out_channels is None:
            out_channels = in_channels
            
        self.stabilizer = InputStabilizer(dim)
        
        # 1. 扫描引擎升级: 使用 SS2D (4-Way)
        self.mamba_core = SS2D(d_model=dim, d_state=16, d_conv=3, expand=2) 
        
        # 2. 融合修复: 使用 SAF-MSFFN (hcMA)
        self.msffn = ImprovedMSFFN(dim)  # 升级为改进版本 
        # 条件融合层: 将低频特征融入高频流
        self.cond_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False), # 拼接后降维
            nn.GroupNorm(4, dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x ,condition=None):
        # x: [B, C, H, W]
        debug_check(x, "MISM Input")
        input_shortcut = x
        
        # 预处理
        x = self.stabilizer(x)
        
        if condition is not None:# 简单拼接融合
            x = torch.cat([x, condition], dim=1)
            x = self.cond_fusion(x)
        
        # 4路扫描 (需要转换维度适应 SS2D)
        # SS2D 期望 [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.mamba_core(x)
        # 变回 [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # 差分融合修复
        out = self.msffn(x)
        
        return out + input_shortcut
