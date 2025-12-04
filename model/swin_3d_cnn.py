import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange

# 来自: yiming0110/3d-cnn-vswinformer/3D-CNN-VswinFormer-main/model.py

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
    def forward(self, x):
        batch_size, channels, _, _, _ = x.size()
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_out = self.fc(avg_pool.view(batch_size, -1)).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        max_out = self.fc(max_pool.view(batch_size, -1)).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return torch.sigmoid(avg_out + max_out)

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x))

class CBAMModule3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAMModule3D, self).__init__()
        self.channel_attention = ChannelAttention3D(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention3D(kernel_size=spatial_kernel_size)
    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class res_DW_3DCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # 注意：这里将原代码的写死输入1通道改为了参数
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(8, 32)
        self.dwconv = nn.Conv3d(32, 32, kernel_size=3, padding=1, groups=32)
        self.cbam1 = CBAMModule3D(32)
        self.pwconv = nn.Conv3d(32, 3, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.downsample = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
    def forward(self, x):
        residual = x
        # 简单的通道适配，如果输入是1通道，残差连接可能需要1x1卷积，这里假设输入经过conv1后会维度变化
        # 原代码是针对特定输入的，这里为了通用性略作简化，如果报错可能需要调整
        out = self.conv1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.cbam1(out)
        out = self.dwconv(out)
        out = self.norm(out)
        out = self.cbam1(out)
        out = self.pwconv(out)
        out = self.act(out)
        
        # 只有当 residual 和 out 形状一致时相加，或者通过投影匹配
        if residual.shape == out.shape:
            out += residual
            
        out = self.downsample(out)
        out = self.act(out)
        return out

# --- Swin Transformer Parts ---
def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))
        
        # Coords generation skipped for brevity, standard swin logic
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w)) 
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path) # Simplified drop path
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        B, D, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        # Check padding need
        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, Dp, Hp, Wp, _ = x.shape
        
        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
            
        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows) 
        shifted_x = window_reverse(attn_windows, self.window_size, B, Dp, Hp, Wp)
        
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
            
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :D, :H, :W, :].contiguous()
            
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    def forward(self, x):
        B, D, H, W, C = x.shape
        # Pad if needed
        if (H % 2 == 1) or (W % 2 == 1):
             x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class SwinTransformer3D_CNN(nn.Module):
    def __init__(self, num_classes=5, in_chans=1, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        # 3DCNN Part
        self.res_DW_3DCNN = res_DW_3DCNN(in_channels=in_chans)
        
        # Patch Embed (Simulated)
        self.patch_embed_conv = nn.Conv3d(3, embed_dim, kernel_size=(2,4,4), stride=(2,4,4)) # input from CNN is 3 channels
        self.pos_drop = nn.Dropout(p=0.0)
        self.layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock3D(
                    dim=int(embed_dim * 2 ** i_layer),
                    num_heads=num_heads[i_layer],
                    window_size=(2,7,7),
                    shift_size=(0,0,0) if (i % 2 == 0) else (1,3,3)
                ) for i in range(depths[i_layer])
            ])
            downsample = PatchMerging(dim=int(embed_dim * 2 ** i_layer)) if (i_layer < self.num_layers - 1) else None
            self.layers.append(nn.ModuleDict({'blocks': layer, 'downsample': downsample}))
            
        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.head = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.res_DW_3DCNN(x)
        x = self.patch_embed_conv(x)
        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.pos_drop(x)
        
        for layer_dict in self.layers:
            blocks = layer_dict['blocks']
            downsample = layer_dict['downsample']
            for blk in blocks:
                x = blk(x)
            if downsample:
                x = downsample(x)
                
        x = self.norm(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x