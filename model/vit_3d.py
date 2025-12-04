import torch
import torch.nn as nn
import numpy as np

# 来自: qasymjomart/vit_recipe_for_ad/models/vit3d.py
# 工具函数内联，以便独立运行

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # 简化的截断正态分布初始化
    with torch.no_grad():
        return tensor.normal_(mean, std).clamp(a*std, b*std)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() 
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed3D(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=1, embed_dim=768):
        super().__init__()
        # 假设 img_size 是元组 (D, H, W) 或 单个 int
        if isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
            
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        # 计算 patch 数量
        self.n_patches = (img_size[0] // patch_size[0]) * \
                         (img_size[1] // patch_size[1]) * \
                         (img_size[2] // patch_size[2])

    def forward(self, x):
        x = self.proj(x) 
        x = x.flatten(2) 
        x = x.transpose(1, 2) 
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        n_samples, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(n_samples, n_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, drop_path=0., p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim, p=p)
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer3D(nn.Module):
    def __init__(self, 
                img_size=128,  # 请根据实际输入调整
                patch_size=16, 
                in_chans=1, 
                num_classes=5, 
                embed_dim=768, 
                depth=12, 
                n_heads=12, 
                mlp_ratio=4., 
                qkv_bias=True, 
                drop_path_rate=0.,
                p=0., 
                attn_p=0.):
        super().__init__()
        
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)
            
        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.rand(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_path=dpr[ii], p=p, attn_p=attn_p
            ) for ii in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化权重
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        # x shape: (B, C, D, H, W)
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # 简单处理位置编码对齐，如果输入尺寸变化
        if x.shape[1] != self.pos_embed.shape[1]:
             x = x + self.pos_embed[:, :x.shape[1], :]
        else:
             x = x + self.pos_embed
             
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x