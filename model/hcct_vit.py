import torch
import torch.nn as nn
from einops import rearrange
import math

# 来自: arindammajee/alzheimer-detection-with-3d-hcct/ViT/HCCT.py
# 已做适当调整以适配独立调用

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        
    def forward(self, x):
        return self.maxpool(self.act((self.bn(self.conv(x)))))

class NewGELUActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_channels = config.get("num_channels", 1)
        # 这里的 patch embedding 使用了 Conv 堆叠的方式，而不是直接切片
        self.conv_1 = ConvBlock(self.num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_4 = ConvBlock(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_5 = ConvBlock(256, 512, kernel_size=3, stride=1, padding=1)
        self.num_patches = 512 # 根据卷积层输出写死

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = rearrange(x, 'b c d w h -> b c (d w h)')
        return x

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # +1 for CLS token
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 简单的位置编码加法，注意这里假设了 num_patches 是固定的
        if x.size(1) != self.position_embeddings.size(1):
             # 简单的容错处理，如果尺寸不匹配则截断或插值，这里直接切片演示
             x = x + self.position_embeddings[:, :x.size(1), :]
        else:
            x = x + self.position_embeddings
        x = self.dropout(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        return attention_output

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]
        self.heads = nn.ModuleList([
            AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            ) for _ in range(self.num_attention_heads)
        ])
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat(attention_outputs, dim=-1)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        return attention_output

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x):
        attention_output = self.attention(self.layernorm_1(x))
        x = x + attention_output
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["num_hidden_layers"])])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class HCCT_ViT(nn.Module):
    def __init__(self, num_classes=5, input_channels=1):
        super().__init__()
        # 默认配置，根据 arindammajee 仓库代码推断
        self.config = {
            "num_channels": input_channels,
            "image_size": 128, # 注意：这里的ViT实现依赖卷积层，对输入尺寸不完全敏感，但位置编码有固定长度
            "patch_size": 16, # 未在PatchEmbeddings中直接使用，但保留配置
            "hidden_size": 512, # 由Conv_5输出通道决定
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "intermediate_size": 2048,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "initializer_range": 0.02,
            "qkv_bias": True,
            "num_classes": num_classes
        }
        
        self.embedding = Embeddings(self.config)
        self.encoder = Encoder(self.config)
        self.attention_pool = nn.Linear(self.config["hidden_size"], 1)
        self.classifier = nn.Linear(2*self.config["hidden_size"], self.config["num_classes"])
        self.apply(self._init_weights)

    def forward(self, x):
        # x: (Batch, Channel, Depth, Height, Width)
        embedding_output = self.embedding(x)
        encoder_output = self.encoder(embedding_output)
        
        # Take CLS token and Attention Pooling
        cls_logits = encoder_output[:, 0, :]
        activation_logits = encoder_output[:, 1:, :]
        
        # Attention Pooling logic
        activation_logits = torch.matmul(
            nn.functional.softmax(self.attention_pool(activation_logits), dim=1).transpose(-1, -2), 
            activation_logits
        ).squeeze(-2)
        
        logits = torch.cat((cls_logits, activation_logits), dim=1)
        logits = self.classifier(logits)
        return logits

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)