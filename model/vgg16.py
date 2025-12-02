import torch
import torch.nn as nn
from torchvision import models

class VGG16ForAD(nn.Module):
    def __init__(self, num_classes=5):
        """
        用于 AD 诊断的 VGG16 模型 (全参数微调模式)
        
        Args:
            num_classes (int): 分类数量 (默认 5: CN, SMC, EMCI, LMCI, AD)
        """
        super(VGG16ForAD, self).__init__()
        
        # 1. 加载预训练的 VGG16 模型 (ImageNet 权重)
        # weights='DEFAULT' 会自动下载并加载最先进的预训练权重
        print("🔄 正在加载 VGG16 预训练权重 (所有层均可训练)...")
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # -----------------------------------------------------------
        # 注意：这里不再包含冻结代码。
        # PyTorch 默认所有层 param.requires_grad = True
        # -----------------------------------------------------------

        # 2. 修改分类头 (Classifier)
        # VGG16 的 classifier[6] 是最后一个全连接层 (4096 -> 1000)
        # 我们将其替换为 (4096 -> num_classes)
        
        in_features = self.vgg16.classifier[6].in_features
        
        # 替换最后一层，新层的权重默认是随机初始化的
        self.vgg16.classifier[6] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, 3, 224, 224]
        return self.vgg16(x)

# ==========================================
# 快速检查
# ==========================================
if __name__ == "__main__":
    # 实例化
    model = VGG16ForAD(num_classes=5)
    
    # 验证是否真的没冻结
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print("-" * 30)
    print(f"📊 参数统计:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    if trainable_params == total_params:
        print("✅ 确认: 所有参数均已解冻，准备进行全量微调。")
    else:
        print("❌ 警告: 部分参数被冻结了。")