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
        

        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # 修改分类头 (Classifier)
        # VGG16 的 classifier[6] 是最后一个全连接层 (4096 -> 1000)
        # 将其替换为 (4096 -> num_classes)
        
        in_features = self.vgg16.classifier[6].in_features
        
        # 替换最后一层，新层的权重默认是随机初始化的
        self.vgg16.classifier[6] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.vgg16(x)
if __name__ == "__main__":
    model = VGG16ForAD(num_classes=5)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print("-" * 30)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    