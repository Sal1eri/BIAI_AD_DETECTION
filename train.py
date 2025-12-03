import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 指定使用的 GPU 编号
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm  # 用于显示进度条
import time

# ==========================================
# 1. 配置参数
# ==========================================
CSV_PATH = "./dataset/5_class_10_12_2025.csv"
DATA_ROOT = "./dataset/ADNI/ADNI"
IMG_SIZE = 224
BATCH_SIZE = 16          # 如果显存不够，改小这个数字 (例如 8 或 16)
LEARNING_RATE = 1e-4     # 学习率
NUM_EPOCHS = 10          # 训练轮数
NUM_CLASSES = 5          # 5分类

# 标签映射
LABEL_MAP = {
    'CN': 0,
    'EMCI': 1,
    'LMCI': 2,
    'AD': 3,
    'MCI': 4
}

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 使用设备: {device}")

# ==========================================
# 2. 辅助函数 & Dataset 定义
# ==========================================
def find_nii_path(root_dir, subject_id, image_id):
    """在 Subject 文件夹下递归查找具体的 NIfTI 文件"""
    subject_dir = os.path.join(root_dir, subject_id)
    if not os.path.exists(subject_dir):
        return None
    patterns = [
        os.path.join(subject_dir, "**", f"*{image_id}*.nii"),
        os.path.join(subject_dir, "**", f"*{image_id}*.nii.gz")
    ]
    for pat in patterns:
        files = glob.glob(pat, recursive=True)
        if files:
            return files[0]
    return None

def extract_middle_slice(nii_path):
    """读取 NIfTI 并提取冠状面(Coronal)中间切片"""
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # 取 Axis 1 (冠状面) 的中间一张
        slice_idx = data.shape[1] // 2
        slice_2d = data[:, slice_idx, :]
        
        # 旋转校正
        slice_2d = np.rot90(slice_2d)
        
        # !!! 重要：移除了 plt.show() 以免阻断训练流程 !!!
        
        # 归一化 (Min-Max) -> 0-255
        d_min, d_max = slice_2d.min(), slice_2d.max()
        if d_max - d_min > 0:
            slice_2d = (slice_2d - d_min) / (d_max - d_min)
        else:
            slice_2d = np.zeros_like(slice_2d)
            
        slice_2d = (slice_2d * 255).astype(np.uint8)
        return Image.fromarray(slice_2d)
        
    except Exception as e:
        print(f"Error reading {nii_path}: {e}")
        return None

class ADNI2DDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV not found: {csv_file}")
            
        df = pd.read_csv(csv_file)
        # 为了确保“按顺序分”是有意义的，我们默认不做 shuffle，完全依赖 CSV 的顺序
        
        print(f"🚀 正在扫描路径 (CSV共 {len(df)} 条)...")
        for _, row in df.iterrows():
            group = row['Group']
            if group not in LABEL_MAP:
                continue
            path = find_nii_path(root_dir, row['Subject'], row['Image Data ID'])
            if path:
                self.samples.append({
                    'path': path,
                    'label': LABEL_MAP[group]
                })
        print(f"✅ 数据加载完毕! 有效样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = extract_middle_slice(item['path'])
        
        if img is None:
            img = Image.new('L', (IMG_SIZE, IMG_SIZE)) # 异常处理：黑图
            
        img = img.convert('RGB') # 转为 RGB 适配 VGG
        
        if self.transform:
            img = self.transform(img)
            
        return img, item['label']

# ==========================================
# 3. 定义 VGG16 模型
# ==========================================
class VGG16ForAD(nn.Module):
    def __init__(self, num_classes=5):
        super(VGG16ForAD, self).__init__()
        # 加载预训练权重
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # 修改分类层
        in_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.vgg16(x)

# ==========================================
# 4. 训练与验证函数
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用 tqdm 显示进度条
    loop = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    loss = running_loss / len(loader)
    acc = 100 * correct / total
    return loss, acc

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    # --- A. 数据准备 ---
    print("\n[Step 1] 准备数据...")
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = ADNI2DDataset(CSV_PATH, DATA_ROOT, transform=data_transforms)
    
    # --- B. 按顺序切分数据集 (7:1:2) ---
    total_len = len(full_dataset)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.1)
    test_len = total_len - train_len - val_len
    
    # 生成有序索引
    indices = list(range(total_len))
    train_idx = indices[:train_len]
    val_idx = indices[train_len : train_len + val_len]
    test_idx = indices[train_len + val_len :]
    
    print(f"📊 数据划分 (Sequential Split):")
    print(f"   Train: {len(train_idx)} (0 - {train_len-1})")
    print(f"   Val:   {len(val_idx)} ({train_len} - {train_len+val_len-1})")
    print(f"   Test:  {len(test_idx)} ({train_len+val_len} - {total_len-1})")
    
    # 使用 Subset 创建子数据集
    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)
    test_set  = Subset(full_dataset, test_idx)
    
    # 创建 DataLoader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # 训练内部可以打乱
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- C. 模型初始化 ---
    print("\n[Step 2] 初始化模型...")
    model = VGG16ForAD(num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- D. 开始训练 ---
    print(f"\n[Step 3] 开始训练 ({NUM_EPOCHS} Epochs)...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        # 1. 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        
        # 2. 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
              
    total_time = time.time() - start_time
    print(f"\n✨ 训练完成! 总耗时: {total_time:.0f}s")
    
    # --- E. 最终测试 ---
    print("\n[Step 4] 在测试集上进行最终评估...")
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"🏆 Test Set Accuracy: {test_acc:.2f}%")