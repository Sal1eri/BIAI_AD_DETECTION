import os
import torch
from torch.utils.data import DataLoader
from utils.dataset import ADNIDataset, load_config
from utils.metric import get_classification_metrics
from model.vgg16 import VGG16ForAD
from tqdm import tqdm
import torch.nn as nn

def test(config, model_path=None):
    """
    在测试集上评估模型
    Args:
        config: yaml 配置
        model_path: 模型权重路径，如果为空则使用 config 中的 save_path
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    dataset = ADNIDataset(config['model']['name'])
    test_dataset = dataset.load_data('test')
    batch_size = int(config['training']['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 加载模型
    model = VGG16ForAD().to(device)
    if model_path is None:
        model_path = config['training']['save_path']
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_dataset)
    metrics = get_classification_metrics(all_labels, all_preds, all_probs)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics: {metrics}")
    return metrics

# ------------------------
# 使用方法
# ------------------------
if __name__ == "__main__":
    model_name = "vgg16"
    cfg_path = os.path.join("configs", "training", f"{model_name}.yaml")
    config = load_config(cfg_path)
    
    test_metrics = test(config)
