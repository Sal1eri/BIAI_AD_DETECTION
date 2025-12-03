import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.dataset import *
from utils.metric import get_classification_metrics
from model.vgg16 import VGG16ForAD
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ADNIDataset(config['model']['name'] )
    train_dataset = dataset.load_data('train')
    val_dataset = dataset.load_data('val')
    batch_size = int(config['training']['batch_size'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = VGG16ForAD()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=float(config['training']['optimizer']['lr']),   
                                 weight_decay=float(config['training']['optimizer']['weight_decay']))
    
    patience = int(config['training']['early_stopping']['patience'])
    best_val_loss = float('inf')
    trigger_times = 0
    epochs = int(config['training']['epochs'])
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_dataset)
        metrics = get_classification_metrics(all_labels, all_preds, all_probs)
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {total_loss/len(train_dataset):.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Metrics: {metrics}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            if not os.path.exists(os.path.dirname(config['training']['save_path'])):
                os.makedirs(os.path.dirname(config['training']['save_path']))
            model.save_model(config['training']['save_path'])
            print("Model saved.")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    model_name = "vgg16"
    cfg_path = os.path.join("configs", "training", f"{model_name}.yaml")
    config = load_config(cfg_path)
    print(config)
    
    train(config)