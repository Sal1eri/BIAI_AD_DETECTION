# AD Detection

follow the setup below to initial env

```
1. >> create a env based on Python version = 3.12

2. >> pip install -r requirements.txt

-- if it dont work, try step4

3. >> bash utils/download_kaggle_dataset.sh

4. >> pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```



| Method | Params | Macro-F1 | Macro-AUC | Balanced Acc. | Cohen's $\kappa$ | MCC (Multi-class) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VGG16 (2D-TL)** | 134.3 M | | | | | |
| **3D-ResNet** | | | | | | |
| **3D-ViT (SOTA)** | | | | | | |
| **P3D-AttnNet (Ours)** | | | | | | |