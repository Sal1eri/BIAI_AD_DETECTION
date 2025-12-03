from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
import numpy as np

# --- 模拟数据 (假设是 3 分类问题: 0, 1, 2) ---
# 真实标签
y_true = [0, 1, 2, 0, 1, 2, 0, 2]
# 模型预测的类别 (Hard labels)
y_pred = [0, 2, 2, 0, 1, 1, 0, 2]
# 模型预测的概率 (Softmax output, 形状为 [n_samples, n_classes])
# 每一行加起来应该是 1
y_prob = [
    [0.8, 0.1, 0.1], # pred 0
    [0.2, 0.3, 0.5], # pred 2 (Error)
    [0.1, 0.2, 0.7], # pred 2
    [0.9, 0.05, 0.05], # pred 0
    [0.1, 0.8, 0.1], # pred 1
    [0.3, 0.6, 0.1], # pred 1 (Error)
    [0.7, 0.2, 0.1], # pred 0
    [0.1, 0.1, 0.8]  # pred 2
]

# --- 1. Macro-F1 ---
# 'macro': 计算每个类别的 F1，然后取未加权的平均值
macro_f1 = f1_score(y_true, y_pred, average='macro')

# --- 2. Macro-AUC (需要概率) ---
# multi_class='ovr': 一对多 (One-vs-Rest) 策略，适合 Macro 计算
macro_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

# --- 3. Balanced Accuracy ---
# 自动处理不平衡样本，计算每个类别的 Recall 平均值
balanced_acc = balanced_accuracy_score(y_true, y_pred)

# --- 4. Cohen's Kappa ---
# 衡量一致性，排除偶然猜对的概率

kappa = cohen_kappa_score(y_true, y_pred)

# --- 5. MCC (Multi-class) ---
# 马修斯相关系数，被认为是处理不平衡数据最稳健的指标之一
mcc = matthews_corrcoef(y_true, y_pred)

# --- 输出结果 ---
print(f"Macro-F1:       {macro_f1:.4f}")
print(f"Macro-AUC:      {macro_auc:.4f}")
print(f"Balanced Acc:   {balanced_acc:.4f}")
print(f"Cohen's Kappa:  {kappa:.4f}")
print(f"MCC:            {mcc:.4f}")
