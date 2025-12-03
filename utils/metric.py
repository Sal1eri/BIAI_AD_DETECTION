from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
import numpy as np


def get_classification_metrics(y_true, y_pred, y_prob):
    """
    计算多分类任务的多种评价指标。

    参数:
    y_true: 真实标签列表
    y_pred: 预测标签列表
    y_prob: 预测概率列表 (形状为 [n_samples, n_classes])

    返回:
    metrics: 字典，包含各项指标的值
    """
    metrics = {}
    metrics['Macro-F1'] = f1_score(y_true, y_pred, average='macro')
    metrics['Macro-AUC'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    metrics['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics["Cohen's Kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    return metrics


if __name__ == "__main__":
    
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
    metrics = get_classification_metrics(y_true, y_pred, y_prob)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")