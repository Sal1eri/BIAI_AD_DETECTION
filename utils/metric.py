from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, accuracy_score

def get_classification_metrics(y_true, y_pred, y_prob):
    """
    y_true: 真实标签, shape=(N,)
    y_pred: 预测标签, shape=(N,)
    y_prob: 预测概率, shape=(N, C)
    """
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)  # 新增 Accuracy
    metrics['Macro-F1'] = f1_score(y_true, y_pred, average='macro')
    metrics['Macro-AUC'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    metrics['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics["Cohen's Kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    return metrics
