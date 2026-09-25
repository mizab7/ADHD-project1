import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def compute_classification_metrics(y_true, y_pred, y_probs=None):
    """
    Computes standard classification evaluation metrics for ADHD vs Control.
    Returns dictionary of metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        tn = fp = fn = tp = 0
        spec = 0.0
        
    auc = np.nan
    if y_probs is not None:
        y_probs = np.asarray(y_probs)
        if y_probs.ndim == 2:
            y_probs = y_probs[:, 1]  # P(ADHD)
        try:
            auc = roc_auc_score(y_true, y_probs)
        except Exception:
            auc = np.nan
            
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(spec),
        "f1_score": float(f1),
        "roc_auc": float(auc) if pd.notnull(auc) else 0.5,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }

def print_metrics_summary(site_name, metrics):
    """
    Prints clean summary table for a site or overall aggregate evaluation fold.
    """
    print(f"\n==========================================")
    print(f" Performance Metrics: Held-Out Site [{site_name}]")
    print(f"==========================================")
    print(f"  Accuracy:            {metrics['accuracy']*100:.2f}%")
    print(f"  Precision:           {metrics['precision']*100:.2f}%")
    print(f"  Recall (Sensitivity):{metrics['recall_sensitivity']*100:.2f}%")
    print(f"  Specificity:         {metrics['specificity']*100:.2f}%")
    print(f"  F1-Score:            {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:             {metrics['roc_auc']:.4f}")
    cm = metrics['confusion_matrix']
    print(f"  Confusion Matrix:    [TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}]")
    print(f"==========================================\n")
