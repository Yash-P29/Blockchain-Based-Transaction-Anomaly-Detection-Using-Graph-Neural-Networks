import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import pandas as pd
from src.utils.config import FIGURES_DIR, METRICS_DIR


def _get_plt():
    """Lazy-import matplotlib to avoid numpy 2.x compat issues at module load."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _get_sns():
    import seaborn as sns
    return sns


def evaluate_model(y_true, y_prob, y_pred, model_name="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    metrics = {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc
    }

    try:
        plt = _get_plt()
        sns = _get_sns()

        # Save confusion matrix plot
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(FIGURES_DIR / f'{model_name.lower().replace(" ", "_")}_cm.png')
        plt.close()

        # Save ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(FIGURES_DIR / f'{model_name.lower().replace(" ", "_")}_roc.png')
        plt.close()

        # Save PR curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        plt.figure()
        plt.plot(recall_vals, precision_vals, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(FIGURES_DIR / f'{model_name.lower().replace(" ", "_")}_pr.png')
        plt.close()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not generate plots for {model_name}: {e}")

    return metrics


def save_metrics(metrics_list, filename="baseline_results.csv"):
    df = pd.DataFrame(metrics_list)
    df.to_csv(METRICS_DIR / filename, index=False)
    return df
