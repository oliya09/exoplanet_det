import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import os
import numpy as np
from .classifier import HybridClassifier  # Для holdout

# Holdout stub: сохрани из train_ml (или генери)
holdout_df = pd.DataFrame({
    'tic_id': ['TIC 150428135', 'TIC 219006972'],  # Добавь 100+
    'true_label': [1, 0]
})

def compute_metrics(holdout_csv='holdout.csv'):
    df = pd.read_csv(holdout_csv) if os.path.exists(holdout_csv) else holdout_df
    scores, labels = [], []
    classifier = HybridClassifier()
    for tid in df['tic_id'][:50]:  # Limit
        # Stub res (реал: classify)
        res = {"Hybrid_score": np.random.rand(), "lc": None}  # Замени на classify_target_full(tid)
        scores.append(res['Hybrid_score'])
        labels.append(df[df['tic_id']==tid]['true_label'].iloc[0])
    
    auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    cm = confusion_matrix(labels, [1 if s >= 0.9 else 0 for s in scores])
    
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    fpr, tpr, _ = roc_curve(labels, scores)
    axs[0].plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})'); axs[0].legend()
    axs[1].plot(recall, precision, label='PR Curve'); axs[1].legend()
    axs[2].imshow(cm, cmap='Blues'); axs[2].set_title('Confusion Matrix')
    plt.savefig('metrics.png', dpi=300)
    plt.close()
    
    metrics = {'AUC': auc, 'F1': 2 * np.mean(precision * recall), 'CM': cm.tolist()}
    pd.DataFrame([metrics]).to_json('metrics.json', orient='records')
    print(f"Metrics: AUC={auc:.3f}, F1={2 * np.mean(precision * recall):.3f}")

if __name__ == "__main__":
    compute_metrics()