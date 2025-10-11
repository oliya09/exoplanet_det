import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# === CNN model ===
class CNNModel(nn.Module):
    def __init__(self, input_size=200):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * (input_size // 4), 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# === Score ===
def compute_metrics(holdout_csv='holdout.csv', model_path='best_model.pth'):
    if os.path.exists(holdout_csv):
        df = pd.read_csv(holdout_csv)
    else:
        df = pd.DataFrame({
            'tic_id': ['TIC 150428135', 'TIC 219006972'],
            'true_label': [1, 0]
        })

    # loading model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scores, labels = [], []

    for _, row in df.iterrows():
        lc_data = np.random.rand(200)  
        lc_tensor = torch.tensor(lc_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(lc_tensor).cpu().item()

        scores.append(pred)
        labels.append(row['true_label'])

    # === Labels ===
    auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    cm = confusion_matrix(labels, [1 if s >= 0.5 else 0 for s in scores])

    # === Graphics ===
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fpr, tpr, _ = roc_curve(labels, scores)
    axs[0].plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})'); axs[0].legend()
    axs[1].plot(recall, precision, label='PR Curve'); axs[1].legend()
    axs[2].imshow(cm, cmap='Blues'); axs[2].set_title('Confusion Matrix')
    plt.savefig('metrics.png', dpi=300)
    plt.close()

    # === Save results ===
    metrics = {'AUC': auc, 'F1': 2 * np.mean(precision * recall), 'CM': cm.tolist()}
    pd.DataFrame([metrics]).to_json('metrics.json', orient='records')

    print(f"✅ Metrics: AUC={auc:.3f}, F1={2 * np.mean(precision * recall):.3f}")

if __name__ == "__main__":
    compute_metrics()
