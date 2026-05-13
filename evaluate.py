

# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from sklearn.metrics import confusion_matrix

# from dataset import ChangeDetectionDataset
# from model import get_model

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import ChangeDetectionDataset
from model import get_model
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_path",
    type=str,
    required=True,
    help="Path to dataset"
)

parser.add_argument(
    "--weights",
    type=str,
    required=True,
    help="Path to model weights"
)

args = parser.parse_args()



# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = get_model().to(device)

# model.load_state_dict(
#     torch.load("model_latest.pth", map_location=device)
# )
model.load_state_dict(
    torch.load(args.weights, map_location=device)
)

model.eval()

# -----------------------------
# LOAD DATASET
# -----------------------------
# dataset = ChangeDetectionDataset("dataset/val")
dataset = ChangeDetectionDataset(args.data_path)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

# -----------------------------
# METRICS STORAGE
# -----------------------------
ious = []
precisions = []
recalls = []
f1_scores = []

all_preds = []
all_targets = []

# -----------------------------
# THRESHOLD
# -----------------------------
THRESHOLD = 0.5

# -----------------------------
# EVALUATION LOOP
# -----------------------------
with torch.no_grad():

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        pred = torch.sigmoid(pred)

        pred = (pred > THRESHOLD).float()

        # flatten
        pred_flat = pred.view(-1)
        y_flat = y.view(-1)

        # save for confusion matrix
        all_preds.extend(pred_flat.cpu().numpy())
        all_targets.extend(y_flat.cpu().numpy())

        # -------------------------
        # TP, FP, FN
        # -------------------------
        TP = (pred_flat * y_flat).sum()

        FP = ((pred_flat == 1) & (y_flat == 0)).sum()

        FN = ((pred_flat == 0) & (y_flat == 1)).sum()

        # -------------------------
        # IoU
        # -------------------------
        iou = TP / (TP + FP + FN + 1e-6)

        # -------------------------
        # Precision
        # -------------------------
        precision = TP / (TP + FP + 1e-6)

        # -------------------------
        # Recall
        # -------------------------
        recall = TP / (TP + FN + 1e-6)

        # -------------------------
        # F1 Score
        # -------------------------
        f1 = 2 * precision * recall / (
            precision + recall + 1e-6
        )

        ious.append(iou.item())
        precisions.append(precision.item())
        recalls.append(recall.item())
        f1_scores.append(f1.item())

# -----------------------------
# FINAL METRICS
# -----------------------------
print("\n--- FINAL RESULTS ---")

print(f"Mean IoU      : {np.mean(ious):.4f}")
print(f"Mean Precision: {np.mean(precisions):.4f}")
print(f"Mean Recall   : {np.mean(recalls):.4f}")
print(f"Mean F1 Score : {np.mean(f1_scores):.4f}")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(
    all_targets,
    all_preds
)

print("\n--- CONFUSION MATRIX ---")
print(cm)
