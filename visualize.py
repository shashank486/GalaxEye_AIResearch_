import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

from dataset import ChangeDetectionDataset
from model import get_model
import argparse


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

model.load_state_dict(
    # torch.load("model_latest.pth", map_location=device)
    torch.load(args.weights, map_location=device)
)

model.eval()

# -----------------------------
# DATASET
# -----------------------------
# dataset = ChangeDetectionDataset("dataset/val")
dataset = ChangeDetectionDataset(args.data_path)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

THRESHOLD = 0.5

# -----------------------------
# VISUALIZE
# -----------------------------
with torch.no_grad():

    for idx, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        pred = torch.sigmoid(pred)

        pred = (pred > THRESHOLD).float()

        # -------------------------
        # Compute IoU
        # -------------------------

        pred_flat = pred.view(-1)
        y_flat = y.view(-1)

        intersection = (pred_flat * y_flat).sum()

        union = pred_flat.sum() + y_flat.sum() - intersection

        iou = intersection / (union + 1e-6)

        iou_value = iou.item()

        # -------------------------
        # Auto classify
        # -------------------------

        if iou_value > 0.02:
            result = "SUCCESS"

        elif iou_value > 0.005:
            result = "MIXED"

        else:
            result = "FAILURE"

        print(f"Sample {idx}")
        print(f"IoU: {iou_value:.4f}")
        print(f"Result: {result}")

        # -------------------------
        # Convert tensors
        # -------------------------

        x = x.cpu().squeeze(0)

        y = y.cpu().squeeze().numpy()

        pred = pred.cpu().squeeze().numpy()

        pre = x[:3].permute(1, 2, 0).numpy()

        post = x[3:6].permute(1, 2, 0).numpy()

        # normalize
        pre = (pre - pre.min()) / (
            pre.max() - pre.min() + 1e-6
        )

        post = (post - post.min()) / (
            post.max() - post.min() + 1e-6
        )

        # -------------------------
        # Plot
        # -------------------------

        fig, ax = plt.subplots(1, 4, figsize=(16, 5))

        ax[0].imshow(pre)
        ax[0].set_title("Pre-event")

        ax[1].imshow(post)
        ax[1].set_title("Post-event")

        ax[2].imshow(y, cmap="gray")
        ax[2].set_title("Ground Truth")

        ax[3].imshow(pred, cmap="gray")
        ax[3].set_title(f"Prediction\n{result}")

        for a in ax:
            a.axis("off")

        plt.tight_layout()

        plt.savefig(f"{result}_{idx}.png")

        plt.show()

        # save only first 5
        if idx == 4:
            break