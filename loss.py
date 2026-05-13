import torch
import torch.nn as nn

bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to('cuda'))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bce = nn.BCEWithLogitsLoss(
#     pos_weight=torch.tensor([20.0]).to(device)
# )

def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))

def combined_loss(pred, target):
    return bce(pred, target) + dice_loss(pred, target)