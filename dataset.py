import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.pre_dir = os.path.join(root_dir, "pre-event")
        self.post_dir = os.path.join(root_dir, "post-event")
        self.mask_dir = os.path.join(root_dir, "target")

        self.files = sorted(os.listdir(self.pre_dir))
        self.transform = transform

    def remap_mask(self, mask):
        new_mask = np.zeros_like(mask)
        new_mask[(mask == 2) | (mask == 3)] = 1
        return new_mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        pre = cv2.imread(os.path.join(self.pre_dir, fname))
        post = cv2.imread(os.path.join(self.post_dir, fname))
        mask = cv2.imread(os.path.join(self.mask_dir, fname), 0)

        mask = self.remap_mask(mask)

        pre = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
        post = cv2.cvtColor(post, cv2.COLOR_BGR2RGB)

        # Normalize
        pre = pre / 255.0
        post = post / 255.0

        pre = cv2.resize(pre, (256, 256))
        post = cv2.resize(post, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # Stack → 6 channels
        x = np.concatenate([pre, post], axis=2)

        if self.transform:
            augmented = self.transform(image=x, mask=mask)
            x = augmented['image']
            mask = augmented['mask']

        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return x, mask