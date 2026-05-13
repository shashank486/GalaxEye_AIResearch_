import torch
from torch.utils.data import DataLoader
from dataset import ChangeDetectionDataset
from model import get_model
from loss import combined_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = ChangeDetectionDataset("/content/change_dataset/train") 
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = get_model().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 45

for epoch in range(EPOCHS):
    print(f"Starting Epoch {epoch+1}")   # 👈 ADD THIS

    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        # print(f"Batch {i}") 
        pred = model(x)
        loss = combined_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")