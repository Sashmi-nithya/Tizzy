from cnn.model_unet import UNet
from dataset import TamilNaduClimateDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

image_dir = "/content/drive/MyDrive/TamilNaduClimate/images"
mask_dir = "/content/drive/MyDrive/TamilNaduClimate/masks"

dataset = TamilNaduClimateDataset(image_dir, mask_dir)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):  # Modify as needed
    model.train()
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        output = model(img)
        loss = criterion(output, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "/content/unet_final.pth")
