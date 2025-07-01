import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

from gan.generator import Generator
from gan.discriminator import Discriminator
from gan.gan_utils import save_generated_images

# Custom Dataset
class SatelliteDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB").resize((64, 64))
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.tensor(image).permute(2, 0, 1)
        return image

# Training
def train_gan():
    z_dim = 256
    img_channels = 3
    batch_size = 32
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SatelliteDataset("/content/drive/MyDrive/TamilNaduClimate/images")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator(z_dim, img_channels).to(device)
    D = Discriminator(img_channels).to(device)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for batch in loader:
            batch = batch.to(device)
            real_labels = torch.ones(batch.size(0), 1).to(device)
            fake_labels = torch.zeros(batch.size(0), 1).to(device)

            # Train Discriminator
            outputs = D(batch)
            d_loss_real = criterion(outputs, real_labels)

            z = torch.randn(batch.size(0), z_dim).to(device)
            fake_images = G(z)
            outputs = D(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            z = torch.randn(batch.size(0), z_dim).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            g_loss = criterion(outputs, real_labels)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        if (epoch + 1) % 10 == 0:
            save_generated_images(G, z_dim, device, epoch + 1)

    torch.save(G.state_dict(), "gan_outputs/generator.pth")
    torch.save(D.state_dict(), "gan_outputs/discriminator.pth")

if __name__ == "__main__":
    train_gan()
