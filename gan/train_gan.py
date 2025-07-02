import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import os
import numpy as np

from gan.generator import Generator
from gan.discriminator import Discriminator
from gan.gan_utils import save_generated_images

# --- Dataset class ---
class SatelliteDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB").resize((64, 64))
        image = np.array(image).astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
        image = torch.tensor(image).permute(2, 0, 1)
        return image

# --- Generate synthetic future images (2026–2050) ---
def generate_future_images(generator, z_dim=256, output_dir="/content/drive/MyDrive/tamilnadu_gan_images/", num_images=9125, device="cuda" if torch.cuda.is_available() else "cpu"):
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        for i in range(num_images):
            z = torch.randn(1, z_dim).to(device)
            fake_img = generator(z)
            fake_img = (fake_img + 1) / 2  # Convert from [-1,1] to [0,1]
            save_path = os.path.join(output_dir, f"gen_day_{i+1:04d}.png")
            save_image(fake_img, save_path)
            if (i + 1) % 500 == 0:
                print(f"✅ Generated {i + 1} / {num_images} images")
    print(f"\n🎉 Done! All {num_images} future images saved to: {output_dir}")

# --- Train GAN ---
def train_gan():
    z_dim = 256
    img_channels = 3
    batch_size = 32
    num_epochs = 2000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SatelliteDataset("/content/drive/MyDrive/tamilnadu_png_images")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Generator(z_dim, img_channels).to(device)
    discriminator = Discriminator(img_channels).to(device)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for batch in loader:
            batch = batch.to(device)
            real_labels = torch.ones(batch.size(0), 1).to(device)
            fake_labels = torch.zeros(batch.size(0), 1).to(device)

            # Train Discriminator
            outputs = discriminator(batch)
            d_loss_real = criterion(outputs, real_labels)

            z = torch.randn(batch.size(0), z_dim).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            z = torch.randn(batch.size(0), z_dim).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        if (epoch + 1) % 10 == 0:
            save_generated_images(generator, z_dim, device, epoch + 1)

    # --- Save model checkpoints ---
    os.makedirs("gan_outputs", exist_ok=True)
    torch.save(generator.state_dict(), "gan_outputs/generator.pth")
    torch.save(discriminator.state_dict(), "gan_outputs/discriminator.pth")

    # --- Automatically generate future climate images ---
    generate_future_images(
        generator=generator,
        z_dim=z_dim,
        output_dir="/content/drive/MyDrive/tamilnadu_gan_images",
        num_images=9125,
        device=device
    )

if __name__ == "__main__":
    train_gan()
