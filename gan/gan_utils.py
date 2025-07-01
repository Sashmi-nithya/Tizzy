import torch
import torchvision
import os

def save_generated_images(generator, z_dim, device, epoch, output_dir="gan_outputs"):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(16, z_dim).to(device)
        fake_images = generator(z)
        fake_images = (fake_images + 1) / 2  # Convert to [0, 1]
        os.makedirs(output_dir, exist_ok=True)
        torchvision.utils.save_image(fake_images, f"{output_dir}/epoch_{epoch}.png", nrow=4)
    generator.train()
