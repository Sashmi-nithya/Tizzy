import torch
import os

# Load GAN Generator
def load_generator(checkpoint_path="checkpoints/gan_generator.pth", z_dim=256, device="cpu"):
    from gan.generator import Generator
    model = Generator(z_dim=z_dim).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded GAN Generator from {checkpoint_path}")
    else:
        print("❌ Generator checkpoint not found.")
    return model.eval()

# Load GAN Discriminator
def load_discriminator(checkpoint_path="checkpoints/gan_discriminator.pth", device="cpu"):
    from gan.discriminator import Discriminator
    model = Discriminator().to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded GAN Discriminator from {checkpoint_path}")
    else:
        print("❌ Discriminator checkpoint not found.")
    return model.eval()

# Load U-Net Classifier
def load_unet(checkpoint_path="checkpoints/unet_classifier.pth", in_channels=3, out_classes=10, device="cpu"):
    from cnn.model_unet import UNet
    model = UNet(in_channels=in_channels, out_classes=out_classes).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded U-Net Classifier from {checkpoint_path}")
    else:
        print("❌ U-Net checkpoint not found.")
    return model.eval()
