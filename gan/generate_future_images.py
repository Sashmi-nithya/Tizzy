import torch
import os
from torchvision.utils import save_image
from gan.generator import Generator

def generate_future_images(
    generator_path,
    output_dir,
    z_dim=256,
    num_images=9125,  # 25 years × 365 days
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(output_dir, exist_ok=True)

    # Load trained generator
    generator = Generator(z_dim=z_dim).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    # Generate images
    with torch.no_grad():
        for i in range(num_images):
            z = torch.randn(1, z_dim).to(device)
            fake_img = generator(z)
            fake_img = (fake_img + 1) / 2  # scale from [-1,1] to [0,1]
            save_path = os.path.join(output_dir, f"gen_day_{i+1:04d}.png")
            save_image(fake_img, save_path)
            if (i + 1) % 500 == 0:
                print(f"✅ Generated {i + 1} / {num_images} images")

    print(f"\n🎉 Done! All {num_images} future images saved in: {output_dir}")

# Example usage
if __name__ == "__main__":
    generate_future_images(
        generator_path="/content/drive/MyDrive/TIZZY/checkpoints/gan_generator.pth",
        output_dir="/content/drive/MyDrive/TIZZY/generated/"
    )
