import torch
import matplotlib.pyplot as plt
import numpy as np
from cnn.dataset import TamilNaduClimateDataset
from cnn.model_unet import UNet
import os

label_colors = {
    0: (255, 255, 255),  # Unclassified - White
    1: (0, 0, 255),      # Cold - Blue
    2: (100, 150, 255),  # Cool - Light Blue
    3: (0, 255, 0),      # Mild - Green
    4: (180, 220, 60),   # Warm - Yellow-Green
    5: (255, 255, 0),    # Hot - Yellow
    6: (0, 0, 128),      # Water - Navy Blue
    7: (160, 160, 160),  # Urban - Gray
    8: (0, 100, 0),      # Forest - Dark Green
    9: (210, 180, 140),  # Dry/Affected - Tan
}

def decode_segmap(mask):
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in label_colors.items():
        rgb_mask[mask == class_id] = color
    return rgb_mask

def visualize_prediction(index=0):
    image_dir = "/content/drive/MyDrive/TamilNaduClimate/images"
    mask_dir = "/content/drive/MyDrive/TamilNaduClimate/masks"
    model_path = "checkpoints/unet_classifier.pth"

    dataset = TamilNaduClimateDataset(image_dir, mask_dir)
    image, true_mask = dataset[index]
    input_tensor = image.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_classes=10)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Convert tensors to displayable formats
    image_np = image.permute(1, 2, 0).numpy()
    true_mask_np = decode_segmap(true_mask.numpy())
    pred_mask_np = decode_segmap(pred_mask)

    # Plot side-by-side
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask_np)
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask_np)
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Example usage:
# visualize_prediction(index=5)
