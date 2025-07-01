import numpy as np
import cv2

label_colors = {
    (0, 0, 255): 1,
    (100, 150, 255): 2,
    (0, 255, 0): 3,
    (180, 220, 60): 4,
    (255, 255, 0): 5,
    (0, 0, 128): 6,
    (160, 160, 160): 7,
    (0, 100, 0): 8,
    (210, 180, 140): 9,
}

reverse_label_colors = {v: k for k, v in label_colors.items()}

# Convert RGB pixel to class label
def match_color(pixel, tolerance=30):
    for color, label in label_colors.items():
        if np.allclose(pixel, color, atol=tolerance):
            return label
    return 0  # Unclassified

# Decode a label mask to RGB
def decode_segmap(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in reverse_label_colors.items():
        rgb[mask == label] = color
    return rgb

# Normalize image
def normalize_image(image, method="zero_one"):
    if method == "zero_one":
        return image.astype(np.float32) / 255.0
    elif method == "minus_one_to_one":
        return image.astype(np.float32) / 127.5 - 1.0
    else:
        raise ValueError("Unknown normalization method")

# Resize image/mask
def resize_pair(image, mask, size=(256, 256)):
    image = cv2.resize(image, size)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return image, mask
