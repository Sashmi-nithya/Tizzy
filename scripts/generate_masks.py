import cv2
import numpy as np
import os
from scripts.utils import match_color

image_dir = "/content/drive/MyDrive/TamilNaduClimate/images"
mask_dir = "/content/drive/MyDrive/TamilNaduClimate/masks"
os.makedirs(mask_dir, exist_ok=True)

for fname in os.listdir(image_dir):
    if fname.endswith(".png"):
        img = cv2.imread(os.path.join(image_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                mask[y, x] = match_color(img[y, x])

        out_path = os.path.join(mask_dir, fname.replace(".png", "_mask.png"))
        cv2.imwrite(out_path, mask)
