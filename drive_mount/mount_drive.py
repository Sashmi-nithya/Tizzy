from google.colab import drive
import os

def mount_and_set_paths():
    drive.mount('/content/drive')
    image_dir = "/content/drive/MyDrive/TamilNaduClimate/images"
    mask_dir = "/content/drive/MyDrive/TamilNaduClimate/masks"
    os.makedirs(mask_dir, exist_ok=True)
    return image_dir, mask_dir
