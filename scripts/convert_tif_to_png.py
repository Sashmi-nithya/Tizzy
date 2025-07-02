import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio

def convert_tif_to_png(tif_folder, png_folder, cmap='terrain'):
    os.makedirs(png_folder, exist_ok=True)
    tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]

    for idx, tif_file in enumerate(tif_files):
        tif_path = os.path.join(tif_folder, tif_file)
        with rasterio.open(tif_path) as src:
            data = src.read(1)  # read the first band
            data = np.where(data == src.nodata, np.nan, data)  # handle nodata

        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(data, cmap=cmap)
        png_name = os.path.splitext(tif_file)[0] + ".png"
        png_path = os.path.join(png_folder, png_name)
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        if (idx + 1) % 50 == 0 or idx == len(tif_files) - 1:
            print(f"✅ Converted {idx + 1}/{len(tif_files)} TIFs to PNG")

# Example usage
if __name__ == "__main__":
    convert_tif_to_png(
        tif_folder="/content/drive/MyDrive/tamilnadu_tif_files",
        png_folder="/content/drive/MyDrive/tamilnadu_png_images",
        cmap="terrain"  # You can try "viridis", "jet", or "plasma" too
    )
