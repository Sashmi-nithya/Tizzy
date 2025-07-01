# TIZZY Dataset Metadata

This folder contains references and metadata for training data.

## 🔁 Data Flow

1. **GAN generates** future images:
   - Output saved in `/content/drive/MyDrive/TamilNaduClimate/generated/`
   - Example: `gen_day_001.png`, `gen_day_002.png`, ...

2. **CNN uses** these images as input for classification:
   - Paired with corresponding `*_mask.png` if available

3. **Mask Labels** are stored in:
   - `/content/drive/MyDrive/TamilNaduClimate/masks/`

## 📁 Metadata Files

- `generated_files.csv`: Index of all GAN-generated samples
- `sample_paths.txt`: Debugging aid for image-mask inspection
