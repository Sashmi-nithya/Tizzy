
# 🌦️ Tizzy: Climate Forecasting for Tamil Nadu using GAN & CNN

**Tizzy** is a deep learning project designed to **forecast regional climate changes** in Tamil Nadu (India) using satellite imagery and AI models.  
It uses a **Generative Adversarial Network (GAN)** to generate future synthetic climate images (2026–2050) and a **U-Net CNN** to classify those images into regions like **Cold**, **Warm**, and **Severe** based on past climate data (2000–2024).

---

## 📌 Features

- 🔄 Converts MODIS `.tif` images to `.png` using colormaps
- 🤖 Trains a GAN to generate future climate maps
- 🧠 Classifies regions with CNN (Cold, Warm, Severe)
- 🌐 Supports visual temperature overlays and region labeling
- 🗺️ Tamil Nadu boundary contour extraction

---


## ⚙️ Setup Instructions

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/Sashmi-nithya/Tizzy.git
cd Tizzy
```

### ✅ Step 2: Install Dependencies

Make sure you are using Python 3.8+.

```bash
pip install -r requirements.txt
```

Or, if you want to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🛰️ Dataset Preparation

Download `.tif` satellite images of Tamil Nadu (2000–2025) using MODIS from Google Earth Engine.  
Place them into `data/tif/`.

### ➤ Convert `.tif` to `.png` (for GAN training):
```bash
python scripts/convert_tif_to_png.py
```

The converted images will be stored in `data/png/` or `tamilnadu_png_images/`.

---

## 🧠 Train the GAN (to forecast 2026–2050)

```bash
python gan/train_gan.py
```

This trains the GAN on 2000–2025 `.png` images and saves synthetic 2026–2050 climate images into `tamilnadu_gan_images/`.

---

## 🎯 Train CNN to Classify (Cold, Warm, Severe)

```bash
python cnn/train_cnn.py
```

> Ensure you have corresponding masks in the `/masks` folder.

---

## 🖼️ Visualize Output from CNN

```bash
python sample_cnn_result.py
```

This script shows:
- Original input image
- Ground truth mask
- Predicted temperature classification

---

## ✅ Pretrained Models

Model weights (`.pth`) should be placed in:

```
checkpoints/
├── gan/generator.pth
├── gan_discriminator/discriminator.pth
└── cnn/unet_classifier.pth
```

These files are **not included in the repo**. Please download them from the Google Drive link and place them manually.

---

## 👩‍💻 Author

**Nithya Shree R.**  
GitHub: [@Sashmi-nithya](https://github.com/Sashmi-nithya)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- MODIS Satellite Data  
- Google Earth Engine  
- PyTorch, Matplotlib, Rasterio, OpenCV
