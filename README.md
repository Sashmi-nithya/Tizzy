 README.md — Full Version
markdown
Copy code
# 🌦️ Tizzy: Climate Forecasting for Tamil Nadu using GAN & CNN

**Tizzy** is a deep learning project designed to **forecast regional climate changes** in Tamil Nadu (India) using satellite imagery and AI models.  
It uses a **Generative Adversarial Network (GAN)** to generate future synthetic climate images (2026–2050) and a **U-Net CNN** to classify those images into regions like **Cold**, **Warm**, and **Severe** based on past climate data (2000–2025).

---

## 📌 Features

- 🔄 Converts MODIS `.tif` images to `.png` using colormaps
- 🤖 Trains a GAN to generate future climate maps
- 🧠 Classifies regions with CNN (Cold, Warm, Severe)
- 🌐 Supports visual temperature overlays and region labeling
- 🗺️ Tamil Nadu boundary contour extraction

---

## 🗂️ Project Directory Structure

Tizzy/
├── gan/
│ ├── generator.py # GAN generator model
│ ├── discriminator.py # GAN discriminator model
│ ├── gan_utils.py # GAN utilities
│ └── train_gan.py # Train GAN on historical data
│
├── cnn/
│ ├── model_unet.py # U-Net CNN model
│ ├── dataset.py # Dataset loader for CNN
│ └── train_cnn.py # Train CNN to classify images
│
├── scripts/
│ └── convert_tif_to_png.py # Converts MODIS .tif to .png
│
├── checkpoints/ # Pretrained .pth model files (add manually)
├── outputs/ # Visual predictions (classified maps)
├── data/ # Raw and processed satellite images
├── sample_cnn_result.py # Visualize classification result
├── requirements.txt # All required Python packages
└── README.md

yaml
Copy code

---

## ⚙️ Setup Instructions

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/Sashmi-nithya/Tizzy.git
cd Tizzy
✅ Step 2: Install Dependencies
Make sure you are using Python 3.8+.

bash
Copy code
pip install -r requirements.txt
Or, if you want to use a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
🛰️ Dataset Preparation
Download .tif satellite images of Tamil Nadu (2000–2025) using MODIS from Google Earth Engine.

Place them into data/tif/.

➤ Convert .tif to .png (for GAN training):
bash
Copy code
python scripts/convert_tif_to_png.py
The converted images will be stored in data/png/ or tamilnadu_png_images/.

🧠 Train the GAN (to forecast 2026–2050)
bash
Copy code
python gan/train_gan.py
This:

Trains the GAN on 2000–2025 .png images

Saves generated 2026–2050 climate images into tamilnadu_gan_images/

🎯 Train CNN to Classify (Cold, Warm, Severe)
bash
Copy code
python cnn/train_cnn.py
Ensure you have corresponding masks (/masks) for training.

🖼️ Visualize Output from CNN
bash
Copy code
python sample_cnn_result.py
This script shows:

Original input image

Ground truth mask

Predicted temperature classification

📸 Output Example
GAN Input (Day)	Predicted Temperature Map

✅ Pretrained Models
Model weights (.pth) should be placed in:

bash
Copy code
checkpoints/
├── gan/generator.pth
├── gan_discriminator/discriminator.pth
└── cnn/unet_classifier.pth
These files are not included in the repo. Please download them from the Google Drive link and place them manually.

👩‍💻 Author
Nithya Shree R.
GitHub: @Sashmi-nithya

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgements
MODIS Satellite Data

Google Earth Engine

PyTorch, Matplotlib, Rasterio, and OpenCV

vbnet
Copy code
