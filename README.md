# 🧠 Multimodal Vision AI: Image Segmentation & Captioning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%2FCUDA-red.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-MS%20COCO%202017-yellow.svg)](https://cocodataset.org/)

An end-to-end multimodal Deep Learning project combining **Computer Vision** and **Natural Language Processing**. This application processes input images to simultaneously detect, segment, and localize objects with pixel-level masks while generating coherent descriptive captions explaining the scene context.

> **Project Motivation:**  
> I built this project as a hands-on deep dive to learn and understand core AI/ML architectures—specifically exploring vision feature extraction, sequence-to-sequence attention mechanisms (Transformers), and deploying hybrid PyTorch/TensorFlow pipelines locally on Apple Silicon hardware.

---

## ✨ Key Features

- **Transformer-based Image Captioning:** Translates high-level visual features into descriptive natural language sentences using multi-head self-attention and cross-attention.
- **Instance Segmentation & Object Detection:** Detects individual object classes, generates bounding boxes, and overlays pixel-level segmentation masks using **Mask R-CNN (ResNet-50-FPN)** or custom **U-Net**.
- **Interactive Streamlit Web Dashboard:** Upload any local image or enter an image URL to inspect real-time segmentation overlays, confidence scores, and predicted captions.
- **Hardware Acceleration:** Native support for Apple Silicon GPU acceleration (**MPS / Metal**) and CUDA.

---

## 🏗️ Architecture & How It Works
                  ┌───────────────────────────────────────────────┐
                    │                  Input Image                  │
                    └───────┬───────────────────────────────┬───────┘
                            │                               │
                [Segmentation Pipeline]            [Captioning Pipeline]
                            │                               │
               Mask R-CNN (ResNet-50-FPN)        InceptionV3 Feature Extractor
                            │                               │
                    Region Proposals               Spatial Visual Features
                            │                               │
                 Bounding Boxes & Masks            Transformer Encoder
                            │                               │
                            │                      Transformer Decoder
                            │                  (Multi-Head Cross-Attention)
                            │                               │
                    Pixel Mask Overlays             Natural Language Caption
                            └───────┬───────────────────────┘
                                    ▼
                         Streamlit Interactive UI


### 1. Captioning Pipeline (CNN + Transformer)
- **Visual Encoder:** An `InceptionV3` backbone pretrained on ImageNet extracts 2048-dimensional spatial feature representations from resized $299 \times 299$ images.
- **Transformer Encoder:** Processes extracted visual embeddings through self-attention layers with layer normalization and residual connections.
- **Transformer Decoder:** Utilizes causal self-attention over generated tokens combined with multi-head cross-attention over visual encoder embeddings to generate sequential tokens until the `[end]` sequence is reached.
- **Vocabulary:** Tokenized over MS COCO captions with custom standardization and positional embeddings.

### 2. Segmentation Pipeline
- **Mask R-CNN:** Employs a Feature Pyramid Network (FPN) with a ResNet-50 backbone for multi-scale region proposal, classification, and binary mask generation across 80 COCO categories.
- **Custom U-Net Option:** Includes an encoder-decoder architecture with skip connections for custom semantic segmentation experiments.

---

## 📁 Repository Structure

```text
Image_Segmentation_Captioning/
├── Pretrain_app.py          # Main Streamlit app (Mask R-CNN + Transformer Captioner)
├── app.py                   # Streamlit app for custom U-Net segmentation
├── model.py                 # Transformer captioning model definition & inference utilities
├── Image_Caption_train.ipynb # Training pipeline for InceptionV3 + Transformer on COCO
├── image-segmentation.ipynb # Training pipeline for custom segmentation model
├── saved_models/            # Stored model weights (.h5 / .weights.h5) [git-ignored]
├── saved_vocabulary/        # Serialized tokenizer vocabulary (vocab_coco.file)
└── requirements.txt         # Project dependencies


🚀 Installation & Local Setup
1. Clone the Repository
bash
git clone https://github.com/ashishverma101/Image_Segmentation_Captioning.git
cd Image_Segmentation_Captioning
2. Create and Activate a Virtual Environment
bash
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
On Apple Silicon (M1/M2/M3/M4 Mac with Metal GPU):
bash
pip install --upgrade pip
pip install tensorflow-macos tensorflow-metal
pip install torch torchvision torchaudio
pip install streamlit Pillow numpy pandas matplotlib opencv-python requests tqdm
On Linux / Windows / Standard Environments:
bash
pip install --upgrade pip
pip install tensorflow torch torchvision torchaudio
pip install streamlit Pillow numpy pandas matplotlib opencv-python requests tqdm
🏋️ Training the Models
Captioning Model (Image_Caption_train.ipynb)
Open Image_Caption_train.ipynb in Kaggle (or Google Colab) with a GPU accelerator enabled (e.g. Tesla T4).
Attach the MS COCO 2017 Dataset.
Run the notebook to train the InceptionV3 + Transformer pipeline across caption pairs.
Download the generated artifacts and organize them into your local project directory:
vocab_coco.file 
→
→ saved_vocabulary/vocab_coco.file
image_captioning_coco_weights.weights.h5 
→
→ saved_models/image_captioning_coco_weights.weights.h5
Custom Segmentation Model (image-segmentation.ipynb)
Run image-segmentation.ipynb to train the custom U-Net segmentation network.
Save weights as saved_models/segmentation_model.h5.
🖥️ Running the Application
Option A: Pretrained ResNet / Mask R-CNN App (Recommended)

This mode runs torchvision's Mask R-CNN instance segmentation alongside the trained Transformer captioning model:

bash
streamlit run Pretrain_app.py

Open http://localhost:8501 in your browser, upload any image, and view generated captions and segmentation masks.

Option B: Custom U-Net App

To run the app with your custom-trained segmentation network:

bash
streamlit run app.py
💡 Key Learnings & Takeaways
Multimodal Alignment: How visual features from a spatial CNN encoder can be projected into the latent space of an autoregressive Transformer decoder.
Attention Mechanisms: Implementing causal attention masks to prevent look-ahead bias and multi-head cross-attention to focus on relevant image regions during text generation.
Hardware Acceleration: Configuring Apple Silicon Metal Performance Shaders (MPS) and fallback routines for mixed-framework pipelines (PyTorch + TensorFlow).
Model Deployment: Building interactive ML web dashboards using Streamlit with cached model loaders to minimize inference latency.