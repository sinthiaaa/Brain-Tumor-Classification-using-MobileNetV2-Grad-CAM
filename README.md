# 🧠 Brain Tumor Classification using MobileNetV2 + Grad-CAM


<img width="1479" height="511" alt="image" src="https://github.com/user-attachments/assets/4b9babb2-2a2f-47df-b3f7-4e36bbcf5c68" />

A deep learning project for classifying brain tumors from MRI scans into 4 categories using Transfer Learning (MobileNetV2) with Grad-CAM explainability visualization.

---

## 📌 Problem Statement

Brain tumor detection from MRI is a critical and time-sensitive task in medical diagnosis. Manual analysis by radiologists is prone to fatigue and human error. This project explores how deep learning can assist in automated, explainable brain tumor classification.

---

## 🗂️ Dataset

- **Source:** [Brain Tumor MRI Dataset – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes:** Glioma, Meningioma, No Tumor, Pituitary
- **Size:** ~5,700 training images, 1,600 test images
- **Format:** JPG, resized to 224×224

---

## 🏗️ Model Architecture

```
MobileNetV2 (pretrained on ImageNet, frozen)
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dense(256, relu) → Dropout(0.4)
        ↓
Dense(128, relu) → Dropout(0.3)
        ↓
Dense(4, softmax)
```

- **Base Model:** MobileNetV2 (weights frozen during initial training)
- **Input Shape:** (224, 224, 3)
- **Output:** 4-class softmax probability

---

## 📊 Results

### Overall Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | **87%** |
| Macro F1-Score | 0.86 |
| Macro Precision | 0.87 |
| Macro Recall | 0.87 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 0.94 | 0.69 | 0.80 |
| Meningioma | 0.81 | 0.80 | 0.80 |
| No Tumor | 0.87 | 0.98 | 0.93 |
| Pituitary | 0.86 | 0.99 | 0.92 |

### Confusion Matrix

<img width="742" height="590" alt="image" src="https://github.com/user-attachments/assets/38c69e57-a662-497d-bd28-3ae1033194c7" />


### Key Observation
Glioma recall (0.69) is the primary bottleneck — likely due to visual similarity with meningioma in MRI scans. This highlights the need for fine-tuning and class-specific augmentation as next steps.

---

## 🔍 Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) is used to visualize which regions of the MRI the model focuses on when making predictions.

> Grad-CAM output shows the model correctly activating on tumor regions, providing interpretability essential for medical AI applications.

<img width="1161" height="3237" alt="image" src="https://github.com/user-attachments/assets/cbdebb30-b0b9-4862-bf24-1e95760d66cf" />

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-classification.git
cd brain-tumor-classification
```

### 2. Install Dependencies
```bash
pip install tensorflow keras scikit-learn matplotlib seaborn opencv-python kaggle
```

### 3. Download Dataset
```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip
```

### 4. Run the Notebook
Open `brain_tumor_classification.ipynb` in Google Colab or Jupyter Notebook and run all cells.

---

## 📁 Project Structure

```
brain-tumor-classification/
│
├── brain_tumor_classification.ipynb   # Main notebook
├── brain_tumor_model.h5               # Saved model
├── README.md                          # This file
└── results/
    ├── accuracy_loss_curve.png
    ├── confusion_matrix.png
    └── gradcam_output.png
```

---

## 🔬 Critical Analysis & Future Work

### Current Limitations
- Glioma recall (69%) indicates the model struggles to distinguish glioma from meningioma
- MobileNetV2 is optimized for mobile efficiency, not medical imaging precision

### Proposed Improvements
- **Fine-tuning:** Unfreeze last 30 layers of MobileNetV2 for domain adaptation
- **Architecture Upgrade:** Experiment with DenseNet121 — proven in medical imaging (e.g., CheXNet)
- **Class-specific Augmentation:** Apply stronger augmentation to glioma class
- **Cross-validation:** 5-fold CV for more robust evaluation
- **Deployment:** Streamlit web app for real-time inference

---

## 🛠️ Tech Stack

- Python 3.10+
- TensorFlow / Keras
- OpenCV
- Scikit-learn
- Matplotlib / Seaborn

---

