# 🔬 Histopathological Image Classification with CNN and ResNet-50

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange)
![Publication](https://img.shields.io/badge/Published-UNIGOU%202025-success)
![Environment](https://img.shields.io/badge/Developed%20on-Google%20Colab-F9AB00)

## 📄 Publication

This project resulted in an international academic publication:

**Detection of Lung Cancer Through Histopathological Images**  
UNIGOU Remote 2025 – Czech-Brazilian Academic Program  

📎 Read the full paper here:  
[Download Publication (PDF)](https://www.incbac.org/wp-content/uploads/2025/06/UNIGOU-Remote-Publication-2025-Mirelle-Silva-Vieira.pdf)

## Project Overview

This project focuses on lung cancer classification using deep learning techniques applied to histopathological images.

Two architectures were implemented and compared:

- A custom Convolutional Neural Network (CNN)
- A pre-trained ResNet-50 (transfer learning)

The models were trained and evaluated on three classes:

- Adenocarcinoma  
- Benign  
- Squamous Cell Carcinoma

<img width="857" height="305" alt="histopathological_images" src="https://github.com/user-attachments/assets/20cc782f-ea0e-4478-9f2f-32c4466fad59" />

## 🛠️ Technologies

- Python 3.13
- PyTorch 2.6.0
- NumPy 2.2.2
- scikit-learn 1.6.1
- Matplotlib 3.10.1

## 💻 Development Environment

This project was developed and executed using **Google Colab**, leveraging:

- NVIDIA T4 GPU
- Cloud-based dataset integration via Kaggle API
- PyTorch framework

## 📁 Dataset

Dataset used: The experiments were conducted using the 
[Histopathological Images](https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images ) dataset available on Kaggle.

- 15,000 images
- 3 balanced classes
- 5,000 images per class

Images were resized to 224×224 and normalized using ImageNet statistics.

## Model Architecture

### 🔹 Custom CNN

- 2 Convolutional layers (32, 64 filters)
- MaxPooling layers
- 2 Fully Connected layers
- ReLU activation
- CrossEntropyLoss
- SGD optimizer (lr=0.001, momentum=0.9)
- 50 epochs
- Batch size: 32

### 🔹 ResNet-50

- Pre-trained on ImageNet
- Final layer modified for 3 classes
- Fine-tuned on histopathological dataset

## 📊 Results

### Custom CNN
- Final Accuracy: **94.90%**

### ResNet-50
- Final Accuracy: **99.93%**
- Only 2 misclassifications

ResNet-50 demonstrated faster convergence and superior generalization performance.

## 📈 Performance Comparison

ResNet-50:

- Faster convergence
- Lower training loss
- Higher classification robustness

Both models stabilized within relatively few epochs, indicating efficient learning dynamics.

## ▶️ Running the Notebook (Google Colab Recommended)

1. Download the `.ipynb` file from this repository.
2. Open Google Colab: https://colab.research.google.com/
3. Upload the notebook.
4. Enable GPU:
   - Runtime → Change runtime type → Select GPU.
5. Run all cells sequentially.

⚠️ Make sure to download the dataset from Kaggle and upload it to the Colab session before execution.
