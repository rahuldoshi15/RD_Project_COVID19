# COVID-19 Detection from Chest X-rays using Convolutional Neural Networks (CNN)

## Project Overview

This project presents an automated deep learning system for detecting COVID-19 from chest X-ray images. The model classifies X-ray images into three categories:

- COVID-19
- Normal
- Viral Pneumonia

The system is designed as a screening support tool to assist healthcare professionals by providing rapid and consistent image classification using Artificial Intelligence.

This project was developed as part of the MCA (AI) Minor Project – 21CSA697A.

---

## Problem Statement

Manual interpretation of chest radiographs is time-consuming and requires significant expertise. During high patient volumes, diagnostic delays may occur.

This project addresses the need for an automated multi-class classification system capable of distinguishing between:

- COVID-19 infection
- Viral Pneumonia
- Healthy lungs (Normal)

---

## Model Architecture

The implemented model is a Custom Convolutional Neural Network (CNN) built using TensorFlow/Keras.

### Architecture Details:

- Conv2D (32 filters, 3×3 kernel, ReLU)
- MaxPooling2D (2×2)
- Conv2D (64 filters, 3×3 kernel, ReLU)
- MaxPooling2D (2×2)
- GlobalAveragePooling2D
- Dense Layer (128 neurons, ReLU)
- Dropout (0.5)
- Output Layer (Softmax – 3 classes)

### Training Configuration:

- Loss Function: Categorical Crossentropy
- Optimizer: Adam (learning rate = 0.0001)
- Input Image Size: 128 × 128 RGB
- Data Augmentation: Rotation, Zoom, Horizontal Flip
- EarlyStopping to prevent overfitting

---

## Dataset

Dataset Used:
**COVID-19 Radiography Database (Kaggle)**

The dataset contains labeled chest X-ray images categorized into:

- COVID
- Normal
- Viral Pneumonia
- (Lung Opacity class ignored for this project)

Dataset Link:
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

---

## Evaluation Metrics

The model performance is evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- Confusion Matrix

Special emphasis was placed on Recall for COVID-19 cases to minimize False Negatives.

---

## Project Structure
RD_Project_COVID19/
│
├── train_model.ipynb
├── covid_xray_model.h5
├── requirements.txt
├── app.py
└── README.md


---

## Installation & Setup

### 1. Clone Repository

git clone https://github.com/rahuldoshi15/RD_Project_COVID19.git

cd RD_Project_COVID19

### 2. Install Dependencies

pip install -r requirements.txt

---

## Running the Project

### Run Training Notebook

Open:

RD_Covid_MiniProject.ipynb


Run all cells in Google Colab or Jupyter Notebook.

---

### Run Streamlit Web Application
streamlit run app.py


Then open the provided local URL in your browser.

---

## Streamlit Deployment

The Streamlit interface allows users to:

- Upload a chest X-ray image
- Receive prediction (COVID / Normal / Viral Pneumonia)
- View confidence score
- Visualize prediction probabilities

---

## Key Contributions

- Custom CNN architecture optimized for 3-class classification
- Data augmentation for improved generalization
- GPU-accelerated training
- Real-time Streamlit deployment
- Academic reproducibility via GitHub version control

---

## Disclaimer

This system is developed strictly for academic and research purposes.

It is not intended for clinical diagnosis, medical advice, or real-world medical decision-making.

---

## Author

Rahul Doshi  
MCA (Artificial Intelligence)  
Amrita Vishwa Vidyapeetham  
Minor Project – 21CSA697A  
February 2026
