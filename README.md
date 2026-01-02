# Automated Lung Cancer Diagnosis Using CNN Architectures on CT Imaging

## Project Overview

This project implements an **Automated Lung Cancer Diagnosis System**
using **Convolutional Neural Networks (CNN)** trained on CT scan
images.\
The system provides:

-   A Flask-based web interface\
-   Users can upload a CT scan image\
-   CNN model predicts cancer type\
-   Confidence score displayed on screen

The model classifies images into **4 categories**:

1.  **Adenocarcinoma**\
2.  **Large Cell Carcinoma**\
3.  **Squamous Cell Carcinoma**\
4.  **Normal**

------------------------------------------------------------------------

## Dataset Description

Dataset used: **Chest CT-Scan Images Dataset (4-class)**\
Folder structure:

    Data/
    │── train/
    │     ├── Adenocarcinoma
    │     ├── Large cell carcinoma
    │     ├── Squamous cell carcinoma
    │     └── Normal
    │
    │── valid/
    │     ├── Adenocarcinoma
    │     ├── Large cell carcinoma
    │     ├── Squamous cell carcinoma
    │     └── Normal
    │
    └── test/
          ├── Adenocarcinoma
          ├── Large cell carcinoma
          ├── Squamous cell carcinoma
          └── Normal

Images are in **JPG/PNG format**, not DICOM.

------------------------------------------------------------------------

## How to Run the Project

### 1. Install Dependencies

    pip install -r requirements.txt

### 2. Train the Model

    python train_model.py

### 3. Run the Flask App

    python app.py

Open browser:

    http://127.0.0.1:5000/

Upload CT scan → get prediction.

------------------------------------------------------------------------

## Project Structure

    lung_cancer_cnn/
    │── app.py
    │── train_model.py
    │── requirements.txt
    │── README.md
    │
    │── model/
    │     └── lung_cancer_cnn.h5
    │
    │── templates/
    │     └── index.html
    │
    └── static/
          └── css/js files (optional)

------------------------------------------------------------------------

## Disclaimer

This project is for **educational purposes only**, not for real medical
diagnosis.

## 🔗 Model File Download Notice

NOTE:  
The trained model file (`lung_cancer_cnn.h5`) exceeds GitHub’s 100 MB file size limit, so it is **not stored in this repository**.

Please download the model from the Google Drive link below and place it inside the following directory before running the Flask app:


Google Drive link:  
**https://drive.google.com/file/d/1XyQDWaVLd5OuKMva4bXKKiwdBBn4b3QN/view?usp=sharing**
