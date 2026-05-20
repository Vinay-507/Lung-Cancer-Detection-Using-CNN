# Team Structure -- Automated Lung Cancer Diagnosis Using CNN

This document defines the team structure and responsibilities for the
project **Automated Lung Cancer Diagnosis Using CNN Architectures on CT
Imaging**.

------------------------------------------------------------------------

## 👥 Team Overview

The project is divided into **3 groups**, each containing 
based on core components:

-   **1 Member  → Website / Frontend**
-   **2 Members → Model Training**
-   **1 Member → Flask Backend & Integration**

------------------------------------------------------------------------

## 🖥️ Team A --- Website / Frontend (1 Member)

### **Member:**

-   Sazad

### **Responsibilities:**

-   Design and build the frontend interface\
-   Create `index.html`\
-   Add image upload UI\
-   Display prediction and confidence\
-   Add optional CSS styling\
-   Ensure compatibility with Flask templates

### **Deliverables:**

-   `templates/index.html`\
-   `static/style.css` (optional)

------------------------------------------------------------------------

## 🧠 Team B --- Model Training (2 Members)

### **Members:**

-   Gowtham
-   Bhagya Sri

### **Responsibilities:**

-   Prepare dataset (train/valid/test)\
-   Build and train the CNN model\
-   Evaluate model performance\
-   Save final trained model\
-   Provide training results

### **Deliverables:**

-   `train_model.py`\
-   `model/lung_cancer_cnn.h5`\
-   Accuracy/Loss graphs

------------------------------------------------------------------------

## 🔧 Team C --- Flask Backend (1 Member)

### **Member:**

-   Vinay

### **Responsibilities:**

-   Create Flask backend (`app.py`)\
-   Load trained CNN model\
-   Integrate frontend and model\
-   Handle file uploads\
-   Display prediction on webpage

### **Deliverables:**

-   `app.py`\
-   Working Flask application\
-   Full integration of UI + Model

------------------------------------------------------------------------

## 🧩 Summary Table

  Team                    Members   Role                  Deliverables
  ----------------------- --------- --------------------- --------------------
  **Team A -- Website**        1       UI /Frontend          index.html

  **Team B -- Model**        2 & 3     CNN training          lung_cancer_cnn.h5

  **Team C -- Flask**          4       Backend integration   app.py

------------------------------------------------------------------------
