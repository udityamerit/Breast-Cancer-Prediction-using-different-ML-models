# 🧬 OncoVision AI: Breast Cancer Cytology Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

An advanced, interactive web application and machine learning engine designed to predict whether a breast tumor is **Benign** or **Malignant** based on quantitative cellular measurements extracted from digitized fine needle aspirate (FNA) images.

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset Context](#-dataset-context)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Installation & Setup](#-installation--setup)
- [Application Structure](#-application-structure)
- [Disclaimer](#%EF%B8%8F-medical-disclaimer)

---

## 🔬 Project Overview
OncoVision AI leverages state-of-the-art Machine Learning algorithms (optimized Support Vector Machines) to provide clinical decision support. By analyzing 30 distinct morphological features of cell nuclei (such as radius, texture, perimeter, and smoothness), the AI Engine accurately classifies tumor signatures and provides a statistical confidence score.

The accompanying dashboard provides deep visual insights into the dataset, allowing researchers and students to understand how artificial intelligence interprets clinical oncology data.

## ✨ Key Features
* **🩺 AI Diagnostic Engine**: Input cytology report metrics to receive an instant prediction (Benign/Malignant) alongside an AI Certainty Gauge.
* **🌌 Advanced Data Visualizer**: High-performance, clinical-grade visualizations using Seaborn and Matplotlib:
    * **PCA Morphological Clustering**: 2D dimensionality reduction showing how the algorithm separates malignant and benign structures.
    * **Tumor Density Topography**: Heatmaps illustrating concentration zones of tumor types.
    * **Clinical Blueprint Radar Charts**: Compare the average "shape" of a cancerous vs. non-cancerous cell.
    * **Feature Correlation Matrix**: Identify highly correlated cellular features.
* **🗂️ Clinical Database**: A built-in data explorer to view the raw patient records from the WDBC dataset.
* **🌙 Next-Gen Medical UI**: A sleek, responsive, dark-mode clinical dashboard built with Streamlit.

---

## 📊 Dataset Context
This project utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset**.
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

**Core Feature Categories Analyzed (Mean, Standard Error, Worst):**
* **Radius** (mean of distances from center to points on the perimeter)
* **Texture** (standard deviation of gray-scale values)
* **Perimeter** & **Area**
* **Smoothness** (local variation in radius lengths)
* **Compactness** (perimeter² / area - 1.0)
* **Concavity** & **Concave Points**
* **Symmetry** & **Fractal Dimension**

---

## 🧠 Machine Learning Pipeline
1.  **Preprocessing**: Data cleaning, removal of ID columns, and zero-mean/unit-variance scaling using `StandardScaler`.
2.  **Model Selection**: Evaluated Logistic Regression, Decision Trees, Random Forests, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM).
3.  **Hyperparameter Tuning**: Utilized `GridSearchCV` to prevent overfitting and optimize the models.
4.  **Final Selection**: **Support Vector Machine (SVM)** was selected as the final production model due to its exceptional Recall score (minimizing False Negatives, which is critical in cancer diagnosis).

---

## 💻 Installation & Setup

To run this application locally on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/udityamerit/breast-cancer-prediction-using-different-ml-models.git](https://github.com/udityamerit/breast-cancer-prediction-using-different-ml-models.git)
cd breast-cancer-prediction-using-different-ml-models

```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```
### 3. Install Dependencies

Make sure you have all required libraries installed:

```bash
pip install -r requirements.txt

```
*(If you don't have a requirements file, run: `pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly joblib`)*

### 4. Run the Streamlit Application

```bash
streamlit run app.py

```



The application will automatically open in your default web browser at `http://localhost:8501`.

---



## 📁 Application Structure

```text

├── data.csv                 # Raw Wisconsin Diagnostic dataset
├── model.ipynb              # Jupyter Notebook containing EDA and Model Training
├── app.py                   # Main Streamlit Dashboard application file
├── breast_cancer_model.pkl  # Serialized, pre-trained optimal ML model
├── scaler.pkl               # Serialized feature scaler for input normalization
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation

```

---

## ⚠️ Medical Disclaimer

**For Investigational and Educational Use Only.** This application is a machine learning proof-of-concept. It is **not** a regulated medical device and should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician, oncologist, or other qualified health provider with any questions you may have regarding a medical condition.

---