# Machine-Learning-Project
# 🫁 Lung Cancer Detection using Machine Learning

This project focuses on detecting the likelihood of lung cancer based on various patient symptoms and lifestyle features. Multiple classification algorithms were implemented and compared for accuracy, precision, recall, and F1 score.

## 📊 Dataset
- Source: Survey Lung Cancer Dataset (CSV Format)
- Total Records: 309
- Features: 15 (e.g., Age, Gender, Smoking, Anxiety, Peer Pressure, etc.)
- Target Variable: LUNG_CANCER (Yes / No)

## 🔧 Tools & Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn (EDA & Visualization)
- Scikit-learn (ML Models)
- Jupyter Notebook

## 🧠 Models Implemented
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest

## ✅ Model Evaluation Metrics
Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 🎯 Best Performing Model
- **Random Forest Classifier**
  - Accuracy: **89.32%**
  - Precision: **90.42%**
  - Recall: **97.70%**
  - F1 Score: **93.92%**

## 📈 Data Preprocessing
- Categorical encoding for features like gender and target variable
- Null values checked and handled
- Feature correlation and distribution visualized

## 🔍 Visualizations
- Correlation Matrix (Heatmap)
- Feature Distributions
- Confusion Matrix (per model)

## 📂 Folder Structure

project-folder/
│
├── Lung_Cancer_Detection.ipynb
├── survey lung cancer.csv
├── README.md
└── model_outputs/
