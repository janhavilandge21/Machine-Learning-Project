# K Nearest Neighbours (kNN) Classification - Breast Cancer Prediction
📂 Project Overview

This project implements the K Nearest Neighbours (kNN) classification algorithm to predict whether a breast tumor is benign or malignant using the Breast Cancer dataset. The goal is to build an efficient classification model, evaluate its performance, and analyze metrics such as accuracy, precision, recall, and ROC AUC.

📊 Dataset

Source: Breast cancer dataset (CSV file)

Total records: 698

Features:

Clump Thickness

Uniformity of Cell Size

Uniformity of Cell Shape

Marginal Adhesion

Single Epithelial Cell Size

Bare Nuclei

Bland Chromatin

Normal Nucleoli

Mitoses

Target Variable:

Class → 2 (benign), 4 (malignant)

🔢 Key Steps in the Project
1. Data Loading and Exploration

Loaded dataset using pandas.

Checked shape, data types, and missing values.

Analyzed frequency distributions and summary statistics.

2. Data Cleaning

Converted the Bare_Nuclei column to numeric and handled missing values using median imputation.

Dropped irrelevant columns (e.g., Id).

3. Data Visualization

Plotted histograms and heatmaps to understand feature distributions and correlations.

Used seaborn and matplotlib for visual representation.

4. Feature Engineering and Preprocessing

Defined feature matrix X and target vector y.

Split data into training and testing sets (80% train, 20% test).

Applied standard scaling to normalize the dataset.

5. Model Building

Used KNeighborsClassifier from scikit-learn.

Experimented with different values of k (3, 5, 6, 7, 8, 9).

Selected the best-performing model based on accuracy and other metrics.

6. Model Evaluation

Checked model accuracy on training and test data.

Analyzed confusion matrix and classification report.

Computed metrics like:

Accuracy

Precision

Recall

Specificity

ROC AUC

7. Cross-validation

Performed 10-fold cross-validation to validate the model's performance and stability.

✅ Key Results

Best model accuracy (with k=3): 97.14%

ROC AUC: 0.9883

Cross-validated ROC AUC: 0.9811

Model generalizes well with consistent cross-validation results.

📂 Tools & Libraries

Python 3.x

Jupyter Notebook

Libraries:

numpy, pandas

matplotlib, seaborn

scikit-learn (KNeighborsClassifier, train_test_split, StandardScaler, roc_curve, classification_report, etc.)

🚀 How to Run This Project

Install the required libraries:

pip install numpy pandas matplotlib seaborn scikit-learn


Load the dataset (breast cancer.csv) into the same folder.

Open the Jupyter Notebook K Nearest Neighbours.ipynb.

Run the notebook cells step-by-step to explore, preprocess, train, and evaluate the model.

📈 Insights

kNN is simple yet effective for classification tasks.

Preprocessing steps like missing value imputation and scaling are crucial.

Cross-validation helps ensure the robustness of the model.

ROC AUC gives a deeper understanding of the model's performance across thresholds.


