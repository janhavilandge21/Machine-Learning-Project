# 🥑 Avocado Price Analysis & Prediction
📌 Project Overview

This project focuses on analyzing and predicting avocado prices across different regions in the United States from 2015–2018. Using the dataset, we perform Exploratory Data Analysis (EDA), visualize sales patterns, and build Machine Learning & Deep Learning models to predict avocado prices.

📂 Dataset

Source: [avocado.csv]

Shape: 18,249 rows × 14 columns

Features include:

Date – date of observation

AveragePrice – average price of avocados

Total Volume – total number of avocados sold

4046, 4225, 4770 – sales volume by PLU type

Total Bags, Small Bags, Large Bags, XLarge Bags

type – conventional or organic

year – year of sale

region – geographical region

🔍 Exploratory Data Analysis (EDA)

Checked missing values, duplicates, and distributions.

Analyzed price variation across years, regions, and types (conventional vs. organic).

Identified outliers in volume using boxplots.

Created grouped summaries and pivot tables to compare organic vs. conventional prices.

Extracted Month feature from Date for seasonality analysis.

📊 Visualizations

Price trends across regions and years.

Distribution plots for average price.

Boxplots comparing organic vs. conventional avocados.

Month-wise sales distribution.

🤖 Machine Learning Models

Implemented and compared multiple regression models:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Support Vector Regression (SVR)

K-Nearest Neighbors (KNN)

XGBoost

📈 Model Performance (R² score):

Linear Regression → 0.598

Decision Tree → 0.706

Random Forest → 0.854 ✅

SVR → 0.669

KNN → 0.629

XGBoost → 0.849

🧠 Deep Learning Model

Implemented a Deep Neural Network (DNN) using TensorFlow/Keras.

Architecture: Multiple Dense layers with ReLU activation and Dropout.

Achieved R² ≈ 0.658 (moderate performance compared to Random Forest & XGBoost).

⚙️ Tech Stack

Python: Pandas, NumPy, Matplotlib, Seaborn

Scikit-learn: ML models & preprocessing

XGBoost: Gradient Boosted Trees

TensorFlow/Keras: Deep Learning

Jupyter Notebook: Data analysis and modeling

🚀 How to Run the Project

Run Jupyter Notebook:

jupyter notebook


Open and explore:

EDA in Avocado.ipynb → Exploratory analysis

Avocados Data Analysis project.ipynb → Data cleaning & visualizations

Price Regression.ipynb → ML/DL model building

📌 Key Insights

Organic avocados are consistently more expensive than conventional ones.

Price fluctuations show seasonal and regional trends.

Random Forest and XGBoost provided the best price prediction accuracy.
