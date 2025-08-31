# 🏠 House Price Prediction – USA Housing Data
📌 Project Overview

This project predicts house prices based on various features like income, house age, rooms, bedrooms, and area population.
It involves:

Training multiple regression models.

Comparing their performance.

Deploying a Flask web app for predictions.

📂 Dataset

Source: USA_Housing.csv

Features:

Avg. Area Income

Avg. Area House Age

Avg. Area Number of Rooms

Avg. Area Number of Bedrooms

Area Population

Target: Price

🛠️ Technologies Used

Python 3

Flask → Web application framework

Pandas, NumPy → Data processing

Scikit-learn → ML models & evaluation

LightGBM, XGBoost → Advanced boosting models

Matplotlib / Seaborn (optional) → Visualization

🔎 Models Implemented

Linear Regression

Robust Regression (Huber)

Ridge, Lasso, ElasticNet

Polynomial Regression (degree=4)

SGD Regressor

Artificial Neural Network (MLP Regressor)

Random Forest

Support Vector Machine (SVR)

LightGBM

XGBoost

K-Nearest Neighbors

📊 Evaluation Metrics

Each model is evaluated using:

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

R² Score

Results are stored in:
📄 model_evaluation_results.csv

🚀 How to Run
🔹 1. Train Models

Run model.py to train and save models:

python model.py


This will:

Train all models

Save them as .pkl files

Generate model_evaluation_results.csv

🔹 2. Launch Flask App

Run the app:

python app.py


Visit in browser:

http://127.0.0.1:5000/

🌐 App Features

Select a model & enter input values → Get house price prediction

View model performance results in tabular format

📌 Future Improvements

Add hyperparameter tuning for better accuracy

Deploy to Heroku / AWS / Render for live access

Add visualization dashboard (e.g., Streamlit or Plotly Dash)
