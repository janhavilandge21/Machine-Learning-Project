#  🚗 Car MPG Prediction using Linear, Ridge & Lasso Regression
📌 Project Overview

This project focuses on predicting a car’s Miles Per Gallon (MPG) based on various attributes such as horsepower, displacement, weight, and origin. To improve model performance and reduce overfitting, Regularization techniques (Ridge & Lasso Regression) are applied and compared against a baseline Linear Regression.

📊 Dataset

Source: UCI Car MPG dataset

Rows: 398

Features:

cyl – Number of cylinders

disp – Engine displacement

hp – Horsepower

wt – Weight of the car

acc – Acceleration

yr – Model year

car_type – Car type (encoded)

origin – Region of origin (America, Asia, Europe)

mpg – Target variable

🛠️ Data Preprocessing

Removed irrelevant column: car_name

Converted categorical origin into dummy variables

Replaced missing values with median

Standardized features using sklearn.preprocessing.scale

🤖 Models Used

Simple Linear Regression

Baseline model to estimate MPG

Ridge Regression (L2 Regularization)

Penalizes large coefficients, reduces multicollinearity

Lasso Regression (L1 Regularization)

Performs feature selection by shrinking some coefficients to zero

📈 Model Evaluation

R² Score (Coefficient of Determination)

RMSE (Root Mean Squared Error)

Model	Train R²	Test R²
Linear Regression	0.834	0.851
Ridge Regression	0.834	0.851
Lasso Regression	0.794	0.838

✅ Ridge performed slightly better than Linear.
✅ Lasso reduced less important features, improving interpretability.

📊 Key Insights

Weight (wt) has the strongest negative impact on MPG.

Model year (yr) and car type significantly increase MPG.

Lasso eliminated weaker predictors, highlighting the most influential features.

📷 Visualizations

Residual plots for regression analysis

Predicted vs Actual MPG scatter plot

Coefficient comparison for Linear, Ridge, and Lasso


# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook

🚀 Future Improvements

Hyperparameter tuning with cross-validation

Adding Polynomial Regression for comparison

Deploying model using Streamlit/Flask
