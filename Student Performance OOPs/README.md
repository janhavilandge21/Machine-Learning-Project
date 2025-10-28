# 🎓 Student ML App 

A Streamlit-based Machine Learning Web Application built using an Object-Oriented Programming (OOP) approach.
This app allows users to analyze student performance using Regression, Classification, and Clustering models with interactive visualizations.

# 🚀 Features

✅ Upload CSV Dataset or use a default dataset

✅ Auto Data Cleaning (column trimming and mapping)

✅ Dynamic Column Selection

✅ Regression Model (Predict Scores using Study Hours)

✅ Classification Model (Predict Pass/Fail outcomes)

✅ Clustering Model (Group students by performance)

✅ Performance Metrics & Visualizations

Linear Regression Scatter Plot

Confusion Matrix Heatmap

KMeans Cluster Visualization

# 🧠 Machine Learning Models Used

Task	Algorithm	Evaluation Metric

Regression	Linear Regression	RMSE (Root Mean Squared Error)

Classification	Logistic Regression	Accuracy & Confusion Matrix

Clustering	KMeans	Silhouette Score

# ⚙️ Installation and Setup

Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


Upload a CSV file 
Ensure your dataset includes:

A score column (numeric)

A study hours column (numeric)

# 🧾 Example Dataset Format
Student_ID  	Study_Hours	Score

1	2.5       	55

2	5.0       	78

3	1.0	        35

4	8.0	        92

# 📊 App Preview
🔹 Regression

Predict student scores based on study hours using Linear Regression.

🔹 Classification

Classify students as Pass or Fail using Logistic Regression.

🔹 Clustering

Group students into 5 performance clusters using KMeans Clustering.

# 🧩 Technologies Used

Python 3.12+

Streamlit

Pandas, NumPy

Scikit-learn

Matplotlib

# 🧰 Requirements

Create a requirements.txt file:

streamlit

pandas

numpy

scikit-learn

matplotlib

# 💡 Future Improvements

🔸 Add more ML models (Decision Tree, Random Forest)

🔸 Integrate Power BI or Plotly Dash visualizations

🔸 Enable live dataset editing and export options

🔸 Add data preprocessing insights and reports

