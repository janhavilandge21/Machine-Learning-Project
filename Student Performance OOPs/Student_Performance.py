
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt

# ----------------- Data Handler Class -----------------
class DataHandler:
    def __init__(self, path_or_file):
        self.df = pd.read_csv(path_or_file)
        self.df.columns = [col.strip() for col in self.df.columns]  # Remove spaces
    
    def add_columns(self, score_col, study_col):
        if score_col not in self.df.columns or study_col not in self.df.columns:
            st.error(f"Columns not found. Available columns: {self.df.columns.tolist()}")
            return None

        def get_distinction(score):
            if score >= 90:
                return "1st-A"
            elif score >= 80:
                return "2nd-A+"
            elif score >= 70:
                return "3rd-B"
            elif score >= 50:
                return "4th-C"
            else:
                return "5th-Fail"

        self.df["Distinction"] = self.df[score_col].apply(get_distinction)
        self.df["Pass_Fail"] = self.df[score_col].apply(lambda x: "Pass" if x >= 40 else "Fail")
        return self.df

# ----------------- Regression Class -----------------
class RegressionModel:
    def __init__(self, df, study_col, score_col):
        self.X = df[[study_col]]
        self.y = df[score_col]
        self.model = LinearRegression()
    
    def train_linear(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)

        fig, ax = plt.subplots()
        ax.scatter(y_test, preds, color="blue")
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title("Linear Regression: Actual vs Predicted")
        return rmse, fig
    
    def predict_score(self, hours):
        return self.model.predict(np.array([[hours]]))[0]

# ----------------- Classification Class -----------------
class ClassificationModel:
    def __init__(self, df, study_col):
        self.X = df[[study_col]]
        self.y = df["Pass_Fail"]
        self.le = LabelEncoder()
        self.y_encoded = self.le.fit_transform(self.y)
        self.model = LogisticRegression()
    
    def train_logistic(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y_encoded, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        fig, ax = plt.subplots()
        cax = ax.matshow(cm, cmap="Blues")
        fig.colorbar(cax)
        ax.set_title("Confusion Matrix")
        return acc, fig
    
    def predict_pass_fail(self, hours):
        pred_encoded = self.model.predict(np.array([[hours]]))[0]
        return self.le.inverse_transform([pred_encoded])[0]

# ----------------- Clustering Class -----------------
class ClusteringModel:
    def __init__(self, df, score_col):
        self.X = df[[score_col]]
    
    def train_kmeans(self):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        model = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)

        fig, ax = plt.subplots()
        scatter = ax.scatter(range(len(self.X)), self.X, c=labels, cmap="viridis")
        ax.set_xlabel("Student Index")
        ax.set_ylabel("Score")
        ax.set_title("KMeans Clusters")
        fig.colorbar(scatter)
        return score, fig

# ----------------- Streamlit App -----------------
def main():
    st.title("🎓 Student ML App (OOP Version)")

    # ===== File Upload =====
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    if uploaded_file is not None:
        handler = DataHandler(uploaded_file)
    else:
        default_path = r"C:\Users\JANHAVI\Desktop\Dataset\Univeristy_Results(1).csv"
        try:
            handler = DataHandler(default_path)
            st.info("Using default dataset from local path.")
        except FileNotFoundError:
            st.warning("Please upload a dataset to continue.")
            return

    # ===== Column Selection =====
    st.subheader("Select Columns")
    st.write("Available columns:", handler.df.columns.tolist())
    score_col = st.selectbox("Select Score Column", handler.df.columns.tolist())
    study_col = st.selectbox("Select Study Hours Column", handler.df.columns.tolist())

    df = handler.add_columns(score_col, study_col)
    if df is None:
        return

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

    task = st.selectbox("Choose Task", ["Regression", "Classification", "Clustering"])

    if task == "Regression":
        reg = RegressionModel(df, study_col, score_col)
        rmse, fig = reg.train_linear()
        st.write(f"✅ RMSE: {rmse:.2f}")
        st.pyplot(fig)

        hours = st.number_input("Enter Study Hours to Predict Score", min_value=0.0, step=0.5)
        if st.button("Predict Score"):
            st.success(f"Predicted Score: {reg.predict_score(hours):.2f}")

    elif task == "Classification":
        clf = ClassificationModel(df, study_col)
        acc, fig = clf.train_logistic()
        st.write(f"✅ Accuracy: {acc:.2f}")
        st.pyplot(fig)

        hours = st.number_input("Enter Study Hours to Predict Pass/Fail", min_value=0.0, step=0.5, key="clf_hours")
        if st.button("Predict Pass/Fail"):
            st.success(f"Result: {clf.predict_pass_fail(hours)}")

    else:
        clus = ClusteringModel(df, score_col)
        score, fig = clus.train_kmeans()
        st.write(f"✅ Silhouette Score: {score:.2f}")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
