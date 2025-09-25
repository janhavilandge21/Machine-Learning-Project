import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

def create_dataset():
    np.random.seed(0)
    df = pd.DataFrame({
        'Feature1': np.random.randint(10, 50, 20),
        'Feature2': np.random.randint(5, 25, 20),
        'DV': np.random.randint(50, 200, 20)
    })
    return df

# ----------------------------- #
# EDA
# ----------------------------- #
def run_eda(df):
    st.subheader("Dataset Overview")
    st.dataframe(df)
    st.write(df.describe())
   
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'{col} Distribution')
    st.pyplot(fig)

# ----------------------------- #
# Regression
# ----------------------------- #
def run_regression(df, dt_max_depth, dt_min_samples, rf_n_estimators, rf_max_depth):
    X = df[['Feature1','Feature2']]
    y = df['DV']
    
    regressors = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=dt_max_depth, min_samples_split=dt_min_samples),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=rf_n_estimators, max_depth=rf_max_depth),
        'SVR': SVR()
    }
    
    results = {}
    coef_plots = {}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    for name, model in regressors.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        df[name+'_Pred'] = model.predict(X)
        
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        results[name] = {'Train MSE': mse_train, 'Test MSE': mse_test}
        
        if mse_test > mse_train*1.5:
            results[name]['Warning'] = 'Potential Overfitting'
        elif mse_train > mse_test*1.5:
            results[name]['Warning'] = 'Potential Underfitting'
        else:
            results[name]['Warning'] = 'Good Fit'
        
        if name=='Linear Regression':
            coef = model.coef_
        elif name in ['Decision Tree','Random Forest']:
            coef = model.feature_importances_
        else:
            coef = None
        if coef is not None:
            fig, ax = plt.subplots()
            ax.bar(X.columns, coef, color='skyblue')
            ax.set_title(f'{name} Feature Importance / Coefficients')
            coef_plots[name] = fig
    
    return df, results, coef_plots

# ----------------------------- #
# Clustering
# ----------------------------- #
def run_clustering(df, n_clusters):
    X = df[['Feature1','Feature2']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clusterers = {
        'KMeans': KMeans(n_clusters=n_clusters, random_state=42),
        'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
        'DBSCAN': DBSCAN(eps=1.5, min_samples=2)
    }
    
    for name, model in clusterers.items():
        clusters = model.fit_predict(X_scaled)
        df[name+'_Cluster'] = clusters
        
        fig, ax = plt.subplots()
        sns.scatterplot(x='Feature1', y='Feature2', hue=clusters, palette='Set1', data=df, ax=ax, s=100)
        ax.set_title(f"{name} Clustering")
        st.pyplot(fig)
    
    return df

# ----------------------------- #
# Classification
# ----------------------------- #
def run_classification(df, dt_max_depth, dt_min_samples, rf_n_estimators, rf_max_depth):
    df['Target'] = np.where(df['DV'] > df['DV'].median(), 'High','Low')
    le = LabelEncoder()
    df['Target_enc'] = le.fit_transform(df['Target'])
    
    X = df[['Feature1','Feature2']]
    y = df['Target_enc']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=dt_max_depth, min_samples_split=dt_min_samples),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=rf_n_estimators, max_depth=rf_max_depth),
        'SVM': SVC()
    }
    
    results = {}
    feature_plots = {}
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        y_pred = clf.predict(X)
        df[name+'_Class'] = y_pred
        
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        results[name] = {'Train Acc': acc_train, 'Test Acc': acc_test}
        
        if acc_train>acc_test*1.2:
            results[name]['Warning'] = 'Potential Overfitting'
        elif acc_test>acc_train*1.2:
            results[name]['Warning'] = 'Potential Underfitting'
        else:
            results[name]['Warning'] = 'Good Fit'
        
        if name in ['Decision Tree','Random Forest']:
            coef = clf.feature_importances_
            fig, ax = plt.subplots()
            ax.bar(X.columns, coef, color='orange')
            ax.set_title(f'{name} Feature Importance')
            feature_plots[name] = fig
        
        fig, ax = plt.subplots()
        sns.scatterplot(x='Feature1', y='Feature2', hue=y_pred, palette='Set2', data=df, ax=ax, s=100)
        ax.set_title(f"{name} Classification")
        st.pyplot(fig)
    
    return df, results, feature_plots

# ----------------------------- #
# Streamlit Tabs with Filtering & Hyperparameters
# ----------------------------- #
def main():
    st.title("Interactive ML Project with Filtering & Hyperparameters")
    
    df = create_dataset()
    
    # Sidebar for filtering
    st.sidebar.header("Filter Dataset")
    f1_min, f1_max = int(df['Feature1'].min()), int(df['Feature1'].max())
    f2_min, f2_max = int(df['Feature2'].min()), int(df['Feature2'].max())
    
    feature1_range = st.sidebar.slider("Feature1 Range", f1_min, f1_max, (f1_min, f1_max))
    feature2_range = st.sidebar.slider("Feature2 Range", f2_min, f2_max, (f2_min, f2_max))
    
    filtered_df = df[(df['Feature1'] >= feature1_range[0]) & (df['Feature1'] <= feature1_range[1]) &
                     (df['Feature2'] >= feature2_range[0]) & (df['Feature2'] <= feature2_range[1])]
    
    tabs = st.tabs(["EDA", "Regression", "Clustering", "Classification"])
    
    # EDA Tab
    with tabs[0]:
        st.header("Exploratory Data Analysis")
        run_eda(filtered_df)
    
    # Regression Tab
    with tabs[1]:
        st.header("Regression Models Hyperparameters")
        dt_max_depth = st.slider("Decision Tree Max Depth", 1, 10, 3)
        dt_min_samples = st.slider("Decision Tree Min Samples Split", 2, 10, 2)
        rf_n_estimators = st.slider("Random Forest n_estimators", 10, 200, 50)
        rf_max_depth = st.slider("Random Forest Max Depth", 1, 10, 3)
        
        df_reg, reg_results, reg_coefs = run_regression(filtered_df, dt_max_depth, dt_min_samples, rf_n_estimators, rf_max_depth)
        st.write("Regression Metrics & Overfitting Check:")
        st.write(reg_results)
        st.subheader("Regression Feature Importance / Coefficients")
        for name, fig in reg_coefs.items():
            st.pyplot(fig)
        st.subheader("Regression Predictions")
        st.dataframe(df_reg)
    
    # Clustering Tab
    with tabs[2]:
        st.header("Clustering Models Hyperparameters")
        n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        df_clust = run_clustering(filtered_df, n_clusters)
        st.dataframe(df_clust)
    
    # Classification Tab
    with tabs[3]:
        st.header("Classification Models Hyperparameters")
        dt_max_depth_cls = st.slider("Decision Tree Max Depth (Classification)", 1, 10, 3)
        dt_min_samples_cls = st.slider("Decision Tree Min Samples Split (Classification)", 2, 10, 2)
        rf_n_estimators_cls = st.slider("Random Forest n_estimators (Classification)", 10, 200, 50)
        rf_max_depth_cls = st.slider("Random Forest Max Depth (Classification)", 1, 10, 3)
        
        df_class, class_results, class_feats = run_classification(filtered_df, dt_max_depth_cls, dt_min_samples_cls, rf_n_estimators_cls, rf_max_depth_cls)
        st.write("Classification Metrics & Overfitting Check:")
        st.write(class_results)
        st.subheader("Classification Feature Importance")
        for name, fig in class_feats.items():
            st.pyplot(fig)
        st.subheader("Classification Predictions")
        st.dataframe(df_class)

if __name__=="__main__":
    main()
