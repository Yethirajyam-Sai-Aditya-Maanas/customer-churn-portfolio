
# 1. IMPORTS AND SETUP
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# 2. DATA LOADING AND MODEL TRAINING (with Caching)


# Use Streamlit's caching to load data and train the model only once
@st.cache_data
def load_data(csv_path):
    """Loads data from a CSV, cleans it, and returns a DataFrame."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{csv_path}' was not found. Please ensure it's in the same directory as app.py.")
        return None
        
    # Clean data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    
    # Create the target variable
    df['Churn_encoded'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return df

@st.cache_resource
def train_model(df):
    """Trains a logistic regression model using a pipeline."""
    # We use the original 'Churn' column for stratification and training
    X = df.drop(['customerID', 'Churn', 'Churn_encoded'], axis=1)
    y = df['Churn_encoded']

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', LogisticRegression(random_state=42))])
    
    # We train on the full dataset for the final app, but splitting is key for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate model to show performance metrics
    y_pred = model_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'], output_dict=True)
    
    return model_pipeline, report

# 3. STREAMLIT APP UI

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="👋",
    layout="wide"
)

# --- Load Data and Model ---
df_full = load_data('Telco-Customer-Churn.csv')

if df_full is not None:
    # We pass the dataframe with the encoded churn column to the model training function
    model_pipeline, report = train_model(df_full.copy())

    # --- Sidebar for Navigation ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Project Overview", "Exploratory Data Analysis", "Churn Prediction Tool", "Power BI Dashboard"])

    # --- Page 1: Project Overview ---
    if page == "Project Overview":
        st.title("Telco Customer Churn Analysis & Prediction")
        st.image("https://placehold.co/1200x300/0072B2/FFFFFF?text=Customer+Churn+Dashboard", use_container_width=True)

        st.header("Business Problem")
        st.write("""
        Customer churn is a critical challenge for telecommunication companies. This project aims to identify the key drivers of customer churn and develop a predictive model to identify customers who are at a high risk of leaving. 
        The ultimate goal is to enable the business to take proactive measures to retain these valuable customers.
        """)

        st.header("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Accuracy", f"{report['accuracy']:.2%}")
        col2.metric("Churn Precision", f"{report['Churn']['precision']:.2%}")
        col3.metric("Churn Recall", f"{report['Churn']['recall']:.2%}")
        st.info("The model's performance metrics show its ability to identify customers who are likely to churn (Recall) and the accuracy of those predictions (Precision).")

    # ---  Exploratory Data Analysis ---
    elif page == "Exploratory Data Analysis":
        st.title("Exploratory Data Analysis (EDA)")
        st.write("Visualizing the factors that influence customer churn.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn Rate by Contract Type")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df_full, x='Contract', hue='Churn', ax=ax1, palette='viridis')
            st.pyplot(fig1)
            st.write("**Insight:** Customers on a month-to-month contract are significantly more likely to churn.")

        with col2:
            st.subheader("Churn Rate by Tech Support")
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df_full, x='TechSupport', hue='Churn', ax=ax2, palette='magma')
            st.pyplot(fig2)
            st.write("**Insight:** Customers without tech support have a much higher churn rate.")

        st.subheader("Customer Tenure Distribution by Churn Status")
        fig3, ax3 = plt.subplots()
        sns.histplot(data=df_full, x='tenure', hue='Churn', multiple='stack', bins=30, palette='coolwarm')
        st.pyplot(fig3)
        st.write("**Insight:** New customers (low tenure) are more likely to churn.")

    # --- Churn Prediction Tool ---
    elif page == "Churn Prediction Tool":
        st.title("Interactive Churn Prediction")
        st.write("Select a customer's details to predict their churn probability.")
        
        # We need to drop the original Churn and the encoded version for prediction
        df_predict = df_full.drop(['Churn', 'Churn_encoded'], axis=1)

        # --- Predict on Existing Customer ---
        st.subheader("Predict for an Existing Customer")
        customer_id = st.selectbox("Select a Customer ID", options=df_predict['customerID'].unique())
        
        if st.button("Predict for Selected Customer"):
            customer_data = df_predict[df_predict['customerID'] == customer_id]
            prediction_proba = model_pipeline.predict_proba(customer_data)[:, 1][0]
            
            if prediction_proba > 0.5:
                st.error(f"This customer is at HIGH RISK of churning with a probability of {prediction_proba:.2%}.")
            else:
                st.success(f"This customer is at LOW RISK of churning with a probability of {prediction_proba:.2%}.")

      # --- Power BI Dashboard ---
    elif page == "Power BI Dashboard": 
        st.title("Power BI Customer Churn Dashboard")
        st.write("""
        This interactive dashboard was built in Power BI to analyze the key drivers of customer churn. 
        The recording below demonstrates its functionality, including slicers and cross-filtering.
        """)

   
        gif_url = "https://github.com/Yethirajyam-Sai-Aditya-Maanas/customer-churn-portfolio/blob/main/powerbi-demo.gif?raw=true"
    
        st.image(gif_url)
