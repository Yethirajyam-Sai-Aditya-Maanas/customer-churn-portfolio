# Customer Churn Prediction & Analysis App

An end-to-end data analytics project that uses a machine learning model to predict customer churn and visualizes key business insights using an interactive dashboard.



## 🚀 Live Application

**You can access the live, deployed Streamlit application here:** [https://customer-churn-portfolio-hcgvkpygmkmyi8t4n8v5wm.streamlit.app/](https://customer-churn-portfolio-hcgvkpygmkmyi8t4n8v5wm.streamlit.app/)


---

## 📋 Project Overview

This project tackles the critical business problem of customer churn within a telecommunications company. By analyzing historical customer data, this application aims to:
1.  **Identify** the key factors that lead to customers leaving the service.
2.  **Predict** which existing customers are at a high risk of churning.
3.  **Present** these findings through an interactive and easy-to-understand web interface.

---

## ✨ Features

This application is a multi-page tool that provides a comprehensive analysis:

* **Project Overview:** A landing page that explains the business context and displays the final performance metrics (Accuracy, Precision, and Recall) of the predictive model.
* **Exploratory Data Analysis (EDA):** A page dedicated to visualizing the relationships between customer attributes (like contract type and tech support) and churn rates.
* **Churn Prediction Tool:** An interactive tool where you can select an existing customer ID and get a real-time prediction of their churn probability.
* **Power BI Dashboard:** A demonstration of a business intelligence dashboard built in Power BI to provide a high-level overview of churn drivers and key performance indicators (KPIs).

![Power BI Dashboard Demo](https://github.com/Yethirajyam-Sai-Aditya-Maanas/customer-churn-portfolio/blob/main/powerbi-demo.gif?raw=true)


---

## 🛠️ Tech Stack

This project was built using a combination of data science and web development tools:

* **Data Analysis & Modeling:** Python, Pandas, Scikit-learn
* **Business Intelligence & Visualization:** Power BI, Matplotlib, Seaborn
* **Web Application Framework:** Streamlit
* **Deployment:** Streamlit Community Cloud & GitHub

---

## ⚙️ Setup and Installation

To run this application locally, please follow these steps:

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Yethirajyam-Sai-Aditya-Maanas/customer-churn-portfolio.git](https://github.com/Yethirajyam-Sai-Aditya-Maanas/customer-churn-portfolio.git)
    cd customer-churn-portfolio
    ```

2.  **Install the required libraries:**
    ```sh
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```
