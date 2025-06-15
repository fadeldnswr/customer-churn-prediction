# Customer Churn Prediction 🧠📉
A deep learning project that leverages Artificial Neural Networks (ANN) to predict customer churn in a telecom dataset. This project is deployed as an interactive web application using Streamlit, enabling users to easily input customer data and obtain churn predictions in real-time.

## Project Overview
Customer churn is a critical KPI in many industries, especially telecommunications. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project aims to build a robust predictive model using deep learning to anticipate customer churn based on behavioral and demographic features.

### Key Features
- Deep Learning Model (ANN) implemented with TensorFlow/Keras
- Preprocessing pipeline with label encoding, scaling, and balancing
- Streamlit Web App for real-time predictions
- User-friendly interface for manual input or batch prediction
- Model evaluation metrics (accuracy, confusion matrix, etc.)

## Dataset
- The dataset used in this project is a customer churn dataset sourced from a telecom company. It includes features such as:
- Demographics: Gender, Age, SeniorCitizen, etc.
- Services signed up: InternetService, OnlineSecurity, etc.
- Account information: Tenure, MonthlyCharges, TotalCharges
- Target variable: Churn (Yes/No)

## Model Architecture
The ANN model consists of:
- Input layer: Normalized input features
- Hidden layers: Multiple dense layers with ReLU activation
- Output layer: Sigmoid activation for binary classification

Compiled with:
- Loss function: binary_crossentropy
- Optimizer: adam
- Metrics: accuracy

## Deployment
The model is deployed using Streamlit, making it accessible via a simple web interface.
Try the web app locally: <pre> ```bash streamlit run app.py ``` </pre>

## Requirements
Install dependencies: <pre> ```pip install -r requirements.txt ``` </pre>