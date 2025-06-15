# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Define the path to the model and scaler
FILE_PATH = "C:/MySkill/Data Science/Projects/ann-dl-project/model"

# Define list of pkl files to load
pkl_files = ["label_encoder_gender.pkl", "ohe_geography.pkl", "scaler.pkl"]

# Load the model
model = tf.keras.models.load_model(f"{FILE_PATH}/model.h5")

# Iterate through the pkl files and load the pkl files
for pkl in pkl_files:
  with open(f"{FILE_PATH}/{pkl}", "rb") as file:
    if pkl == "label_encoder_gender.pkl":
      label_encoder_gender = pickle.load(file=file)
    elif pkl == "ohe_geography.pkl":
      ohe_geography = pickle.load(file=file)
    else:
      scaler = pickle.load(file=file)

# Streamlit app configuration
st.title("Customer Churn Prediction")

# Create user input form
geography = st.selectbox("Geography", ohe_geography.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
  "CreditScore": [credit_score],
  "Gender": [label_encoder_gender.transform([gender])[0]],
  "Age": [age],
  "Tenure": [tenure],
  "Balance": [balance],
  "NumOfProducts": [num_of_products],
  "HasCrCard": [has_credit_card],
  "IsActiveMember": [is_active_member],
  "EstimatedSalary": [estimated_salary],
})

# One-hot encode the geography
geo_encoded = ohe_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geography.get_feature_names_out(["Geography"]))

# Combine the input data with the one-hot encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scaled the input data
input_data_scaled = scaler.transform(input_data)

# Predict using the model
predict = model.predict(input_data_scaled)
prediction_probability = predict[0][0]

if prediction_probability > 0.5:
  st.write(f"The customer is likely to churn with probability: {prediction_probability:.2f}")
else:
  st.write(f"The customer is not likely to churn with probability: {1 - prediction_probability:.2f}")

