import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('churn_model.pkl')

st.title("Customer Churn Predictor")
st.write("Enter customer details to predict if they will churn.")

# Input fields
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])

if st.button("Predict"):
    # Build a single-row dataframe matching training format
    input_dict = {
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender_Male': 1 if gender == "Male" else 0,
        'Partner_Yes': 1 if partner == "Yes" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    # Align with model's expected columns, fill missing with 0
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN (probability: {probability:.2%})")
    else:
        st.success(f"✅ This customer is likely to STAY (probability of churn: {probability:.2%})")