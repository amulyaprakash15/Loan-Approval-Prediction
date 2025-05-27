# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# Load the model
model = joblib.load("model.pkl")

st.title("🏦 Loan Approval Prediction System")
st.markdown("### Fill out the applicant details to check loan approval status")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Term (in months)", [360, 180, 240, 120, 300])
credit_history = st.selectbox("Credit History", ["Good", "Bad"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encoding user input
def encode_input():
    return pd.DataFrame([{
        'Gender': 1 if gender == "Male" else 0,
        'Married': 1 if married == "Yes" else 0,
        'Dependents': 3 if dependents == "3+" else int(dependents),
        'Education': 0 if education == "Graduate" else 1,
        'Self_Employed': 1 if self_employed == "Yes" else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': 1 if credit_history == "Good" else 0,
        'Property_Area': {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    }])

# Predict
if st.button("Check Loan Status"):
    user_input = encode_input()
    prediction = model.predict(user_input)
    result = "✅ Approved" if prediction[0] == 1 else "❌ Rejected"
    st.markdown(f"### 🎯 Prediction Result: **{result}**")
