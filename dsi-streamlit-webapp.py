
import streamlit as st
import pandas as pd
import joblib

# Load our model pipeline
model = joblib.load("model.joblib")

# add title and instructions
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and hit submit for likelihood for purchase")

# age input form
age = st.number_input(
    label= "01. Enter the customer's age",
    min_value= 18, max_value= 120,
    value = 35
)

# gender input form
gender = st.radio(
    label= "02. Enter the customer's gender",
    options= ["Male", "Female"]
)

# credit score input form
credit_score = st.number_input(
    label= "03. Enter the customer's credit score",
    min_value= 0, max_value= 1000,
    value = 500
)

# Submit input
if st.button("Submit for prediction"):
    data = pd.DataFrame(
        {"age" : [age], "gender" : [gender], "credit_score" : [credit_score]}
    )
    pred_prob = model.predict_proba(data)[0][1]
    st.subheader(f"Based on these attributes, our model predicts a purchase probability of {pred_prob:.0%}")
