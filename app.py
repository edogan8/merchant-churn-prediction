import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the trained model and scaler
model = joblib.load('merchant_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Page Title & Description
st.set_page_config(page_title = 'Merchant Churn Predictor', page_icon = '🛡️')
st.title('Merchant Churn Prediction System 🛡️')
st.markdown("""
This application predicts the likelihood of a merchant churning based on various input features.
**Machine Learning Model:** XGBoost Classifier
""")

# 3. Sidebar for User Inputs
st.sidebar.header('Input Features')

def user_input_features():
    tenure = st.sidebar.slider('Tenure (months)', 0, 60, 12)
    complain = st.sidebar.selectbox('Has any complain in last month?', (1, 0))
    days_since_last_order = st.sidebar.number_input('Days since last order', min_value = 0, value = 2)
    cashback = st.sidebar.number_input('Cashback amount (Avg)', min_value = 0, value = 150)

    st.sidebar.markdown("---") # Separator

    order_cat = st.sidebar.selectbox(
        "Preferred Order Category",
        ("Laptop & Accessory", "Mobile Phone", "Mobile", "Grocery", "Others")
    )

    marital = st.sidebar.selectbox(
        "Marital Status",
        ("Married", "Single", "Divorced")
    )

    # Let's fill in the other features that the model expects with default (average) values.
    # In real life, you'd need to ask about all the features, but for the demo, we got the critical ones.
    # This is a representative data frame.
    data = {
            'Tenure': tenure,
            'Complain': complain,
            'DaySinceLastOrder': days_since_last_order,
            'CashbackAmount': cashback,
            
            # --- MAPPING ---
            # Change categorical inputs to one-hot encoding
            
            # Category Mapping
            'PreferedOrderCat_Laptop & Accessory': 1 if order_cat == "Laptop & Accessory" else 0,
            'PreferedOrderCat_Mobile Phone': 1 if order_cat == "Mobile Phone" else 0,
            'PreferedOrderCat_Mobile': 1 if order_cat == "Mobile" else 0,
            'PreferedOrderCat_Grocery': 1 if order_cat == "Grocery" else 0,
            'PreferedOrderCat_Others': 1 if order_cat == "Others" else 0,
            
            # Martial Status Mapping
            'MaritalStatus_Married': 1 if marital == "Married" else 0,
            'MaritalStatus_Single': 1 if marital == "Single" else 0,
            
            # --- DUMMY VALUES ---
            # Adding other features with average values
            'CityTier': 2,
            'WarehouseToHome': 15,
            'HourSpendOnApp': 3,
            'NumberOfDeviceRegistered': 4,
            'SatisfactionScore': 3,
            'NumberOfAddress': 3,
            'OrderAmountHikeFromlastYear': 15,
            'CouponUsed': 1,
            'OrderCount': 5,
            'Gender_Male': 1,
            # Other missing columns will be automatically filled in below.
        }
    return pd.DataFrame(data, index=[0])

# Let's get user input (Here's a simplified input).
# In a real project, all columns from 'X.columns' list would be created individually.
# For now, we're using a simple trick to avoid errors:
# We put what we get from the user and fill the rest with '0'.

# Model's expected column list (remembered from training)
expected_columns = [
    'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 
    'NumberOfDeviceRegistered', 'SatisfactionScore', 
    'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 
    'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 
    'CashbackAmount', 'PreferredLoginDevice_Mobile Phone', 
    'PreferredLoginDevice_Phone', 'PreferredPaymentMode_COD', 
    'PreferredPaymentMode_Cash on Delivery', 
    'PreferredPaymentMode_Credit Card', 
    'PreferredPaymentMode_Debit Card', 'PreferredPaymentMode_E wallet', 
    'PreferredPaymentMode_UPI', 'Gender_Male', 
    'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop & Accessory', 
    'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile Phone', 
    'PreferedOrderCat_Others', 'MaritalStatus_Married', 
    'MaritalStatus_Single']

# Let's get user input (Here's a simplified input).
input_df = user_input_features()

# Complete the missing columns.
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0 # Other properties should default to 0.

# Equalize column order
input_df = input_df[expected_columns]

# 4. Prediction Button
if st.button('Analyze (Calculate Risk)'):
    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] # Churn probability

    st.divider()
    
    if prediction[0] == 1:
        st.error(f"⚠️ WARNING! This merchant is very likely to leave.: %{probability*100:.1f}")
        st.write("Recommendation: Contact the Account Manager immediately.")
    else:
        st.success(f"✅ SAFE. The merchant is unlikely to leave.: %{probability*100:.1f}")