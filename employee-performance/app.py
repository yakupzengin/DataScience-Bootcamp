import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.title("Employee Performance Score Prediction")

model = joblib.load("notebooks/random_forest_model.pkl")
scaler = joblib.load("notebooks/scalerv1.pkl")

st.sidebar.header("Set Input Values")
def get_user_input():
    Years_At_Company = st.sidebar.number_input("Years at Company", min_value=0, max_value=50, value=3)
    Monthly_Salary = st.sidebar.number_input("Monthly Salary", min_value=0, max_value=100000, value=4500)
    Overtime_Hours = st.sidebar.number_input("Overtime Hours", min_value=0, max_value=100, value=10)
    Projects_Handled = st.sidebar.number_input("Projects Completed", min_value=0, max_value=50, value=50)
    Promotions = st.sidebar.number_input("Number of Promotions", min_value=0, max_value=10, value=1)
    Employee_Satisfaction_Score = st.sidebar.slider("Employee Satisfaction Score", min_value=0.0, max_value=5.0, value=2.0)

    data = {  
        "Years_At_Company": Years_At_Company,
        "Monthly_Salary": Monthly_Salary,
        "Overtime_Hours": Overtime_Hours,
        "Projects_Handled": Projects_Handled,
        "Promotions": Promotions,
        "Employee_Satisfaction_Score": Employee_Satisfaction_Score, 
    }
    return pd.DataFrame([data])


user_input = get_user_input()
st.subheader("Your Input Values")
st.write(user_input)

scaled_input = scaler.transform(user_input)

prediction = model.predict(scaled_input)
st.subheader("Predicted Performance Score")
st.write(prediction[0])
