import streamlit as st
import requests
import joblib
from PIL import Image

# Add some information about the service
st.title("Credit Scoring")
st.subheader("Enter variabel below then click Predict button")

# Create form of input
with st.form(key="data_form"):
    # Create box for number input
    unsecured_line = st.slider(
        label = "Enter Unsecured Line Value:",
        min_value = 0,
        max_value = 100,
        help = "Value range from 0 to 100"
    )

    debt_ratio = st.slider(
        label = "Enter Debt Ratio % Value:",
        min_value = 0,
        max_value = 100,
        help = "Value range from 0 to 100"
    )
    
    monthly_income = st.slider(
        label = "Enter Monthly Income Value:",
        min_value = 0,
        max_value = 100000,
        help = "Value range from 0 to 100.000"
    )

    age = st.slider(
        label = "Enter Age Value:",
        min_value = 0,
        max_value = 109,
        help = "Value range from 0 to 109"
    )

    no_59_past_worse = st.slider(
        label = "Enter Number Of Time 30-59 Days Past Due Not Worse Value:",
        min_value = 0,
        max_value = 59,
        help = "Value range from 0 to 59"
    )

    open_credit = st.slider(
        label = "Enter Open Credit Value:",
        min_value = 0,
        max_value = 58,
        help = "Value range from 0 to 58"
    )

    real_estate_loan = st.slider(
        label = "Enter Real Estate Loan Value:",
        min_value = 0,
        max_value = 10,
        help = "Value range from 0 to 10"
    )

    dependents = st.slider(
        label = "Enter Dependents Value:",
        min_value = 0,
        max_value = 5,
        help = "Value range from 0 to 5"
    )
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "RevolvingUtilizationOfUnsecuredLines": unsecured_line,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "Age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": no_59_past_worse,
            "NumberOfOpenCreditLinesAndLoans": open_credit,
            "NumberRealEstateLoansOrLines": real_estate_loan,
            "NumberOfDependents": dependents
        }

        
        
        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://api_backend:8080/predict", json = raw_data).json()
            
        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "No Default Potential":
                st.warning("Potential to Default")
            else:
                st.success("No Default Potential")