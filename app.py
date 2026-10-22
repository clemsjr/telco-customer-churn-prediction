import streamlit as st
import pandas as pd
import joblib 

#page config
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="==",
    layout="wide"
    )

#Load trained pipeline
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

st.title("Telco Customer Churn Prediction")
st.write(
    "Enter a customer's information to estimate "
    "their likelihood of leaving the telecom company."
    )

st.info(
    "This tool provides a machine learning prediction, "
    "not a guarantee of future customer behaviour."
)

with st.form("Customer_form"):
    st.subheader("Customer Details")

    col1,col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0,1])
        partner = st.selectbox("Has Partner", ["Yes","No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure(months)", min_value=0.0, max_value=84.0, value=12.0, step=1.0
        )
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fibre optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0,
            step=10.0
        )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(customer)[0]
    if prediction == "Yes":
        st.warning("Prediction: Customer is likely to churn.")
    else:
        st.success("Prediction: Customer is likely to stay.")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(customer)[0]
        classes = list(model.classes_)
        churn_probability = probabilities[classes.index("Yes")]
        st.metric("Estimated churn probability", f"{churn_probability:.1%}")
