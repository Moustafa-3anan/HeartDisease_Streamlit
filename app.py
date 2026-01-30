import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Load everything (model, scaler, features)
data = joblib.load("heart_rf_model_streamlit.pkl")
model = data["model"]
scaler = data["scaler"]
features = data["features"]

st.title("❤️ Heart Disease Prediction App")
st.write("🏥 Enter patient medical details to predict heart disease risk")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex (Male=1, Female=0)", [0,1], index=1)
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
trestbps = st.number_input("Resting BP", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1], index=0)
restecg = st.number_input("Rest ECG (0-2)", min_value=0, max_value=2, value=0)
thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", [0,1], index=0)
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
slope = st.number_input("Slope (0-2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Vessels (0-3)", min_value=0, max_value=3, value=0)
thal = st.selectbox("Thal", ["normal","fixed","reversible"], index=0)

if st.button("🔮 Predict Heart Disease"):

    # Prepare user input
    user_input = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # Apply get_dummies to match training
    user_encoded = pd.get_dummies(user_input)
    user_encoded = user_encoded.reindex(columns=features, fill_value=0)

    # Scale numerical features
    user_scaled = scaler.transform(user_encoded)

    # Predict
    prediction = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0][1] * 100

    if prediction == 1:
        st.error(f"💔 High Risk of Heart Disease ({prob:.2f}%)")
    else:
        st.success(f"❤️ Low Risk of Heart Disease ({prob:.2f}%)")

