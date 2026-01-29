import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Load saved model and scaler
model = joblib.load("heart_rf_model_streamlit.pkl")
scaler = joblib.load("heart_scaler.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("🏥 Enter patient medical details to predict heart disease risk")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex (1=Male, 0=Female)", [0,1], index=1)
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
thal = st.number_input("Thal (1=normal)", min_value=0, max_value=3, value=1)

# Feature names used during training
feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal']

if st.button("🔮 Predict Heart Disease"):
    # تحويل القيم لـ DataFrame بنفس ترتيب الـ features
    input_dict = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    input_df = pd.DataFrame([input_dict], columns=feature_names)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        st.error(f"💔 High Risk of Heart Disease ({prob:.2f}%)")
    else:
        st.success(f"❤️ Low Risk of Heart Disease ({prob:.2f}%)")


