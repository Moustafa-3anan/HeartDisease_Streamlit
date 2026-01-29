import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# 1️⃣ Load models
# -----------------------------
rf_model = joblib.load("heart_rf_model.pkl")
xgb_model = joblib.load("heart_xgb_model.pkl")
scaler = joblib.load("heart_scaler.pkl")

# -----------------------------
# 2️⃣ Streamlit Interface
# -----------------------------
st.title("Heart Disease Prediction App ❤️")
st.markdown("Enter patient data to predict the risk of heart disease (all inputs are numeric).")

# --- Inputs ---
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.number_input("Sex (Male=1 / Female=0)", min_value=0, max_value=1, value=1)
cp = st.number_input("Chest pain type (0-3)", min_value=0, max_value=3, value=1)
trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.number_input("Fasting blood sugar > 120 (0/1)", min_value=0, max_value=1, value=0)
restecg = st.number_input("Rest ECG (0-2)", min_value=0, max_value=2, value=0)
thalach = st.number_input("Max heart rate achieved", min_value=50, max_value=250, value=150)
exang = st.number_input("Exercise induced angina (0/1)", min_value=0, max_value=1, value=0)
oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.number_input("Slope of ST segment (0-2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of major vessels colored (0-3)", min_value=0, max_value=3, value=0)
thal = st.number_input("Thal (0=normal, 1=fixed defect, 2=reversible defect)", min_value=0, max_value=2, value=0)

# -----------------------------
# 3️⃣ Prepare data
# -----------------------------
user_df = pd.DataFrame([{
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

# Scale inputs
user_scaled = scaler.transform(user_df)

# -----------------------------
# 4️⃣ Prediction (Risk-based)
# -----------------------------
def get_risk_label(prob):
    if prob < 0.4:
        return "Low risk"
    elif prob < 0.7:
        return "Medium risk"
    else:
        return "High risk"

rf_prob = rf_model.predict_proba(user_scaled)[0][1]
rf_label = get_risk_label(rf_prob)

xgb_prob = xgb_model.predict_proba(user_scaled)[0][1]
xgb_label = get_risk_label(xgb_prob)

# -----------------------------
# 5️⃣ Display results
# -----------------------------
st.subheader("Random Forest Prediction")
st.write(f"Risk: {rf_label} ({rf_prob * 100:.2f}%)")

st.subheader("XGBoost Prediction")
st.write(f"Risk: {xgb_label} ({xgb_prob * 100:.2f}%)")
