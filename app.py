import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -----------------------------
# 1️⃣ Load models and columns
# -----------------------------
with open("Random Forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("XGBoost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("rf_columns.pkl", "rb") as f:
    rf_columns = pickle.load(f)

# -----------------------------
# 2️⃣ Streamlit Interface
# -----------------------------
st.title("Heart Disease Prediction App ❤️")
st.markdown("Enter patient data to predict the risk of heart disease (all inputs are numeric):")

# --- Inputs with explanations ---
st.markdown("**Sex:** Male = 1, Female = 0")
sex = st.number_input("Sex", min_value=0, max_value=1, value=1)

st.markdown("**Chest pain type (cp):** 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic")
cp = st.number_input("Chest pain type", min_value=0, max_value=3, value=1)

st.markdown("**Fasting blood sugar > 120 (fbs):** 1 = True, 0 = False")
fbs = st.number_input("Fasting blood sugar > 120", min_value=0, max_value=1, value=0)

st.markdown("**Rest ECG (restecg):** 0 = normal, 1 = ST-T abnormality, 2 = left ventricular hypertrophy")
restecg = st.number_input("Rest ECG", min_value=0, max_value=2, value=0)

st.markdown("**Exercise induced angina (exang):** 1 = Yes, 0 = No")
exang = st.number_input("Exercise induced angina", min_value=0, max_value=1, value=0)

st.markdown("**Thal:** 0 = normal, 1 = fixed defect, 2 = reversible defect")
thal = st.number_input("Thal", min_value=0, max_value=2, value=0)

# Continuous inputs
age = st.number_input("Age", min_value=1, max_value=120, value=50)
trestbps = st.number_input("Resting blood pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
thalach = st.number_input("Max heart rate achieved", min_value=50, max_value=250, value=150)
oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.number_input("Slope of ST segment (0-2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of vessels colored (0-3)", min_value=0, max_value=3, value=0)

# -----------------------------
# 3️⃣ Prepare data for RF
# -----------------------------
user_df = pd.DataFrame([{
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
}])

# one-hot encoding for RF
user_encoded = pd.get_dummies(user_df)
user_encoded = user_encoded.reindex(columns=rf_columns, fill_value=0)

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

# Random Forest
rf_prob = rf_model.predict_proba(user_encoded)[0][1]
rf_label = get_risk_label(rf_prob)

# XGBoost
xgb_prob = xgb_model.predict_proba(user_encoded)[0][1]
xgb_label = get_risk_label(xgb_prob)

# -----------------------------
# 5️⃣ Display results
# -----------------------------
st.subheader("Random Forest Prediction")
st.write(f"Risk: {rf_label} ({rf_prob*100:.2f}%)")

st.subheader("XGBoost Prediction")
st.write(f"Risk: {xgb_label} ({xgb_prob*100:.2f}%)")
