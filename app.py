#import streamlit as st
#import numpy as np
#from sklearn.preprocessing import StandardScaler
#import joblib
#import pandas as pd

import joblib
import streamlit as st
import pandas as pd

# Load model pipeline (includes scaler & encoders)
model_pipeline = joblib.load("heart_rf_model_streamlit.pkl")  # pipeline شامل preprocessing

st.title("❤️ Heart Disease Prediction App")

# Input fields
age = st.number_input("Age", 1, 120, 40)
sex = st.selectbox("Sex (1=Male, 0=Female)", [0,1], index=1)
cp = st.selectbox("Chest Pain Type", ["typical angina","atypical angina","non-anginal","asymptomatic"])
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
restecg = st.selectbox("Rest ECG", ["normal","ST-T abnormality","left ventricular hypertrophy"])
thalach = st.number_input("Max Heart Rate", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 0.0, 0.1)
slope = st.selectbox("Slope", ["upsloping","flat","downsloping"])
ca = st.number_input("Number of Vessels", 0, 3, 0)
thal = st.selectbox("Thal", ["normal","fixed defect","reversable defect"])
dataset = st.selectbox("Dataset", ["Hungary","Switzerland","Cleveland","Long Beach VA"])

# Convert to DataFrame
input_data = pd.DataFrame([{
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
    "thal": thal,
    "dataset": dataset
}])

if st.button("Predict Heart Disease"):
    try:
        prediction = model_pipeline.predict(input_data)[0]
        prob = model_pipeline.predict_proba(input_data)[0][1]*100
        if prediction==1:
            st.error(f"💔 High Risk ({prob:.2f}%)")
        else:
            st.success(f"❤️ Low Risk ({prob:.2f}%)")
    except Exception as e:
        st.error(f"Error: {e}")
