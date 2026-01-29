import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Load saved RandomForest model, scaler, and features
# ----------------------------
with open("heart_rf_model.pkl", "rb") as file:
    saved_data = pickle.load(file)
    model = saved_data["model"]      # RandomForest model
    scaler = saved_data["scaler"]    # StandardScaler
    features = saved_data["features"]  # Feature names

# ----------------------------
# Streamlit App UI
# ----------------------------
st.title("🫀Heart Disease Prediction App")
st.write("🏥 Enter patient medical details to predict heart disease risk")

# Input fields dictionary
input_data = {}
input_data['age'] = st.number_input("Age", min_value=1, max_value=120, value=40)
input_data['sex'] = st.selectbox("Sex (1=Male, 0=Female)", [0,1], index=1)
input_data['cp'] = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
input_data['trestbps'] = st.number_input("Resting BP", min_value=80, max_value=200, value=120)
input_data['chol'] = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
input_data['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1], index=0)
input_data['restecg'] = st.number_input("Rest ECG (0-2)", min_value=0, max_value=2, value=0)
input_data['thalach'] = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=150)
input_data['exang'] = st.selectbox("Exercise Induced Angina", [0,1], index=0)
input_data['oldpeak'] = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
input_data['slope'] = st.number_input("Slope (0-2)", min_value=0, max_value=2, value=1)
input_data['ca'] = st.number_input("Number of Vessels (0-3)", min_value=0, max_value=3, value=0)
input_data['thal'] = st.number_input("Thal (1=normal)", min_value=0, max_value=3, value=1)

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("🔮 Predict Heart Disease"):

    # Arrange input in same order as features
    input_array = np.array([[input_data[feat] for feat in features]])

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability / Risk
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[0][1] * 100  # Classifier probability
    else:
        prob = prediction * 100  # Regressor output (continuous risk score)

    # Display Result
    if prediction == 1:
        st.error(f"💔 High Risk of Heart Disease ({prob:.2f}%)")
    else:
        st.success(f"❤️ Low Risk of Heart Disease ({prob:.2f}%)")
