import streamlit as st
import joblib
import numpy as np

# ================================
# Load model, scaler, features
# ================================
model = joblib.load("parkinsons_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
feature_means = joblib.load("feature_means.pkl")

# ================================
# Streamlit UI
# ================================
st.title("🧠 Early Prediction of Parkinson’s Disease")
st.write(
    "This app predicts whether a patient is likely to have Parkinson’s Disease "
    "based on acoustic and biomedical features."
)

st.sidebar.header("Enter Patient Features")

# Sidebar inputs for all features
user_input = []
for feature in feature_names:
    value = st.sidebar.number_input(
        f"{feature}",
        value=float(round(feature_means[feature], 4)),  # default = mean
        format="%.4f"
    )
    user_input.append(value)

# Convert input to array
input_array = np.array(user_input).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# ================================
# Prediction
# ================================
prediction = model.predict(input_scaled)[0]
probability = (
    model.predict_proba(input_scaled)[0][1]
    if hasattr(model, "predict_proba")
    else None
)

st.subheader("Prediction Result")
if prediction == 1:
    st.error("⚠️ The model predicts this patient is likely to have **Parkinson’s Disease**.")
else:
    st.success("✅ The model predicts this patient is **Healthy**.")

if probability is not None:
    st.write(f"Prediction confidence (probability of PD): **{probability:.2f}**")
