import streamlit as st
import io
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ================================
# Load model, scaler, features
# ================================
model = joblib.load("parkinsons_rf_model.pkl")  # ✅ Match your saved model name
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")  # ✅ Load PCA too
feature_names = joblib.load("feature_names.pkl")
feature_means = joblib.load("feature_means.pkl")

# Load test data (optional, for future dashboard analytics)
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Parkinson’s Prediction", page_icon="🧠", layout="wide")

st.title("🧠 Early Prediction of Parkinson’s Disease")
st.write("This app predicts the likelihood of Parkinson’s Disease using acoustic and biomedical voice features.")

st.sidebar.header("Enter Patient Features")

user_input = []
for feature in feature_names:
    if feature.lower() == "gender":
        gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
        value = 1 if gender == "Male" else 0
    else:
        default_val = float(round(feature_means[feature], 4))
        value = st.sidebar.number_input(f"{feature}", value=default_val, format="%.4f")
    user_input.append(value)

# Convert input to scaled PCA features
input_array = np.array(user_input).reshape(1, -1)
input_scaled = scaler.transform(input_array)
input_pca = pca.transform(input_scaled)

# ================================
# Prediction
# ================================
prediction = model.predict(input_pca)[0]
probability = model.predict_proba(input_pca)[0][1] if hasattr(model, "predict_proba") else None

st.subheader("Prediction Result")
if prediction == 1:
    st.error("⚠️ **Likely Parkinson’s Disease**")
else:
    st.success("✅ **Healthy (No Parkinson’s detected)**")

if probability is not None:
    st.metric("Prediction Confidence", f"{probability*100:.2f}%")

# ================================
# 📂 CSV Upload for Batch Prediction
# ================================
st.markdown("---")
st.subheader("📂 Upload a CSV File for Batch Prediction")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", uploaded_df.head())

    # Handle Gender column if present
    if "Gender" in uploaded_df.columns:
        uploaded_df["Gender"] = uploaded_df["Gender"].map({"Male": 1, "Female": 0})

    # ✅ NEW CHECK: Ensure correct columns and no missing values
    missing_cols = [col for col in feature_names if col not in uploaded_df.columns]
    extra_cols = [col for col in uploaded_df.columns if col not in feature_names]

    if missing_cols:
        st.error(f"❌ The uploaded file is missing these required features: {missing_cols}")
    elif extra_cols:
        st.error(f"⚠️ The uploaded file has extra/unexpected columns: {extra_cols}")
    elif uploaded_df[feature_names].isnull().values.any():
        st.error("❌ The uploaded file contains missing (NaN) values. Please clean your data.")
    else:
        try:
            uploaded_scaled = scaler.transform(uploaded_df[feature_names])
            uploaded_pca = pca.transform(uploaded_scaled)
            uploaded_preds = model.predict(uploaded_pca)
            uploaded_probs = model.predict_proba(uploaded_pca)[:, 1] if hasattr(model, "predict_proba") else None

            uploaded_df["Prediction"] = ["Parkinson's" if p == 1 else "Healthy" for p in uploaded_preds]
            if uploaded_probs is not None:
                uploaded_df["Confidence"] = (uploaded_probs * 100).round(2)

            st.success("✅ Predictions generated successfully!")
            st.write(uploaded_df.head())

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ================================
# 📥 Download Results
# ================================
st.markdown("---")
st.subheader("📥 Download Results")

download_mode = st.radio(
    "Choose download mode:",
    ["Batch upload results", "Single patient input"],
)

file_format = st.selectbox("Choose file format:", ["CSV", "Excel (.xlsx)"])

if download_mode == "Batch upload results":
    if "uploaded_df" in locals():
        df_to_download = uploaded_df.copy()
        if file_format == "CSV":
            csv = df_to_download.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV Results",
                data=csv,
                file_name="parkinsons_predictions.csv",
                mime="text/csv",
            )
        else:
            output = io.BytesIO()
            df_to_download.to_excel(output, index=False, engine="xlsxwriter")
            st.download_button(
                label="📥 Download Excel Results",
                data=output.getvalue(),
                file_name="parkinsons_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info("Please upload a CSV file first.")
else:
    if "prediction" in locals():
        single_data = pd.DataFrame([user_input], columns=feature_names)
        single_data["Prediction"] = ["Parkinson's" if prediction == 1 else "Healthy"]
        if file_format == "CSV":
            csv = single_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Single Prediction (CSV)",
                data=csv,
                file_name="single_patient_prediction.csv",
                mime="text/csv",
            )
        else:
            output = io.BytesIO()
            single_data.to_excel(output, index=False, engine="xlsxwriter")
            st.download_button(
                label="📥 Download Single Prediction (Excel)",
                data=output.getvalue(),
                file_name="single_patient_prediction.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info("Run a single prediction first to enable download.")
