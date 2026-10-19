import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Churn Predictor", page_icon="📞")
st.title("📞 Customer Churn Predictor")
st.caption("هل العميل ده هيسيب الشركة؟")

if not (MODELS_DIR / "churn_model.joblib").exists():
    st.error("مفيش موديل متسيف. شغّل الأول:  python train.py")
    st.stop()

model = joblib.load(MODELS_DIR / "churn_model.joblib")
best_name = (MODELS_DIR / "best_model_name.txt").read_text()

tab1, tab2 = st.tabs(["Predict", "Models comparison"])

# ---------------- Tab 1: prediction form ----------------
with tab1:
    st.write(f"Model in use: **{best_name}**")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    with col2:
        security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])

    monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total charges", 0.0, 10000.0, float(monthly * tenure))

    if st.button("Predict"):
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }])

        prob = model.predict_proba(row)[0][1]
        st.metric("Churn probability", f"{prob:.1%}")
        st.progress(float(prob))

        if prob >= 0.5:
            st.error("⚠️ العميل ده غالبًا هيسيب — كلمه وادّيله عرض!")
        else:
            st.success("✅ العميل ده غالبًا هيكمّل معانا.")

# ---------------- Tab 2: comparison ----------------
with tab2:
    results = pd.read_csv(MODELS_DIR / "results.csv")
    st.dataframe(results.round(3), use_container_width=True, hide_index=True)
    st.bar_chart(results.set_index("Model")[["F1", "ROC-AUC"]])
