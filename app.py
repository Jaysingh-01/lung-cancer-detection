import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Clinical Decision Support System", layout="centered")
st.title("🫁 Lung Cancer Risk Assessment Dashboard")
st.write("Input patient metrics below to calculate diagnostic model risk probabilities.")

model = joblib.load('lung_cancer_random_forest_model.pkl')
scaler = joblib.load('data_scaler_transform.pkl')

st.sidebar.header("Demographics & Lifestyle")
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"], index=0)
age = st.sidebar.slider("Age (Years)", min_value=18, max_value=90, value=50)
smoking = st.sidebar.selectbox("Smoking History", options=["No", "Yes"], index=0)
alcohol = st.sidebar.selectbox("Alcohol Consumption", options=["No", "Yes"], index=0)

st.header("Clinical Symptoms Profile")
col1, col2 = st.columns(2)

with col1:
    yellow_fingers = st.checkbox("Yellow Fingers")
    anxiety = st.checkbox("Anxiety Symptoms")
    peer_pressure = st.checkbox("Peer Pressure Factors")
    chronic_disease = st.checkbox("Chronic Disease History")
    fatigue = st.checkbox("Persistent Fatigue")

with col2:
    allergy = st.checkbox("Allergic History")
    wheezing = st.checkbox("Wheezing / Restricted Airways")
    coughing = st.checkbox("Persistent Coughing")
    dyspnea = st.checkbox("Shortness of Breath")
    dysphagia = st.checkbox("Swallowing Difficulty")
    chest_pain = st.checkbox("Thoracic Chest Pain")

def map_binary(val):
    return 2 if val or val == "Yes" else 1

gender_mapped = 1 if gender == "Male" else 0

input_data = pd.DataFrame([{
    'GENDER': gender_mapped,
    'AGE': age,
    'SMOKING': map_binary(smoking),
    'YELLOW_FINGERS': map_binary(yellow_fingers),
    'ANXIETY': map_binary(anxiety),
    'PEER_PRESSURE': map_binary(peer_pressure),
    'CHRONIC_DISEASE': map_binary(chronic_disease),
    'FATIGUE': map_binary(fatigue),
    'ALLERGY': map_binary(allergy),
    'WHEEZING': map_binary(wheezing),
    'ALCOHOL_CONSUMING': map_binary(alcohol),
    'COUGHING': map_binary(coughing),
    'SHORTNESS_OF_BREATH': map_binary(dyspnea),
    'SWALLOWING_DIFFICULTY': map_binary(dysphagia),
    'CHEST_PAIN': map_binary(chest_pain)
}])

if st.button("Run Diagnostic Analytics Pipeline", type="primary"):
    scaled_input = scaler.transform(input_data)
    risk_prob = model.predict_proba(scaled_input)[0][1] * 100

    st.markdown("---")
    st.subheader("Diagnostic Engine Outputs")

    if risk_prob >= 50:
        st.error(f"⚠️ High Risk Assessment Profile: **{risk_prob:.2f}% Probability**")
    else:
        st.success(f"✅ Normal Risk Assessment Profile: **{risk_prob:.2f}% Probability**")