import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration & Styling ---
st.set_page_config(page_title="Coral Reef Regime Predictor", page_icon="🌊", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f0f8ff;}
    h1 {color: #005f73;}
    .stButton>button {background-color: #0a9396; color: white; border-radius: 5px; height: 50px; font-size: 18px;}
    .stButton>button:hover {background-color: #005f73; color: white;}
    .prediction-box {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center;}
    </style>
""", unsafe_allow_html=True)

# --- Title and Description ---
st.title("🌊 CORAL REEF REGIME PREDICTOR")
st.markdown("Predict the coral reef regime based on 20 anthropogenic and biophysical/environmental features.")

# --- Load Models & Features ---
@st.cache_resource
def load_models():
    model_path = os.path.join("models", "best_model.pkl")
    features_path = os.path.join("models", "features.pkl")
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

try:
    model, features_list = load_models()
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'best_model.pkl' is saved in the models directory by running `python train_and_save.py`.")
    st.stop()

# --- Input Features ---
st.header("Enter Reef Parameters")

# Group features logically for UI
human_pressures = [
    'Effluent', 'Sedimentation', 'New_Development', 'Habitat_Modification', 
    'Fishing_Comm_Total', 'Fishing_NonComm_Boat_Total', 'Fishing_NonComm_Shore_Line', 
    'Fishing_NonComm_Shore_Net', 'Fishing_NonComm_Shore_Spear'
]

environmental = [
    'Invasive_Algae', 'SST_CLIM_M', 'SST_STD', 'CHL_CLIM_M', 'CHL_ANOM_F', 
    'PAR_CLIM_M', 'PAR_STD', 'WAV_CLIM_M', 'WAV_ANOM_F', 'Complexity', 'Depth'
]

col1, col2 = st.columns(2)
inputs = {}

with col1:
    st.subheader("Human Pressures")
    for feat in human_pressures:
        inputs[feat] = st.number_input(feat.replace("_", " "), value=0.0, format="%.4f")

with col2:
    st.subheader("Environmental & Biophysical")
    for feat in environmental:
        inputs[feat] = st.number_input(feat.replace("_", " "), value=0.0, format="%.4f")

st.markdown("<br>", unsafe_allow_html=True)

# --- Prediction Logic ---
if st.button("PREDICT REGIME", use_container_width=True):
    # Calculate engineered features dynamically
    inputs['sqrt_power_x_compexity'] = np.sqrt(max(0, inputs['WAV_CLIM_M'] * inputs['Complexity']))
    inputs['log_power_over_depth'] = np.log1p(max(0, inputs['WAV_CLIM_M'] / (inputs['Depth'] + 1e-5)))
    inputs['complexity_over_depth'] = inputs['Complexity'] / (inputs['Depth'] + 1e-5)
    inputs['Irradiance_x_inv_algae'] = inputs['PAR_CLIM_M'] * inputs['Invasive_Algae']
    inputs['both_anomolies'] = inputs['CHL_ANOM_F'] * inputs['WAV_ANOM_F']

    # Convert to dataframe in the exact order of features
    input_data = pd.DataFrame([inputs])[features_list]
    
    with st.spinner("Analyzing parameters..."):
        try:
            # Make Prediction
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence = np.max(probabilities) * 100
            
            st.divider()
            
            # Layout for results
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style='color: #005f73;'>Predicted Regime</h2>
                    <h1 style='font-size: 48px; margin: 10px 0; color: #111;'>REGIME {prediction}</h1>
                    <h3 style='color: #0a9396;'>Confidence: {confidence:.1f}%</h3>
                    <p style='color: gray; margin-top: 15px;'>Model Used: Random Forest</p>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown("### Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Regime': [f'Regime {i}' for i in model.classes_],
                    'Probability': probabilities
                })
                st.bar_chart(prob_df.set_index('Regime'))
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
