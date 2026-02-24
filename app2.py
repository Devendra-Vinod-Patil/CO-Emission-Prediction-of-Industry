# ==============================
# CO2 Emission Prediction App
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# Load Trained Model
# ------------------------------
model = joblib.load("xgboost_emission_model.joblib")

st.set_page_config(page_title="CO2 Emission Predictor")

st.title("🌍 CO₂ Emission Prediction of Industry")
st.write("Enter industrial parameters to predict CO₂ emissions.")

# ------------------------------
# User Inputs
# ------------------------------

industry_sector = st.selectbox(
    "Industry Sector",
    ["Cement", "Steel", "Chemicals", "Power", "Oil & Gas", "Fertilizer"]
)

energy_source = st.selectbox(
    "Energy Source",
    ["Coal", "Natural Gas", "Electricity", "Biomass", "Oil"]
)

location_region = st.selectbox(
    "Location Region",
    ["India-West", "India-North", "India-South", "India-East", "India-Central"]
)

process_type = st.selectbox(
    "Process Type",
    ["Combustion", "Chemical Reaction", "Mixed Process"]
)

production_volume = st.number_input(
    "Production Volume (tons)", min_value=1000.0, value=50000.0
)

fuel_consumption = st.number_input(
    "Fuel Consumption (GJ)", min_value=1000.0, value=200000.0
)

emission_factor = st.number_input(
    "Emission Factor (kgCO2/GJ)", min_value=10.0, value=75.0
)

efficiency = st.number_input(
    "Efficiency (GJ per ton)", min_value=0.5, value=4.5
)

year = st.slider("Year", 2015, 2025, 2022)

# ------------------------------
# Prediction Button
# ------------------------------

if st.button("Predict CO₂ Emission"):

    # Create dataframe
    input_data = pd.DataFrame({
        "Industry_Sector": [industry_sector],
        "Production_Volume_tons": [production_volume],
        "Energy_Source": [energy_source],
        "Fuel_Consumption_GJ": [fuel_consumption],
        "Emission_Factor_kgCO2_per_GJ": [emission_factor],
        "Location_Region": [location_region],
        "Year": [year],
        "Process_Type": [process_type],
        "Efficiency_GJ_per_ton": [efficiency]
    })

    # ------------------------------
    # Feature Engineering (same as training)
    # ------------------------------

    input_data['Fuel_Emission_Interaction'] = (
        input_data['Fuel_Consumption_GJ'] *
        input_data['Emission_Factor_kgCO2_per_GJ']
    )

    input_data['Energy_per_Production'] = (
        input_data['Fuel_Consumption_GJ'] /
        (input_data['Production_Volume_tons'] + 1)
    )

    input_data['Log_Fuel_Consumption'] = np.log1p(
        input_data['Fuel_Consumption_GJ']
    )

    input_data['Log_Production_Volume'] = np.log1p(
        input_data['Production_Volume_tons']
    )

    input_data['Efficiency_squared'] = (
        input_data['Efficiency_GJ_per_ton'] ** 2
    )

    # ------------------------------
    # One-hot encoding
    # ------------------------------
    input_data = pd.get_dummies(input_data)

    # Align columns with training model
    model_columns = model.feature_names_in_

    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # ------------------------------
    # Prediction
    # ------------------------------
    prediction = model.predict(input_data)[0]

    # ------------------------------
    # Output
    # ------------------------------
    st.success(f"✅ Predicted CO₂ Emission: {prediction:.6f} MtCO₂")
