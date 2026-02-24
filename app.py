# ==============================
# CO2 Emission Prediction App (Improved)
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model + training columns
model = joblib.load("xgboost_emission_model.joblib")
model_columns = joblib.load("model_columns.joblib")

st.set_page_config(page_title="CO2 Emission Predictor")

st.title("🌍 CO₂ Emission Prediction of Industry")

# ------------------------------
# Inputs
# ------------------------------

industry_sector = st.selectbox(
    "Industry Sector",
    ["Cement","Steel","Chemicals","Power","Oil & Gas","Fertilizer"]
)

energy_source = st.selectbox(
    "Energy Source",
    ["Coal","Natural Gas","Electricity","Biomass","Oil"]
)

location_region = st.selectbox(
    "Location Region",
    ["India-West","India-North","India-South","India-East","India-Central"]
)

process_type = st.selectbox(
    "Process Type",
    ["Combustion","Chemical Reaction","Mixed Process"]
)

production_volume = st.number_input(
    "Production Volume (tons)",1000.0,500000.0,50000.0
)

fuel_consumption = st.number_input(
    "Fuel Consumption (GJ)",1000.0,800000.0,200000.0
)

emission_factor = st.number_input(
    "Emission Factor (kgCO2/GJ)",10.0,120.0,75.0
)

efficiency = st.number_input(
    "Efficiency (GJ per ton)",0.5,10.0,4.5
)

year = st.slider("Year",2015,2025,2022)

# ------------------------------
# Prediction
# ------------------------------

if st.button("Predict CO₂ Emission"):

    input_data = pd.DataFrame({
        "Industry_Sector":[industry_sector],
        "Production_Volume_tons":[production_volume],
        "Energy_Source":[energy_source],
        "Fuel_Consumption_GJ":[fuel_consumption],
        "Emission_Factor_kgCO2_per_GJ":[emission_factor],
        "Location_Region":[location_region],
        "Year":[year],
        "Process_Type":[process_type],
        "Efficiency_GJ_per_ton":[efficiency]
    })

    # -------- Improved Feature Engineering --------

    # weaker interaction (avoid dominance)
    input_data['Fuel_Emission_Ratio'] = (
        input_data['Fuel_Consumption_GJ'] /
        (input_data['Emission_Factor_kgCO2_per_GJ'] + 1)
    )

    input_data['Energy_per_Production'] = (
        input_data['Fuel_Consumption_GJ'] /
        (input_data['Production_Volume_tons'] + 1)
    )

    input_data['Log_Fuel'] = np.log1p(input_data['Fuel_Consumption_GJ'])
    input_data['Log_Production'] = np.log1p(input_data['Production_Volume_tons'])

    input_data['Efficiency_squared'] = (
        input_data['Efficiency_GJ_per_ton'] ** 2
    )

    # -------- Encoding --------
    input_data = pd.get_dummies(input_data)

    # align columns EXACTLY like training
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[model_columns]

    # -------- Prediction --------
    prediction = model.predict(input_data)[0]

    st.success(f"✅ Predicted CO₂ Emission: {prediction:.6f} MtCO₂")
