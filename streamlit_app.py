import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle

from sklearn.preprocessing import StandardScaler

ridge_regressor = pickle.load(open("models/ridge.pkl", "rb"))
elasticnet_regressor = pickle.load(open("models/elasticnet.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.title("Streamlit fire weather index prediction")

st.badge("Never before")
st.subheader("Enter values to predict Fire Weather Index (FWI):", )

temp = st.number_input("Temperature (Celsius)", min_value=0)
RH = st.number_input("Relative Humidity (%)", min_value=0)
Ws = st.number_input("Wind Speed (km/h)", min_value=0)
Rain = st.number_input("Rain (mm / sq. m)", min_value=0)
FFMC = st.number_input("FFMC index", min_value=0)
DMC = st.number_input("DMC index", min_value=0)
ISI = st.number_input("ISI index", min_value=0)
Classes = st.number_input("Classes", min_value=0, max_value=1)
Region = st.number_input("Region", min_value=0, max_value=1)

X = [[temp, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]

X_scaled = scaler.transform(X)
result = elasticnet_regressor.predict(X_scaled)

st.write(f"Result: {np.maximum(result[0], 0)}")
