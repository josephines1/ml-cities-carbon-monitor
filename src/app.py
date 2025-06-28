import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Title
st.title("Carbon Emission Forecasting App")

# Load data
@st.cache_data
def load_data():
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'emissions.csv')
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# Sidebar
st.sidebar.header("User Input")
city_options = df['city'].unique()
sector_options = df['sector'].unique()
model_options = ['GRU', 'LSTM']

selected_city = st.sidebar.selectbox("Select City", city_options)
selected_sector = st.sidebar.selectbox("Select Sector", sector_options)
selected_model = st.sidebar.selectbox("Select Model", model_options)

# Filter data
df_filtered = df[(df['city'] == selected_city) & (df['sector'] == selected_sector)]


# Load scaler
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
scaler = joblib.load(SCALER_PATH)

# Load model
GRU_PATH = os.path.join(BASE_DIR, 'model', 'gru_model.sav')
LSTM_PATH = os.path.join(BASE_DIR, 'model', 'lstm_model.sav')
if selected_model == 'GRU':
    model = joblib.load(GRU_PATH)
else:
    model = joblib.load(LSTM_PATH)

# Preprocess: ambil value, scale, dan reshape untuk prediksi
values = df_filtered['value'].values.reshape(-1, 1)
scaled_values = scaler.transform(values)

# Sliding window
window_size = 30
X = []
for i in range(window_size, len(scaled_values)):
    X.append(scaled_values[i-window_size:i])
X = np.array(X)

# Predict
pred = model.predict(X)
pred_inverse = scaler.inverse_transform(pred)

# Prepare actual values (offset karena sliding window)
actual = values[window_size:]

# Plot
st.subheader(f"{selected_model} Prediction vs Actual for {selected_city} - {selected_sector}")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(actual, label='Actual')
ax.plot(pred_inverse, label='Predicted')
ax.set_xlabel('Time Step')
ax.set_ylabel('Emission')
ax.legend()
st.pyplot(fig)