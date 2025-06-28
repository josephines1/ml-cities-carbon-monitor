import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from keras.models import load_model
from dateutil.relativedelta import relativedelta

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
df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")

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

# Load scaler & Model
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
GRU_PATH = os.path.join(BASE_DIR, 'model', 'gru_model.h5')
LSTM_PATH = os.path.join(BASE_DIR, 'model', 'lstm_model.h5')

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_resource
def load_selected_model(model_name):
    if model_name == 'GRU':
        return load_model(GRU_PATH, compile=False)
    else:
        return load_model(LSTM_PATH, compile=False)

scaler = load_scaler()
model = load_selected_model(selected_model)

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

# --- Ambil tanggal dari data (pastikan format datetime)
df_filtered['date'] = pd.to_datetime(df_filtered['date'], dayfirst=True)
dates_plot = df_filtered['date'].values[window_size:]

# --- Squeeze data aktual dan prediksi
actual_plot = actual.squeeze()
pred_plot = pred_inverse.squeeze()

# --- Plot aktual vs prediksi
st.subheader(f"📈 {selected_model} Prediction vs Actual for {selected_city} - {selected_sector}")

# --- Buat input tanggal start & end side by side
col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "📅 Start Date",
        value=pd.to_datetime(dates_plot[-100]),
        min_value=pd.to_datetime(dates_plot[0]),
        max_value=pd.to_datetime(dates_plot[-1])
    )

with col2:
    end_date = st.date_input(
        "📅 End Date",
        value=pd.to_datetime(dates_plot[-1]),
        min_value=start_date,
        max_value=pd.to_datetime(dates_plot[-1])
    )

# --- Konversi array dates_plot ke datetime
dates_dt = pd.to_datetime(dates_plot)

# --- Cari indeks start dan end dari date picker
try:
    start_idx = np.where(dates_dt >= np.datetime64(start_date))[0][0]
    end_idx = np.where(dates_dt <= np.datetime64(end_date))[0][-1] + 1
except IndexError:
    st.error("❌ Rentang tanggal yang dipilih tidak valid.")
    st.stop()

# --- Potong data
dates_display = dates_plot[start_idx:end_idx]
actual_display = actual_plot[start_idx:end_idx]
pred_display = pred_plot[start_idx:end_idx]

forecast_mode = st.radio(
    "🔮 Choose forecast range",
    options=["1 bulan", "3 bulan", "6 bulan", "1 tahun", "Custom"],
    horizontal=True
)

if forecast_mode == "1 bulan":
    n_days_ahead = 30
elif forecast_mode == "3 bulan":
    n_days_ahead = 90
elif forecast_mode == "6 bulan":
    n_days_ahead = 180
elif forecast_mode == "1 tahun":
    n_days_ahead = 365
else:
    custom_date = st.date_input(
        "📅 Select end date for prediction",
        min_value=pd.to_datetime(dates_plot[-1]) + pd.Timedelta(days=1),
        value=pd.to_datetime(dates_plot[-1]) + pd.Timedelta(days=30)
    )
    custom_date = pd.to_datetime(custom_date)
    n_days_ahead = (custom_date - pd.to_datetime(dates_plot[-1])).days

if n_days_ahead <= 0:
    st.warning("⚠️ Harap pilih tanggal prediksi yang berada di masa depan.")
    st.stop()

# --- Prediksi masa depan secara berantai
last_window = scaled_values[-window_size:]
future_preds = []

current_input = last_window.copy()
for _ in range(n_days_ahead):
    input_reshaped = current_input.reshape(1, window_size, 1)
    next_pred = model.predict(input_reshaped, verbose=0)
    future_preds.append(next_pred[0, 0])
    current_input = np.append(current_input[1:], next_pred, axis=0)

# --- Inverse transform hasil prediksi ke depan
future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# --- Generate tanggal ke depan
last_date = pd.to_datetime(dates_plot[-1])
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days_ahead)

# Gabungkan tanggal & nilai masa depan
full_dates = np.concatenate([dates_display, future_dates])
full_pred_values = np.concatenate([pred_display, future_preds_inv.reshape(-1)])

# Plot gabungan: actual + predicted + future
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dates_display, actual_display, label='Actual')
ax.plot(dates_display, pred_display, label='Predicted')
ax.plot(future_dates, future_preds_inv, label='Future Forecast', linestyle='--', color='red')
ax.set_xlabel('Date')
ax.set_ylabel('Emission')
ax.set_title(f"{selected_model} Emission Forecast with Historical Comparison ({selected_city} - {selected_sector})")
ax.legend()
st.pyplot(fig)

# Insight sederhana
st.subheader("📊 Insights")
if pred_inverse[-1] > pred_inverse[0]:
    direction = "meningkat"
elif pred_inverse[-1] < pred_inverse[0]:
    direction = "menurun"
else:
    direction = "stabil"

# Ambil scalar value dari array
delta = abs(pred_inverse[-1][0] - pred_inverse[0][0])
max_pred = pred_inverse.max()
min_pred = pred_inverse.min()

start_pred = last_date + pd.Timedelta(days=1)
end_pred = last_date + pd.Timedelta(days=n_days_ahead)
delta_time = relativedelta(end_pred, start_pred)

duration_str = ""
if delta_time.years > 0:
    duration_str += f"{delta_time.years} tahun "
if delta_time.months > 0:
    duration_str += f"{delta_time.months} bulan "
if delta_time.days > 0:
    duration_str += f"{delta_time.days} hari"
duration_str = duration_str.strip()

# Hitung berapa hari/tahun dari data aktual
start_actual = pd.to_datetime(dates_plot[0])
end_actual = pd.to_datetime(dates_plot[-1])
n_days = (end_actual - start_actual).days

st.markdown(f"""
- Selama periode historis dari **{start_actual.strftime('%d %b %Y')}** hingga **{end_actual.strftime('%d %b %Y')}**, emisi diprediksi telah **{direction}** sebesar **{delta:.2f}**.
- Titik maksimum prediksi: **{max_pred:.2f}**  
- Titik minimum prediksi: **{min_pred:.2f}**
- Prediksi ke depan menunjukkan bahwa emisi akan mencapai **{future_preds_inv[-1][0]:.2f}** pada **{future_dates[-1].strftime('%d %b %Y')}**.
- Rata-rata emisi yang diprediksi dalam {n_days_ahead} hari ke depan: **{np.mean(future_preds_inv):.2f}**
""")

# Evaluasi
mae = mean_absolute_error(actual, pred_inverse)
mse = mean_squared_error(actual, pred_inverse)
rmse = math.sqrt(mse)

# Panel evaluasi
st.subheader("Model Evaluation")

# Interpretasi sederhana
if rmse < 0.05:
    st.success("✅ Model sangat akurat")
elif rmse < 0.1:
    st.info("ℹ️ Model cukup akurat")
else:
    st.warning("⚠️ Model masih bisa ditingkatkan")

st.markdown(f"""
- **MAE (Mean Absolute Error):** {mae:.4f}  
- **RMSE (Root Mean Squared Error):** {rmse:.4f}  
""")