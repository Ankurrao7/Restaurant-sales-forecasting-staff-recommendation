import pandas as pd
import numpy as np
import math
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("restaurant_sales.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Weather'] = df['Weather'].astype('category').cat.codes
    df['Holiday'] = df['Holiday'].map({'Yes': 1, 'No': 0})
    return df

# -------------------------------
# Train Model
# -------------------------------
@st.cache_resource
def train_model(df):
    X = df[['DayOfWeek', 'Weather', 'Holiday', 'Customers']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    return model, rmse

# -------------------------------
# UI
# -------------------------------
st.title("🍽 Restaurant Forecasting")

# Load and train
df = load_data()
model, rmse = train_model(df)

st.write(f"**Model RMSE:** {rmse:.2f}")

# Sidebar input
days = st.sidebar.slider("Days to Forecast", 1, 30, 7)

# Generate future data
future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=days)
future_data = pd.DataFrame({
    'Date': future_dates,
    'DayOfWeek': future_dates.dayofweek,
    'Weather': np.random.randint(0, 4, size=days),
    'Holiday': np.random.randint(0, 2, size=days),
    'Customers': np.random.randint(80, 200, size=days)
})

# Predictions
future_data['PredictedSales'] = model.predict(
    future_data[['DayOfWeek', 'Weather', 'Holiday', 'Customers']]
)
future_data['StaffNeeded'] = future_data['Customers'].apply(lambda x: math.ceil(x / 30))

# Show table
st.dataframe(future_data, use_container_width=True)

# Download CSV
csv = future_data.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "staff_plan.csv", "text/csv")
