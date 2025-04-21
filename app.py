import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
import geocoder
import numpy as np
from prophet.serialize import model_from_json
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
from plotly.express import scatter_geo

# Load models
crime_model = joblib.load("models/crime_type_model.pkl")
with open("models/crime_trend_model.pkl", "rb") as f:
    crime_trend_model = joblib.load(f)
dbscan_model = joblib.load('models/high_risk_zones_model.pkl')

# Label mapping for the Random Forest classifier
label_map = {
    0: 'Assault',
    1: 'Burglary',
    2: 'Fraud',
    3: 'Harassment',
    4: 'Theft'
}

# Geolocator for location-based predictions
geolocator = Nominatim(user_agent="crime_predictor_app")

# Country, State, City structure
locations = {
    "India": {
        "Delhi": ["Connaught Place", "Saket"],
        "Maharashtra": ["Mumbai", "Pune"]
    },
    "USA": {
        "California": ["Los Angeles", "San Francisco"],
        "New York": ["New York City", "Buffalo"]
    }
}

def get_coordinates(place_name):
    try:
        location = geolocator.geocode(place_name)
        return location.latitude, location.longitude
    except:
        st.error("Unable to fetch coordinates. Try again.")
        return None, None

def get_location_info(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language='en', timeout=10)
        if location and location.raw.get('address'):
            address = location.raw['address']
            return {
                'Country': address.get('country', ''),
                'State': address.get('state', ''),
                'City': address.get('city', '') or address.get('town', '') or address.get('suburb', '')
            }
    except Exception as e:
        st.warning(f"Location info error: {e}")
    return {}

def predict_crime_type(lat, lon, hour):
    input_features = pd.DataFrame([[lat, lon, hour, 3, 28.5, 0.0, 1, 2]], 
                                  columns=['Latitude', 'Longitude', 'Hour', 'Crime_Severity',
                                           'Weather_Temperature_C', 'Weather_Rainfall_mm',
                                           'Traffic_Congestion_Level', 'Public_Complaints'])
    pred_label = crime_model.predict(input_features)[0]
    pred_class = label_map.get(pred_label, "Unknown")
    return pred_class

def forecast_crime_trend(future_date):
    future_df = pd.DataFrame({'ds': [pd.to_datetime(future_date)]})
    forecast = crime_trend_model.predict(future_df)
    predicted_count = int(forecast['yhat'].iloc[0])
    return predicted_count, forecast

def check_high_risk_zone(lat, lon):
    # Prepare coordinates for DBSCAN model
    coords_scaled = StandardScaler().fit_transform([[lat, lon]])
    risk_label = dbscan_model.fit_predict(coords_scaled)
    return "High Risk Zone" if risk_label == 1 else "Safe Zone"

# Streamlit UI setup
st.set_page_config(page_title="Crime Insight App", layout="wide")

# Custom CSS for styling the app
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #F47C7C;
        }
        .header {
            color: #2F4F4F;
            font-size: 32px;
            font-weight: 600;
        }
        .card {
            background-color: #F2F2F2;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .button {
            background-color: #5CB85C;
            color: white;
            font-size: 18px;
            font-weight: 600;
            border-radius: 8px;
            padding: 15px 30px;
            border: none;
            transition: background-color 0.3s ease;
        }
        .button:hover {
            background-color: #4CAF50;
        }
        .warning {
            color: #FF8C00;
            font-size: 18px;
        }
        .success {
            color: #28A745;
            font-size: 18px;
        }
        .map {
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for Navigation
st.sidebar.title("🔒 Crime Prediction Dashboard")
st.sidebar.subheader("Choose Mode")
mode = st.sidebar.radio("Location Mode", ["📍 Use My Current Location", "📝 Enter Manually"])

# Main content area
st.markdown("<h1 class='title'>🚨 Crime Type & Trend Predictor</h1>", unsafe_allow_html=True)

if mode == "📍 Use My Current Location":
    g = geocoder.ip('me')
    lat, lon = g.latlng if g.ok else (None, None)
    if lat and lon:
        st.success(f"Current Location: {lat:.4f}, {lon:.4f}")
    else:
        st.warning("Couldn't detect location.")
else:
    country = st.selectbox("Country", list(locations.keys()), help="Select your country.")
    state = st.selectbox("State", list(locations[country].keys()), help="Select the state.")
    city = st.selectbox("City", locations[country][state], help="Select the city.")
    full_location = f"{city}, {state}, {country}"
    lat, lon = get_coordinates(full_location)
    if lat and lon:
        st.success(f"Location Found: {lat:.4f}, {lon:.4f}")
    else:
        lat = st.number_input("Latitude", value=28.6139, help="Enter the latitude of your location.")
        lon = st.number_input("Longitude", value=77.2090, help="Enter the longitude of your location.")

use_current_time = st.checkbox("Use Current Time?", value=True, help="Select if you want to use the current time.")
if use_current_time:
    hour = datetime.now().hour
else:
    hour = st.slider("Select Hour of Day (0-23)", 0, 23, 12, help="Select the hour for crime prediction.")

# Predict Crime Type & Display Info
if st.button("🔮 Predict Crime Type & Trend", key="predict"):
    with st.spinner("Processing your request..."):
        loc_info = get_location_info(lat, lon)
        st.markdown("### 📌 Location Info")
        for key, val in loc_info.items():
            st.write(f"**{key}**: {val}")

        crime_prediction = predict_crime_type(lat, lon, hour)
        st.markdown("### 🧠 Crime Type Prediction")
        st.success(f"**Predicted Crime Type:** {crime_prediction}")

        risk_zone_status = check_high_risk_zone(lat, lon)
        st.markdown(f"### ⚠️ Risk Zone Status: {risk_zone_status}")

        if crime_prediction in ['Assault', 'Theft', 'Harassment']:
            st.warning("🚨 Caution: Avoid isolated areas and stay alert.")
        else:
            st.info("✅ Stay cautious and report suspicious activity.")

        # Forecasting Crime Trend
        future_date = st.date_input("Select a future date", datetime.today() + timedelta(days=1), help="Pick a date to forecast crime trends.")
        predicted_count, forecast = forecast_crime_trend(future_date)
        st.success(f"🔮 Predicted Crime Count on {future_date}: **{predicted_count} incidents**")

        # Plot Crime Trend Forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot')))
        fig.update_layout(title='Forecasted Crime Trend', xaxis_title='Date', yaxis_title='Predicted Crime Count')
        st.plotly_chart(fig, use_container_width=True)

        # Display Crime Data on Map
        st.subheader("📍 Crime Data on Map")
        fig_map = scatter_geo(data_frame=pd.DataFrame({'lat': [lat], 'lon': [lon]}),
                              locations="lat", hover_name="lon",
                              color=["High Risk" if risk_zone_status == "High Risk Zone" else "Safe Zone"],
                              color_continuous_scale="Viridis", projection="natural earth")
        st.plotly_chart(fig_map, use_container_width=True)
