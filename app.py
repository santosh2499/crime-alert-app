import streamlit as st
import pandas as pd
import joblib
from geopy.geocoders import Nominatim
from datetime import datetime
import numpy as np
import geocoder

# Load ML models
crime_model = joblib.load("crime_type_model.pkl")
dbscan_model = joblib.load("high_risk_zones_model.pkl")  # DBSCAN model
geolocator = Nominatim(user_agent="crime_prediction_app")

# Sample location hierarchy (Replace with full list or from a database)
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
        st.error("Unable to get coordinates for selected location.")
        return None, None

def get_location_info(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language='en', timeout=10)  # Increased timeout
        if location and location.raw.get('address'):
            address = location.raw['address']
            return {
                'Country': address.get('country', ''),
                'State': address.get('state', ''),
                'District': address.get('county', ''),
                'City/Locality': address.get('suburb', '') or address.get('town', '') or address.get('city', '')
            }
    except Exception as e:
        st.warning("Location resolution failed: " + str(e))
    return {}


def predict_crime(lat, lon, current_time):
    hour = current_time.hour

    crime_features = pd.DataFrame([[
        lat, lon, hour, 3, 28.5, 0.0, 1, 2
    ]], columns=[
        'Latitude', 'Longitude', 'Hour', 'Crime_Severity',
        'Weather_Temperature_C', 'Weather_Rainfall_mm',
        'Traffic_Congestion_Level', 'Public_Complaints'
    ])

    crime_pred = crime_model.predict(crime_features)[0]
    risk_pred = "High Risk Time" if hour in [20, 21, 22, 23, 0, 1, 2] else "Low Risk"

    cluster = -1  # Default: noise
    labels = dbscan_model.labels_
    for label in set(labels):
        if label != -1:
            cluster = label
            break

    return crime_pred, risk_pred, cluster

# Streamlit UI
st.set_page_config(page_title="Crime Area Prediction", layout="centered")
st.title("📈 Crime Risk and Safety Advisor")

location_mode = st.radio("Choose Location Mode:", ["Use My Current Location (GPS)", "Enter Location Manually"])

if location_mode == "Use My Current Location (GPS)":
    g = geocoder.ip('me')
    lat, lon = g.latlng
    st.success(f"Current Location Detected: Latitude {lat}, Longitude {lon}")

else:
    country = st.selectbox("Select Country", list(locations.keys()))
    state = st.selectbox("Select State", list(locations[country].keys()))
    city = st.selectbox("Select City/Region", locations[country][state])

    full_location = f"{city}, {state}, {country}"
    lat, lon = get_coordinates(full_location)

    if lat and lon:
        st.success(f"Location Found: Latitude {lat:.4f}, Longitude {lon:.4f}")
    else:
        lat = st.number_input("Latitude:", value=28.6139)
        lon = st.number_input("Longitude:", value=77.2090)

if st.button("Predict Crime Risk"):
    loc_info = get_location_info(lat, lon)
    st.write("### Location Info")
    for k, v in loc_info.items():
        st.write(f"**{k}:** {v}")

    current_time = datetime.now()
    crime_type, risk_level, cluster_zone = predict_crime(lat, lon, current_time)

    st.subheader("Predictions")
    st.write(f"- **Predicted Crime Type:** {crime_type}")
    st.write(f"- **Risk Level:** {risk_level}")
    st.write(f"- **High Risk Zone Cluster:** {cluster_zone}")

    if risk_level == "High Risk Time":
        st.warning("Avoid traveling alone in this area during late hours.")
    else:
        st.info("This area is relatively safe at this time.")
