# Crime Alert & Safety Advisory System
This is a smart Machine Learning-based system that helps people stay safe by predicting crime risks, 
sending real-time alerts, and giving safety advice based on location, time, and historical crime patterns.

# What This Project Does
1.Predicts Crime Type likely to happen in a selected area.
2.Shows Risk Level (Low / Medium / High) for any place.
3.Suggests High-Risk Time Periods based on crime trends.
4.Sends Alerts if a user enters a dangerous area (using GPS).
5.Shares Safety Tips to stay protected in risky zones.
6.Displays Crime-Prone Areas on an interactive map.


# How It Works
This project uses a combination of ML models and real-time location tracking:
-> Random Forest – Predicts the type of crime.
-> Risk Level Model – Classifies areas as low/medium/high risk.
-> Prophet / ARIMA – Predicts crime-prone time slots.
-> DBSCAN Clustering – Identifies dangerous zones.
-> Haversine Formula – Sends GPS-based alerts when user enters a danger zone.



# Tech Stack
Component	Tools Used
ML Models	            -->>  Random Forest, Prophet, DBSCAN
Frontend	        		-->>  Streamlit (Prototype)
Backend	       			  -->>  Flask APIs
Maps Integration	    -->>	Google Maps API / Mapbox
Realtime Alerts	      -->>  Firebase (planned)
Data Sources	        -->>  NCRB, Police FIR, OSM, Mock APIs
