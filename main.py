import streamlit as st
import numpy as np
import pandas as pd
import datetime
from machine_learning_assets import load_model
from encoders import load_encoders

model, feature_names = load_model()
encoders = load_encoders()
stop_encoder = encoders["stop_encoder"]
class_encoder = encoders["class_encoder"]
airline_encoder = encoders["airline_encoder"]
from_encoder = encoders["from_encoder"]
to_encoder = encoders["to_encoder"]
season_encoder = encoders["season_encoder"]
departure_time_encoder = encoders["departure_time_encoder"]
arrival_time_encoder = encoders["arrival_time_encoder"]
scaler = encoders["scaler"]

st.title("✈️ Flight Price Prediction")
st.write("Input flight details below to predict the price:")

min_date = datetime.date(2022, 2, 11)
max_date = datetime.date(2022, 3, 31)

with st.form("prediction_form"):
    date = st.date_input("Flight Date", min_value=min_date, max_value=max_date)
    airline = st.selectbox("Airline", ["Air India", "IndiGo", "SpiceJet", "Vistara", "GO FIRST", "AirAsia"])
    origin = st.selectbox("Origin", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad"])
    destination = st.selectbox("Destination", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad"])
    stops = st.selectbox("Stop", ["non-stop", "1-stop", "2+-stop"])
    time_taken = st.number_input("Time Taken (hours)", min_value=0.0, max_value=30.0, step=0.25)
    flight_class = st.selectbox("Class", ["economy", "business"])
    departure_time = st.selectbox("Departure Time", ["Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"])
    arrival_time = st.selectbox("Arrival Time", ["Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"])

    submitted = st.form_submit_button("Predict Price")

if submitted:
    try:
        st.write("Collecting input...")
        input_df = pd.DataFrame([{
            "airline": airline,
            "from": origin,
            "time_taken": time_taken,
            "stop": stops,
            "to": destination,
            "class": flight_class,
            "season": (
                "Winter" if date.month in [12, 1, 2]
                else "Spring" if date.month in [3, 4, 5]
                else "Summer" if date.month in [6, 7, 8]
                else "Autumn"
            ),
            "departure_time": departure_time,
            "arrival_time": arrival_time
        }])
        st.write("Initial input:", input_df)

        # Encode
        st.write("Encoding...")
        input_df["airline"] = airline_encoder.transform([input_df["airline"][0]])
        input_df["from"] = from_encoder.transform([input_df["from"][0]])
        input_df["to"] = to_encoder.transform([input_df["to"][0]])
        input_df["departure_time"] = departure_time_encoder.transform([input_df["departure_time"][0]])
        input_df["arrival_time"] = arrival_time_encoder.transform([input_df["arrival_time"][0]])
        input_df["season"] = season_encoder.transform([input_df["season"][0]])
        input_df["stop"] = stop_encoder.transform(input_df[["stop"]])
        input_df["class"] = class_encoder.transform(input_df[["class"]])

        # Scale
        st.write("Scaling...")
        input_df["time_taken"] = scaler.transform(input_df[["time_taken"]])

        # Ensure correct order
        feature_names = [
            "airline", "from", "time_taken", "stop", "to",
            "class", "season", "departure_time", "arrival_time"
        ]
        input_df = input_df[feature_names]

        # Predict
        st.write("Predicting...")
        prediction = model.predict(input_df)[0]
        st.success(f"✈️ Estimated Flight Price: ₹ {prediction:,.0f}")

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.write("Exception:", str(e))
        import traceback
        st.text(traceback.format_exc())
