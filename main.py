import streamlit as st
import numpy as np
import pandas as pd
import datetime
from machine_learning_assets import load_model
from encoders import load_encoders

def flight_prediction():
    # ====== Distance data ======
    distances_km = {
        ("Delhi", "Mumbai"): 1150,
        ("Delhi", "Bangalore"): 1740,
        ("Delhi", "Kolkata"): 1310,
        ("Delhi", "Hyderabad"): 1260,
        ("Mumbai", "Bangalore"): 840,
        ("Mumbai", "Kolkata"): 1650,
        ("Mumbai", "Hyderabad"): 620,
        ("Bangalore", "Kolkata"): 1560,
        ("Bangalore", "Hyderabad"): 570,
        ("Kolkata", "Hyderabad"): 1210,
    }
    # Make symmetric
    for (a, b), d in list(distances_km.items()):
        distances_km[(b, a)] = d

    avg_speed_kmh = 800  # average cruising speed
    stop_delay_hours = {
        "non-stop": 0,
        "1-stop": 2,
        "2+-stop": 4
    }

    # ====== Load model and encoders ======
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

    # ====== Form ======
    min_date = datetime.date(2022, 2, 11)
    max_date = datetime.date(2022, 3, 31)

    with st.form("prediction_form"):
        date = st.date_input("Flight Date", min_value=min_date, max_value=max_date)
        airline = st.selectbox("Airline", ["Air India", "IndiGo", "SpiceJet", "Vistara", "GO FIRST", "AirAsia"])
        origin = st.selectbox("Origin", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad"])

        # Prevent same city in destination
        available_destinations = [city for city in ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad"] if city != origin]
        destination = st.selectbox("Destination", available_destinations)

        stops = st.selectbox("Stop", ["non-stop", "1-stop", "2+-stop"])
        flight_class = st.selectbox("Class", ["economy", "business"])

        # Dropdown for departure hour
        hour_labels = [
            f"{h:02d}:00" + (
                " (Midnight)" if h == 0 else
                " (Early Morning)" if 4 <= h < 8 else
                " (Morning)" if 8 <= h < 12 else
                " (Afternoon)" if 12 <= h < 16 else
                " (Evening)" if 16 <= h < 20 else
                " (Night)" if 20 <= h < 24 else
                ""
            )
            for h in range(24)
        ]
        hour_options = list(range(24))
        selected_hour_label = st.selectbox(
            "Departure Hour",
            options=hour_labels,
            index=8
        )
        departure_hour = hour_options[hour_labels.index(selected_hour_label)]

        submitted = st.form_submit_button("Predict Price")

    # ====== Prediction Logic ======
    if submitted:
        try:
            # Calculate time taken
            distance = distances_km.get((origin, destination), 0)
            base_time = distance / avg_speed_kmh if distance > 0 else 0
            layover_time = stop_delay_hours[stops]
            time_taken = round(base_time + layover_time, 2)

            # Calculate departure & arrival slots
            departure_dt = datetime.datetime.combine(date, datetime.time(hour=departure_hour))
            arrival_dt = departure_dt + datetime.timedelta(hours=time_taken)

            def get_time_slot(hour):
                if 4 <= hour < 8:
                    return "Early Morning"
                elif 8 <= hour < 12:
                    return "Morning"
                elif 12 <= hour < 16:
                    return "Afternoon"
                elif 16 <= hour < 20:
                    return "Evening"
                elif 20 <= hour < 24:
                    return "Night"
                else:
                    return "Late Night"

            departure_time_slot = get_time_slot(departure_dt.hour)
            arrival_time_slot = get_time_slot(arrival_dt.hour)

            st.write(f"🕒 Estimated Flight Duration: **{time_taken} hours**")
            st.write(f"📅 Departure: {departure_dt.strftime('%Y-%m-%d %H:%M')} ({departure_time_slot})")
            st.write(f"📅 Arrival: {arrival_dt.strftime('%Y-%m-%d %H:%M')} ({arrival_time_slot})")

            # Prepare DataFrame
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
                "departure_time": departure_time_slot,
                "arrival_time": arrival_time_slot
            }])

            # Encode
            input_df["airline"] = airline_encoder.transform([input_df["airline"][0]])
            input_df["from"] = from_encoder.transform([input_df["from"][0]])
            input_df["to"] = to_encoder.transform([input_df["to"][0]])
            input_df["departure_time"] = departure_time_encoder.transform([input_df["departure_time"][0]])
            input_df["arrival_time"] = arrival_time_encoder.transform([input_df["arrival_time"][0]])
            input_df["season"] = season_encoder.transform([input_df["season"][0]])
            input_df["stop"] = stop_encoder.transform(input_df[["stop"]])
            input_df["class"] = class_encoder.transform(input_df[["class"]])

            # Scale
            input_df["time_taken"] = scaler.transform(input_df[["time_taken"]])

            # Ensure order
            feature_names = [
                "airline", "from", "time_taken", "stop", "to",
                "class", "season", "departure_time", "arrival_time"
            ]
            input_df = input_df[feature_names]

            # Predict
            prediction = model.predict(input_df)[0]
            st.success(f"✈️ Estimated Flight Price: ₹ {prediction:,.0f}")

        except Exception as e:
            st.error("❌ Prediction failed.")
            st.write("Exception:", str(e))
            import traceback
            st.text(traceback.format_exc())
