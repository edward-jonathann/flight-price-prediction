def flight_prediction():
    import streamlit as st
    import pandas as pd
    import datetime
    from machine_learning_assets import load_model
    from encoders import load_encoders

    # =========================
    # Load model & encoders
    # =========================
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

    # =========================
    # Constants
    # =========================
    DISTANCES_KM = {
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

    for (a, b), d in list(DISTANCES_KM.items()):
        DISTANCES_KM[(b, a)] = d

    AVG_SPEED_KMH = 800
    STOP_DELAY = {"non-stop": 0, "1-stop": 2, "2+-stop": 4}
    LCC_AIRLINES = ["IndiGo", "SpiceJet", "GO FIRST", "AirAsia"]

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

    def get_safe_season(month):
        return "Winter" if month in [12, 1, 2] else "Spring"
    
    def format_duration(hours_float):
        hours = int(hours_float)
        minutes = int(round((hours_float - hours) * 60))
        return f"{hours}H {minutes}m"

    def format_price(price):
        return f"{int(round(price)):,}"

    # =========================
    # UI
    # =========================
    st.title("✈️ Flight Price Prediction")

    with st.form("prediction_form"):
        date = st.date_input(
            "Departure Date",
            min_value=datetime.date.today(),
            max_value=datetime.date(2030, 12, 31)
        )

        airline = st.selectbox(
            "Airline",
            ["Air India", "IndiGo", "SpiceJet", "Vistara", "GO FIRST", "AirAsia"]
        )

        origin = st.selectbox(
            "Origin",
            ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad"]
        )

        destination = st.selectbox(
            "Destination",
            [c for c in ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad"] if c != origin]
        )

        class_options = ["economy"] if airline in LCC_AIRLINES else ["economy", "business"]
        flight_class = st.selectbox("Class", class_options)

        hour_labels = [
            f"{h:02d}:00" + (
                " (Early Morning)" if 4 <= h < 8 else
                " (Morning)" if 8 <= h < 12 else
                " (Afternoon)" if 12 <= h < 16 else
                " (Evening)" if 16 <= h < 20 else
                " (Night)" if h >= 20 else ""
            )
            for h in range(24)
        ]

        selected_hour = st.selectbox("Departure Time", hour_labels, index=8)
        departure_hour = hour_labels.index(selected_hour)

        submitted = st.form_submit_button("Search Flights")

    if not submitted:
        return

    distance = DISTANCES_KM[(origin, destination)]
    season = get_safe_season(date.month)

    stop_scenarios = {
        "Non-stop": "non-stop",
        "1 Stop": "1-stop",
        "2 Stops": "2+-stop"
    }

    results = []

    for label, stop_value in stop_scenarios.items():
        time_taken = round(distance / AVG_SPEED_KMH + STOP_DELAY[stop_value], 2)

        departure_dt = datetime.datetime.combine(date, datetime.time(hour=departure_hour))
        arrival_dt = departure_dt + datetime.timedelta(hours=time_taken)

        input_df = pd.DataFrame([{
            "airline": airline,
            "from": origin,
            "time_taken": time_taken,
            "stop": stop_value,
            "to": destination,
            "class": flight_class,
            "season": season,
            "departure_time": get_time_slot(departure_hour),
            "arrival_time": get_time_slot(arrival_dt.hour)
        }])

        input_df["airline"] = airline_encoder.transform(input_df[["airline"]])
        input_df["from"] = from_encoder.transform(input_df[["from"]])
        input_df["to"] = to_encoder.transform(input_df[["to"]])
        input_df["stop"] = stop_encoder.transform(input_df[["stop"]])
        input_df["class"] = class_encoder.transform(input_df[["class"]])
        input_df["season"] = season_encoder.transform(input_df[["season"]])
        input_df["departure_time"] = departure_time_encoder.transform(input_df[["departure_time"]])
        input_df["arrival_time"] = arrival_time_encoder.transform(input_df[["arrival_time"]])
        input_df["time_taken"] = scaler.transform(input_df[["time_taken"]])

        input_df = input_df[feature_names]

        price = model.predict(input_df)[0]

        results.append({
            "Stops": label,
            "Flight Duration": format_duration(time_taken),
            "Estimated Price (₹)": format_price(price)
        })


    st.subheader("Available Flight Options")
    st.table(pd.DataFrame(results))
