import streamlit as st
import pandas as pd
import datetime
import random
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

STOP_DELAY = {
    "non-stop": 0,
    "1-stop": 2,
    "2+-stop": 4
}

STOP_LABELS = {
    "non-stop": "Non-stop",
    "1-stop": "1 Stop",
    "2+-stop": "2 Stops"
}

LCC_AIRLINES = ["Indigo", "SpiceJet", "GO FIRST", "AirAsia"]

# =========================
# Helper Functions
# =========================
def format_duration(hours_float):
    hours = int(hours_float)
    minutes = int(round((hours_float - hours) * 60))
    return f"{hours}H {minutes}m"


def format_price(price):
    return f"{int(round(price)):,}"


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


# =========================
# Demand Multipliers
# =========================
def get_weekday_multiplier(date):
    return random.uniform(1.05, 1.10)


def get_last_minute_multiplier(date):
    today = datetime.date.today()

    if date == today:
        return 4.0
    elif date == today + datetime.timedelta(days=1):
        return 4.0
    else:
        return 1.0


# =========================
# Price Prediction Core
# =========================
def predict_price(
    date,
    hour,
    origin,
    destination,
    airline,
    flight_class,
    stop_value
):
    distance = DISTANCES_KM[(origin, destination)]

    time_taken = round(
        distance / AVG_SPEED_KMH + STOP_DELAY[stop_value],
        2
    )

    departure_dt = datetime.datetime.combine(date, datetime.time(hour=hour))
    arrival_dt = departure_dt + datetime.timedelta(hours=time_taken)

    season = get_safe_season(date.month)

    input_df = pd.DataFrame([{
        "airline": airline,
        "from": origin,
        "time_taken": time_taken,
        "stop": stop_value,
        "to": destination,
        "class": flight_class,
        "season": season,
        "departure_time": get_time_slot(hour),
        "arrival_time": get_time_slot(arrival_dt.hour)
    }])

    # Encoding
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

    base_price = model.predict(input_df)[0]

    # Apply multipliers
    weekday_multiplier = get_weekday_multiplier(date)
    last_minute_multiplier = get_last_minute_multiplier(date)

    final_price = base_price * weekday_multiplier * last_minute_multiplier

    return final_price, time_taken


# =========================
# Cheapest Day Finder
# =========================
def find_cheapest_day(
    selected_date,
    origin,
    destination,
    airline,
    flight_class
):
    today = datetime.date.today()

    # =========================
    # Determine date range
    # =========================
    if selected_date <= today:
        # If today selected → show next 7 days
        date_list = [today + datetime.timedelta(days=i) for i in range(7)]
    else:
        # ±3 days around selected date
        temp_list = [
            selected_date + datetime.timedelta(days=i)
            for i in range(-3, 4)
        ]

        # Remove past dates
        date_list = [d for d in temp_list if d >= today]

    hours_to_test = range(0, 24, 3)

    results = []

    for test_date in date_list:

        day_min_price = float("inf")

        for hour in hours_to_test:
            for stop_value in STOP_DELAY.keys():

                price, _ = predict_price(
                    test_date,
                    hour,
                    origin,
                    destination,
                    airline,
                    flight_class,
                    stop_value
                )

                if price < day_min_price:
                    day_min_price = price

        results.append({
            "Date": test_date,
            "Price": int(round(day_min_price))
        })

    return pd.DataFrame(results)


# =========================
# UI
# =========================
def flight_prediction():

    st.title("✈️ Flight Price Prediction")

    with st.form("prediction_form"):

        date = st.date_input(
            "Departure Date",
            min_value=datetime.date.today(),
            max_value=datetime.date(2030, 12, 31)
        )

        airline = st.selectbox(
            "Airline",
            ["Air India", "Indigo", "SpiceJet", "Vistara", "GO FIRST", "AirAsia"]
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

        submitted = st.form_submit_button("Search Flights")

    if not submitted:
        return

    # =========================
    # Cheapest Time Finder
    # =========================
    hours_to_test = range(0, 24, 2)

    all_results = []

    for hour in hours_to_test:
        for stop_value in STOP_DELAY.keys():

            price, duration = predict_price(
                date,
                hour,
                origin,
                destination,
                airline,
                flight_class,
                stop_value
            )

            all_results.append({
                "Departure": f"{hour:02d}:00",
                "Stops": STOP_LABELS[stop_value],
                "Duration": format_duration(duration),
                "Price": int(round(price))
            })

    results_df = pd.DataFrame(all_results)

    cheapest_row = results_df.loc[results_df["Price"].idxmin()]

    st.success(
        f"🔥 Cheapest Time: {cheapest_row['Departure']} "
        f"({cheapest_row['Stops']}) — ₹{format_price(cheapest_row['Price'])}"
    )

    st.subheader("Best Flight Options")

    display_df = results_df.sort_values("Price").head(5).copy()

    # Identify cheapest price
    min_price = display_df["Price"].min()

    def highlight_best(row):
        if row["Price"] == min_price:
            return ["background-color: #1f77b4; color: white; font-weight: bold"] * len(row)
        else:
            return [""] * len(row)

    display_df["Price"] = display_df["Price"].apply(format_price)

    st.dataframe(
        display_df.style.apply(highlight_best, axis=1),
        use_container_width=True, hide_index=True
    )

    # =========================
    # Cheapest Day Finder
    # =========================
    st.subheader("📅 Cheapest Day Around Your Selection")

    cheapest_days_df = find_cheapest_day(
        date,
        origin,
        destination,
        airline,
        flight_class
    )

    best_row = cheapest_days_df.loc[cheapest_days_df["Price"].idxmin()]

    st.success(
        f"⭐ Cheapest Day: {best_row['Date']} — ₹{format_price(best_row['Price'])}"
    )

    display_days = cheapest_days_df.copy()
    display_days["Date"] = display_days["Date"].apply(lambda d: d.strftime("%b %d"))
    display_days["Price"] = display_days["Price"].apply(format_price)

    min_price_day = cheapest_days_df["Price"].min()

    def highlight_day(row):
        if row["Price"] == format_price(min_price_day):
            return ["background-color: #2ca02c; color: white; font-weight: bold"] * len(row)
        return [""] * len(row)

    styled_days = display_days.style.apply(highlight_day, axis=1)

    st.dataframe(styled_days, use_container_width=True, hide_index=True)
