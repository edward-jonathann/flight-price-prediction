import streamlit as st
import pickle
import numpy as np

@st.cache_resource 
def load_model():
    st.title("✈️💲 Flight Price Predictor")
    # For pickle
    with open("assets/machine_learning/model.pkl", "rb") as file:
        model = pickle.load(file)
    
    # Or for joblib
    # model = joblib.load('model.joblib')
    
        return model

    model = load_model()

    def date_to_season(input_date):
        month = input_date.month
        if month == 2:
            return "Winter"
        elif month == 3:
            return "Spring"
        else:
            return "Invalid month"
        
    selected_date = st.date_input(
    "Select a date (Feb 2 - Mar 31 only)",
    min_value=pd.Timestamp("2023-02-02"),  # Adjust year as needed
    max_value=pd.Timestamp("2023-03-31"),  # Adjust year as needed
    value=pd.Timestamp("2023-02-10")  # Default date
)
    Stop = st.radio("Num of Stop", ['2+-stop', '1-stop', 'non-stop'])
    Class = st.radio("What's your flight class?", ['economy', 'business'])
    Airline = st.radio("Feature 1", min_value=0.0, max_value=100.0)
    Origin = st.radio("Feature 2", 0, 10)
    Destination = st.radio("Feature 1", min_value=0.0, max_value=100.0)
    Departure_Time = st.time_input("Feature 1", min_value=0.0, max_value=100.0)
    Arrival_Time = st.time_input("Feature 2", 0, 10)
    Flight_Duration = st.slider("Flight Duration in hrs", min_value=0, max_value=50, step=0.5)

    if selected_date:
        season = date_to_season(selected_date)
        st.write(f"Selected date: {selected_date.strftime('%Y-%m-%d')}")
        st.success(f"Season: {season}")

    Season = season

    
    if st.button("Predict"):
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(2)
        my_bar.empty()

        input_array = np.array([[Stop, Class, Airline, Origin, Destination, Season, Departure_Time, Arrival_Time, Flight_Duration]])  # Reshape for model
        prediction = model.predict(input_array)[0]
        st.metric("Predicted Price", f"${prediction:,.2f}")

