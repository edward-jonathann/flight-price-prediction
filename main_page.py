import streamlit as st

st.set_page_config(page_title="Flight Prediction Demo",
                   layout="wide", page_icon=":airplane:")
st.title("Flight Prediction Demo")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Main Page",
                        ["Project Overview", "Prediction"])


if page == "Project Overview":
    import project_overview
    project_overview.project_overview()
elif page == "Prediction":
    import main
    main.flight_prediction()
