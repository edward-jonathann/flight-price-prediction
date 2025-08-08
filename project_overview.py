import streamlit as st


def project_overview():
    st.title("✈️ Flight Price Prediction — Machine Learning Project")
    st.subheader("By Edward Jonathan")

    st.markdown("""
    ## 👋 Self Introduction
    Hi, I’m Edward — a professional transitioning from Project Management & Business Development into **Data Science**.  
    I’m passionate about turning raw, messy data into **insights that help people make smarter choices**.

    ---
    ## 📌 Project Overview
    One of the things I love about Data Science is turning raw, messy data into something useful — insights that help people make smarter choices.

    For my recent project, I took on the challenge of **predicting flight ticket prices** using Machine Learning.  
    Why? Because flight prices are unpredictable — and a reliable prediction model can help travelers save money and plan better.

    ### 🔍 What I Did
    - Collected & cleaned **300K+ flight records** from multiple sources (Economy & Business classes).
    - Engineered features like **season, duration, stops, departure/arrival times** to capture price patterns.
    - Explored the data to uncover trends:
        - **Class type** is the strongest driver of price (Business ≈ 5× Economy)
        - **Night flights** tend to cost the most, while **mornings** are most popular.
    - Built & compared **4 ML models** — **Random Forest** performed best with **98.2% accuracy** and **MAPE = 12%**.

    ### 💡 Why This Matters
    This approach can be applied far beyond flight prices — from predicting retail demand to optimizing logistics costs.  
    My skills in **data cleaning, feature engineering, exploratory analysis, and predictive modeling** can help businesses:
    - ✅ Identify cost drivers
    - ✅ Forecast trends
    - ✅ Optimize decision-making with data-backed insights
    """)
