import streamlit as st
import time
import pandas as pd
import numpy as np

def tampilkan():
    st.title("Hi, I'm Edward Jonathan")
    st.subheader("Data Scientist Enthusiast | ML Engineer Enthusiast")

    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["Company A", "Company B", "Company C"])

    st.area_chart(chart_data)


    # # Interactive button
    # if st.button("👉 Click for a fun fact!"):
    #     with st.spinner("Generating..."):
    #         time.sleep(1)
    #         st.success("Ironically, the fear of long words is called Hippopotomonstrosesquippedaliophobia. The 36-letter word was first used in the first century BCE to criticise writers with an unreasonable penchant for long words.")

    st.markdown("""
    Welcome to my portfolio! Here’s what you’ll find:
    - **About Me**: Hi there! I’m Edward Jonathan, a curious and driven professional on a transformative journey from project management and business development to the exciting world of data science.
    - **Projects**: See my projects page!
    - **Contact**: Let’s connect on [LinkedIn](https://www.linkedin.com/in/edward-jonathann/)!
    """)