import streamlit as st
import time
import base64

def tampilkan_tentang():
    st.title("About me!")

    st.write("""
    **Who I Am**  
    A results-driven professional transitioning from Project Management and Business 
Development to Data Science. Experienced in analyzing data to inform strategic decisions, 
optimizing project workflows, and delivering data-backed insights. Currently pursuing 
technical expertise in Python, SQL, and data science methodologies, with a strong 
foundation in quantitative analysis from an accounting background. 
    """)

    # Skills expander
    with st.expander("🔍 See my skills in detail"):
        st.progress(80, text="Python")
        st.progress(70, text="Machine Learning")
        st.progress(75, text="Power Bi")
        st.progress(80, text="SQL")
        st.progress(60, text="Tableau")
        st.progress(85, text="Microsoft Excel")

    if st.button("Is this what skills you're looking for?"):
        st.write("Send me a LinkedIn Invitation!")
    else:
        st.write("No? Then look around!")


    # PDF Resume Viewer
    st.divider()
    st.subheader("📄 Resume")

    try:
        with open("assets/misc/resume.pdf", "rb") as file:
            # Download Button
            st.download_button(
                label="Download Resume",
                data=file,
                file_name="Your_Name_Resume.pdf",
                mime="application/pdf"
            )
        
        # Show PDF inline
        with open("assets/misc/resume.pdf", "rb") as file:
            base64_pdf = base64.b64encode(file.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
            st.markdown("Or view inline:")
            st.markdown(pdf_display, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Error: Resume file not found at 'assets/misc/resume.pdf'")
