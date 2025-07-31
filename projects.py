import streamlit as st

def tampilkan():
    # Custom CSS for dropdown/buttons (white text on dark backgrounds)
    st.markdown("""
    <style>
        /* Dropdown text color */
        .stSelectbox label, .stSelectbox div[data-baseweb="select"] {
            color: white !important;
        }
        /* Button text color */
        .stButton>button {
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Project dropdown (list of projects)
    projects = {
        "Customer Segmentation": {
            "description": "A Customer Segmentation analysis with RFM",
            "skills": "Python, Pandas, PowerBI",
            "link": "https://github.com/your-repo1"
        },
        "Customer Clustering": {
            "description": "K-means model with PCA and TSNE to find customer's cluster.",
            "skills": "Python, K-means, PCA, TSNE",
            "link": "https://github.com/your-repo2"
        },
        "Churn Analysis": {
            "description": "Build a model to predict churning",
            "skills": "Python, Scikit-learn",
            "link": "https://github.com/your-repo3"
        },
        "Flight Price Prediction": {
            "description": "Build a model to predict ticket prices",
            "skills": "Python, Scikit-learn, XGBoost, CatBoost",
            "link": "https://github.com/your-repo3"
        }
        }
    
    selected_project = st.selectbox(
        "Select a project to view details:",
        options=list(projects.keys()),
        format_func=lambda x: f"📌 {x}"  # Add icon to dropdown items
    )
    
    # Display selected project details
    with st.container(border=True):
        st.subheader(f"**{selected_project}**")
        st.markdown(f"**Description**: {projects[selected_project]['description']}")
        st.markdown(f"**Skills**: `{projects[selected_project]['skills']}`")
        
        # Interactive buttons (with white text)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Code ↗", key=f"code_{selected_project}"):
                st.markdown(f"[Work In Progress]({projects[selected_project]['link']})")
                
        with col2:
            if st.button("See Demo ↗", key=f"demo_{selected_project}"):
                st.write("No Demo available yet!")  # Replace with actual demo link

    # Overview of all projects (collapsible)
    with st.expander("📋 **Quick Overview of All Projects**", expanded=True):
        for project, details in projects.items():
            st.markdown(f"""
            - **{project}**: {details['description']}  
              *(Skills: {details['skills']})*
            """)