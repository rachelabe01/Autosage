import streamlit as st
from app1 import app1
from app2 import app2
from app3 import app3

def main():
    # Set page config for favicon and layout
    st.set_page_config(page_title="Autosage", page_icon=":sparkles:", layout="wide")

    # Custom CSS for overall styling improvements, including the moving text animation and fancy background
    custom_css = """
        <style>
            /* Gradient background for the whole page */
            body {
                background: linear-gradient(to right, #6dd5ed, #f0f2f6);
            }
            
            /* Styling for headline and moving text */
            .headline {
                font-weight: bold;
                color: #FF4B4B;
            }
            .moving-text {
                white-space: nowrap;
                overflow: hidden;
                position: relative;
                animation: moveText 15s linear infinite;
            }
            @keyframes moveText {
                from { right: -100%; }
                to { right: 100%; }
            }

            /* Additional CSS for improved aesthetics as previously discussed */
            .css-1d391kg {
                padding: 1rem 1rem 1rem 1rem;
                background-color: #f0f2f6;
            }
            .stSidebar > div:first-child {
                background-color: #30475e;
                color: #f0f2f6;
            }
            .st-ae {
                color: #30475e;
            }
            label.st-ae:hover {
                background-color: #f05454;
                color: white;
            }
            .st-cx {
                background-color: #f05454 !important;
                color: white !important;
            }
            .css-1d391kg .st-bx {
                margin-bottom: 20px;
                border-bottom: 1px solid #dee2e6;
                padding-bottom: 20px;
            }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Header section with title and image
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("logo.jpg", width=100)  # Adjust path and size as needed
    with col2:
        # Update here for moving text
        st.markdown("<div class='moving-text'><h1 class='headline'>Welcome to Autosage!</h1></div>", unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title('🚀 Navigation')
    app_selection = st.sidebar.radio(
        'Go to', 
        ['Home', 'Text-based Data Model 🌟', 'Audio-based Data Model 🎈', 'Image-based Data Model 🎉']
    )

    if app_selection == 'Home':
        st.subheader('Home 🏠')
        st.write('Welcome! Use the navigation sidebar to explore the apps.')
        st.markdown('---')
        st.write('This is a multi-application hub where you can navigate to different apps showcasing various features and functionalities. Enjoy exploring!')
    elif app_selection == 'Text-based Data Model 🌟':
        app1()
    elif app_selection == 'Audio-based Data Model 🎈':
        app2()
    elif app_selection == 'Image-based Data Model 🎉':
        app3()

    # Add visual separation after app content
    st.markdown('---')

if __name__ == '__main__':
    main()
