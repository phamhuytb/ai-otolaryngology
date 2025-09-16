import streamlit as st
from streamlit_option_menu import option_menu
import os
import cv2
import sys
import time
import requests
from dotenv import load_dotenv
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from ui_utils import *

sys.path.append(os.path.abspath \
                (os.path.join \
                (os.path.dirname(__file__), '../../../')))
sys.path.append(os.path.abspath \
                (os.path.join \
                (os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath \
                (os.path.join \
                (os.path.dirname(__file__), '../')))

from utils.load_config import load_config
config_path = "config/serving_config.yaml"
config=load_config(config_path)
load_dotenv()

import home,about,metrics

# Set Streamlit page configuration
st.set_page_config(
    page_title="AI Otolaryngology",  # Set the title of the Streamlit page
    layout="wide",  # Optional: Set the layout of the page (wide or centered)
    initial_sidebar_state="expanded"  # Optional: Set the initial state of the sidebar (expanded or collapsed)
)

# Render Google Analytics script using markdown
st.markdown(
    """
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src=f"https://www.googletagmanager.com/gtag/js?id={os.getenv('analytics_tag')}"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', os.getenv('analytics_tag'));
        </script>
    """, unsafe_allow_html=True) # Allow rendering of HTML content in Streamlit

# Print the value of 'analytics_tag' environment variable (for debugging purposes)
print(os.getenv('analytics_tag'))


# Set the title of the Streamlit page
st.title("Demo App: :red[ AI-Assisted Diagnosis] in Otolaryngological Endoscopy")


class MultiApp:
    """Main Streamlit app manager."""

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """
        Add a new app to the MultiApp instance.

        Parameters:
        - title (str): Title of the app.
        - func (function): Function that defines the app.

        Returns:
        None
        """

        self.apps.append({
            "title": title,
            "function": func
        })


    def run() -> None:
        """design app menu and routing to other sides"""

        with st.sidebar:        
            app = option_menu(
                menu_title='Menu ',
                options=['Home' ,'Metrics','About' ],
                icons=['house-fill','','info-circle-fill'],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                    "container": {"padding": "5!important","background-color":'pink'},
        "icon": {"color": "white", "font-size": "23px"}, 
        "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
        "nav-link-selected": {"background-color": "#02ab21"},} 
                )
        # Route to the selected app based on the sidebar menu selection
        if app == "Home":
            home.app() # Call the Home app function
        if app == "Metrics":
            metrics.app()   # Call the Metrics app function
        if app == "About":
            about.app()    # Call the About app function
   
    run()            




