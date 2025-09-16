import streamlit as st
import requests
from os import listdir
from math import ceil
import os
from PIL import Image 
from streamlit.components.v1 import html
import base64
import shutil
import sys
from streamlit_js_eval import streamlit_js_eval
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
IMAGE_UPLOAD_DIRECTORY = config['streamlit']['upload_dir']  + "images"
lottie_url_download = config['streamlit']['lottie_url'] 

lottie_load = load_lottieurl(lottie_url_download)


def app() -> None:
    """Display image from user upload and sent image to predictor api"""
    # Allow users to upload multiple images
    uploaded_files = st.file_uploader(
        "Choose image files",
        accept_multiple_files=True,
        type=config['fastapi']['datatype']  # Specify accepted file types from config
    )
    


    list_file_uploads = [] # Initialize list to store file paths

    # Ensure the upload directory exists
    create_folder_if_not_exists(IMAGE_UPLOAD_DIRECTORY)

    # Process each uploaded file
    for file in uploaded_files:
        file_location = os.path.join(IMAGE_UPLOAD_DIRECTORY, file.name)     # Define file location
        list_file_uploads.append(file_location)     # Add file location to list


        with open(file_location, "wb") as f:     # Save the uploaded file to the defined location
            f.write(file.getbuffer())

        # Print a success message            
        print( "info: " + f"{file_location} : files uploaded successfully")
    
    if uploaded_files  :  
        # Display the uploaded images in a grid format
        html(get_grid_html(image_base64_convert( list_file_uploads)),height=get_dynamic_height(list_file_uploads))

        # Display a button to submit the images for prediction/reload page
        col1, col2 = st.columns(2)
        with col1:
            submit = st.button("Predict", type="primary")
        with col2:
            reload = st.button("Clear", type="primary", key="on_page")


        if submit :
            # Display a loading spinner while waiting for the prediction
            with st_lottie_spinner(lottie_load, key="download"):
                # Send the uploaded files to the FastAPI predictor endpoint
                result = send_files_to_fastapi(uploaded_files)

                # Get predictions from the API response
                predicted = get_predict_caption(result)

                # Display the images along with their predictions
                row_image_display(uploaded_files,predicted)


        if reload:
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
            
    else:
        # Display a message if no images are uploaded
        st.write("No images uploaded yet. Please upload some images.")





 