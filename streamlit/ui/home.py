import base64
import os
import shutil
import sys
from math import ceil
from os import listdir

import requests
import streamlit as st
from PIL import Image
from streamlit.components.v1 import html
from streamlit_js_eval import streamlit_js_eval

from ui_utils import *
from utils.load_config import load_config

# Update system path to include custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Load configuration
config_path = "config/serving_config.yaml"
config = load_config(config_path)
IMAGE_UPLOAD_DIRECTORY = config['streamlit']['upload_dir'] + "images"
lottie_url_download = config['streamlit']['lottie_url']

# Load Lottie animation for loading spinner
lottie_load = load_lottieurl(lottie_url_download)


def create_folder_if_not_exists(folder_path: str) -> None:
    """Check if a folder exists at the given path, and create it if it does not exist.

    Args:
        folder_path (str): The path to the folder to be checked and created if it doesn't exist.

    Returns:
        None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def app() -> None:
    """Display image from user upload and send image to predictor API."""
    # Allow users to upload multiple images
    uploaded_files = st.file_uploader(
        "Choose image files",
        accept_multiple_files=True,
        type=config['fastapi']['datatype']
    )

    list_file_uploads = []

    # Ensure the upload directory exists
    create_folder_if_not_exists(IMAGE_UPLOAD_DIRECTORY)

    # Process each uploaded file
    for file in uploaded_files:
        file_location = os.path.join(IMAGE_UPLOAD_DIRECTORY, file.name)
        list_file_uploads.append(file_location)

        with open(file_location, "wb") as f:
            f.write(file.getbuffer())

        print(f"info: {file_location} : files uploaded successfully")

    if uploaded_files:
        # Display the uploaded images in a grid format
        html(get_grid_html(image_base64_convert(list_file_uploads)), height=get_dynamic_height(list_file_uploads))

        # Display buttons to submit the images for prediction or reload the page
        col1, col2 = st.columns(2)
        with col1:
            submit = st.button("Predict", type="primary")
        with col2:
            reload = st.button("Clear", type="primary", key="on_page")

        if submit:
            # Display a loading spinner while waiting for the prediction
            with st_lottie_spinner(lottie_load, key="download"):
                result = send_files_to_fastapi(uploaded_files)
                predicted = get_predict_caption(result)
                grid_row_image_display(uploaded_files, predicted)

        if reload:
            streamlit_js_eval(js_expressions="parent.window.location.reload()")

    else:
        st.write("No images uploaded yet. Please upload some images.")


if __name__ == "__main__":
    app()
