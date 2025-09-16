from PIL import Image
import streamlit as st
import requests
from os import listdir
from math import ceil
import os
from typing import List
from typing import Optional, Dict, Any
import streamlit as st
from streamlit.components.v1 import html
import base64
import sys
import time
import json
import streamlit as st
import streamlit_lottie
from streamlit_lottie import st_lottie_spinner
from streamlit_lottie import st_lottie
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

# Path to your configuration file
config_path = "config/serving_config.yaml"

# Load the configuration using the load_config function
config=load_config(config_path)


def send_files_to_fastapi(files: object) -> dict:
    '''
    Send files to a FastAPI server endpoint.

    Args:
        files (object): Object containing files to send.

    Returns:
        dict: JSON response from the FastAPI server.
    '''
    # URL of the FastAPI server
    API_URL = f"{config['fastapi']['protocol']}"+ \
            f"{config['streamlit']['host']}" + \
            f"{config['fastapi']['port']}" + \
            f"{config['fastapi']['classify']}"
    
    # Prepare files in the required format
    multiple_files = [('files', (file.name, file.read(), file.type)) for file in files]

    # Send POST request to FastAPI server
    response = requests.post(API_URL, files=multiple_files)

    # Return JSON response from the server
    return response.json()


def get_display_size(image: str) -> tuple:
    '''
    Function to determine appropriate display dimensions for an image.

    Args:
        image (str): Path or URL of the image.

    Returns:
        tuple: Tuple containing new width and height dimensions.
    '''

    # Read the image file
    image = Image.open(image)

    # Define a maximum height to avoid very tall images
    max_width = config['size_image']['width']
    max_height = config['size_image']['height'] 
    
    # Calculate the scale factor needed to fit width or height
    scale_factor = min(max_width / image.width, max_height / image.height)
    
# Calculate new dimensions based on scale factor
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    return new_width, new_height


def grid_image_display (image_list : list, *caption_list:list) -> None:
    '''
    Display images in a grid layout.

    Args:
        image_list (list): List of image paths or URLs.
        *caption_list (list): Variable length list of lists containing captions for each image.
    '''
    grid = st.columns(3)
    col = 0
    if caption_list :   
        for index in range(len(caption_list)):
            display_width, display_height = get_display_size(image_list[index])
            with grid[col]:
                st.image(image, width=display_width,caption=caption_list[index],use_column_width=True)
            col = (col + 1) % 3
    else :
        for image in image_list:
            display_width, display_height = get_display_size(image)
            with grid[col]:
                st.image(image, width=display_width)
            col = (col + 1) % 3



def row_image_display (image_list : list, *caption_list:list) -> None:
    '''
    Display images and their corresponding captions in rows.

    Args:
        image_list (list): List of image paths or URLs.
        *caption_list (list): Variable length list of lists containing predicted captions.
    '''
    for index in range (len(image_list)):
        with st.container():
            col1, col2 = st.columns([2, 1])   
            with col1:
                st.image(image_list[index], use_column_width=True) 
            with col2:
                st.success(caption_list[0][index])



def get_predict_caption (list_predict : dict) -> list :
    '''
    Get JSON from AI server and return a list of diseases.

    Args:
        list_predict (dict): JSON dictionary containing predictions.

    Returns:
        list: List of diseases predicted by the AI server.
    '''
    return list(list_predict['predictions'])



def get_image_base64(image_path: str) -> str:
    """
    Convert an image file to Base64 encoding.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image file, or None if file is not found.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None  # Return None if file is not found


def image_base64_convert(image_sources: list) -> list:
    """
    Convert local images to Base64 format or use the URL directly.

    Args:
        image_sources (list): List of image sources (URLs or file paths).

    Returns:
        list: List of image sources converted to Base64 format or used directly.
    """
    # Convert local images to Base64 or use the URL directly
    images = [
    f"data:image/jpeg;base64,{get_image_base64(image)}" if not image.startswith('http') else image
        for image in image_sources if get_image_base64(image) is not None or image.startswith('http')
    ]
    return images


def get_grid_html (images: list) -> str:  
    """
    Generate HTML with CSS for displaying a grid of images in a scrollable container.

    Args:
        images (list): List of image sources (URLs or file paths).

    Returns:
        str: HTML string with embedded CSS for the image grid.
    """

    # CSS for the scrollable container and grid
    grid_html = f"""
    <style>
        .scrollable-container {{
            overflow-y: auto;
            height: 300px;  # Adjust the height of the scrollable area

        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);  # Adjust the number of columns as necessary
            gap: 10px;
            padding: 10px;
        }}
        .image-grid img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
        }}
    </style>
    <div class="scrollable-container">
        <div class="image-grid">
    """

    # Add each image to the HTML
    for src in images:
        grid_html += f'<img src="{src}" alt="Display Image">'


    # Close the HTML tags
    grid_html += """
        </div>
    </div>
    """
    return grid_html

def dynamic_grid_css(num_images: int, max_columns: int = 3) -> tuple:
    """
    Generate dynamic CSS grid styles based on the number of images and maximum columns.

    Args:
        num_images (int): The number of images to display.
        max_columns (int, optional): The maximum number of columns in the grid. Defaults to 3.

    Returns:
        tuple: A tuple containing the generated CSS styles (str) and the actual number of columns (int).
    """
    columns = min(max_columns, num_images)  # Adjust 'max_columns' to the desired number of columns
    return f"""
        display: grid;
        grid-template-columns: repeat({columns}, 1fr);
        gap: 10px;
        padding: 10px;
    """, columns


def get_dynamic_height(image_sources: list) -> int:
    """
    Calculate the dynamic height for a grid layout based on the number of images.

    Args:
        image_sources (list): A list of image sources.

    Returns:
        int: The calculated dynamic height for the grid layout.
            Adjusted to match the actual row height including gaps.
    """

    # Generate CSS dynamically based on the number of images
    grid_style, num_columns = dynamic_grid_css(len(image_sources))

    # Calculate the number of rows based on the number of images and columns
    num_rows = (len(image_sources) + num_columns - 1) // num_columns  # Rounds up the division

    dynamic_height = 300  # Adjust 160px to match your actual row height including gaps
    return dynamic_height


def extract_file_names(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    """
    Extracts file names from a list of Streamlit UploadedFile objects.

    Parameters:
    - uploaded_files (List[st.runtime.uploaded_file_manager.UploadedFile]): List of UploadedFile objects.

    Returns:
    - List[str]: List of file names extracted from uploaded_files.
    """
    file_names = [file.name for file in uploaded_files]
    return file_names


def create_folder_if_not_exists(folder_path: str) -> None:
    """
    Create a folder if it does not already exist.

    Parameters:
    - folder_path (str): The path of the folder to be created.

    Returns:
    - None
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path) 
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def load_lottiefile(filepath: str):
    """
    Load a Lottie animation from a file.

    Parameters:
    - filepath (str): The path to the Lottie animation JSON file.

    Returns:
    - Any: The JSON content of the Lottie animation.
    """

    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    """
    Load a Lottie animation from a URL.
    
    Parameters:
    - url (str): The URL of the Lottie animation JSON file.
    
    Returns:
    - Optional[Dict]: The JSON content of the Lottie animation if the request is successful.
                     Returns None if the request fails (i.e., status code is not 200).
    """
    r = requests.get(url) # Send a GET request to the specified URL
    if r.status_code != 200: # Check if the request was not successful
        return None # Return None if the request failed
    return r.json() # Return the JSON content of the response if successful


     