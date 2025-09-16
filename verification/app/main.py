import streamlit as st
import requests
from os import listdir
from math import ceil
import os
from PIL import Image 
import base64
import shutil
import sys

# Update system path to include custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.load_config import load_config
from test_utils import verifying_server

config_path = "config/serving_config.yaml"
config=load_config(config_path)

def main():
    """
    Main function to test the API endpoint with a sample file.

    Returns:
        The result of the server verification.
    """
    # Construct the URL of the FastAPI server
    api_url = (
        f"{config['fastapi']['protocol']}://"
        f"{config['streamlit']['host']}:"
        f"{config['fastapi']['port']}"
        f"{config['fastapi']['classify']}"
    )
    
    # Path to the test file
    test_file_path = "test.jpg"
    
    try:
        # Verify the server with the test file
        test_result = verifying_server(api_url=api_url, file_path=test_file_path)
        return test_result
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    result = main()
    if result is not None:
        print("Test result:", result)
    else:
        print("Failed to get a result.")

