import io
from fastapi.responses import JSONResponse
from classification import get_classification
from starlette.responses import Response
from typing import List
from fastapi import FastAPI, File
from fastapi import UploadFile
import shutil
import uuid
import sys
import os
import cv2
import csv
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image
sys.path.append(os.path.abspath \
                (os.path.join \
                (os.path.dirname(__file__), '../../../')))
sys.path.append(os.path.abspath \
                (os.path.join \
                (os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath \
                (os.path.join \
                (os.path.dirname(__file__), '../')))

# Assuming 'config' is a dictionary containing configuration details
# for the application.
from utils.load_config import load_config
config_path = "config/serving_config.yaml"
config=load_config(config_path)
IMAGE_UPLOAD_DIRECTORY = config['fastapi']['upload_dir']  + "images"
RESULT_DIRECTORY = config['fastapi']['upload_dir']  + "/results/results.csv"

# Initialize the FastAPI app with a title, description, and version
app = FastAPI(
    title="Image Classification",
    description=f"Visit this URL at port {config['streamlit']['port']} \
                 for the streamlit interface.",
    version=config['model']['version'],
)


def create_folder_if_not_exists(folder_path: str) -> None:
    """
    Check if a folder exists at the given path, and create it if it does not exist.
    
    Parameters:
    folder_path (str): The path to the folder to be checked and created if it doesn't exist.
    
    Returns:
    None
    """
    
    # Check if the folder exists
    if not os.path.exists(folder_path):  # Check if the path does not exist -> bool
        # Create the folder
        os.makedirs(folder_path)  # Create the directory recursively -> None
        print(f"Folder '{folder_path}' created.")  # Print folder creation message -> None
        
    else:
        print(f"Folder '{folder_path}' already exists.")  # Print folder existence message -> None


@app.get(config['fastapi']['root'])
def read_root() -> dict:
    """
    Handler for the root endpoint of the API.

    Returns:
    dict: A dictionary containing a welcome message.
    """
    return {"message": "Welcome from the API"}  # Return a dictionary with a welcome message -> dict



@app.get(config['fastapi']['testapi'])
def test_api() -> dict:
    '''
    Handler for testing the API endpoint.

    Returns:
    dict: A dictionary containing a message from the API.
    '''
    return {"message": "Message from the API"}  # Return a dictionary with a message from the API -> dict


@app.post(config["fastapi"]["classify"])
async def get_prediction(files: List[UploadFile] = File(...)) -> dict:
    '''
    Handler for classifying image files.

    Parameters:
    files (List[UploadFile]): List of image files uploaded through the API.

    Returns:
    dict: A dictionary containing predictions for each uploaded image.
    '''
    # Create necessary directories if they don't exist
    create_folder_if_not_exists(IMAGE_UPLOAD_DIRECTORY )
    create_folder_if_not_exists(config['fastapi']['upload_dir'] + "/results/" )

    list_file_uploads = []    
    list_predictions = []


    # Process each uploaded file
    for file in files:
        """read all file from the inuput and strorage in list of files"""
        file_location = os.path.join(IMAGE_UPLOAD_DIRECTORY, file.filename)
        list_file_uploads.append(file_location)
        # Save the file to disk
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print( {"info": f"{len(files)} files uploaded successfully"})


    # Get predictions for each uploaded file
    for file_location in list_file_uploads:
        """read all file in list file upload and get predictions"""
        prediction =  get_classification(images=file_location,config=config['model'])
        print(f"file: {file_location} : \
                Deseasses: {prediction}")
        

        # Append prediction to CSV file
        with open(RESULT_DIRECTORY, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(([file_location],[prediction]))

        list_predictions.append(prediction)
    
    return  {"predictions": list_predictions}


@app.post("/test_upload_multiple")
async def create_upload_files(files: list[UploadFile] = File(...))  -> dict:
    """
    Endpoint to handle multiple file uploads and save them to a specified directory.

    Args:
        files (list[UploadFile]): List of UploadFile objects received from the client.

    Returns:
        dict: Dictionary containing a list of uploaded file names and a success message.
            Example: {"uploaded_files": ["file1.jpg", "file2.png"], "message": "Files uploaded successfully"}

    Raises:
        HTTPException: If an error occurs during file processing, returns a 500 status code with an error message.
    """
    try:
        # Ensure the upload directory exists
        create_folder_if_not_exists(IMAGE_UPLOAD_DIRECTORY)

        uploaded_files = []

        # Process each uploaded file
        for file in files:
            file_location = f"/mnt/strorage/images{file.filename}"
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer) # Save the file to disk
            uploaded_files.append(file.filename) # Store the uploaded file names


        # Return success message and list of uploaded file names            
        return {"uploaded_files": uploaded_files, "message": "Files uploaded successfully"}
    

    except Exception as e:
        # Return error message if an exception occurs
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    # Run the FastAPI application using Uvicorn
    uvicorn.run(
        "main:app",  # Specify the module where the FastAPI app object (named 'app') is defined
        host=config['fastapi']['host'],  # Host IP address from configuration
        port=config['fastapi']['port'],  # Port number from configuration
        reload=True  # Enable auto-reload for development
    )

