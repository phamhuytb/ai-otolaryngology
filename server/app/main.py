import io
import os
import shutil
import sys
import uuid
import csv
from typing import List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.responses import Response

from classification import get_classification
from utils.load_config import load_config

# Update system path to include custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Load configuration
config_path = "config/serving_config.yaml"
config = load_config(config_path)
IMAGE_UPLOAD_DIRECTORY = config['fastapi']['upload_dir'] + "images"
RESULT_DIRECTORY = config['fastapi']['upload_dir'] + "/results/results.csv"

# Initialize the FastAPI app
app = FastAPI(
    title="Image Classification",
    description=f"Visit this URL at port {config['streamlit']['port']} for the streamlit interface.",
    version=config['model']['version'],
)


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


@app.get(config['fastapi']['root'])
def read_root() -> dict:
    """Handler for the root endpoint of the API.

    Returns:
        dict: A dictionary containing a welcome message.
    """
    return {"message": "Welcome from the API"}


@app.get(config['fastapi']['testapi'])
def test_api() -> dict:
    """Handler for testing the API endpoint.

    Returns:
        dict: A dictionary containing a message from the API.
    """
    return {"message": "Message from the API"}


@app.post(config["fastapi"]["classify"])
async def get_prediction(files: List[UploadFile] = File(...)) -> dict:
    """Handler for classifying image files.

    Args:
        files (List[UploadFile]): List of image files uploaded through the API.

    Returns:
        dict: A dictionary containing predictions for each uploaded image.
    """
    create_folder_if_not_exists(IMAGE_UPLOAD_DIRECTORY)
    create_folder_if_not_exists(config['fastapi']['upload_dir'] + "/results/")

    list_file_uploads = []
    list_predictions = []

    for file in files:
        file_location = os.path.join(IMAGE_UPLOAD_DIRECTORY, file.filename)
        list_file_uploads.append(file_location)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print({"info": f"{len(files)} files uploaded successfully"})

    for file_location in list_file_uploads:
        prediction = get_classification(images=file_location, config=config['model'])
        print(f"file: {file_location} : Diseases: {prediction}")

        with open(RESULT_DIRECTORY, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(([file_location], [prediction]))

        list_predictions.append(prediction)

    return {"predictions": list_predictions}


@app.post("/test_upload_multiple")
async def create_upload_files(files: List[UploadFile] = File(...)) -> dict:
    """Endpoint to handle multiple file uploads and save them to a specified directory.

    Args:
        files (List[UploadFile]): List of UploadFile objects received from the client.

    Returns:
        dict: Dictionary containing a list of uploaded file names and a success message.

    Raises:
        HTTPException: If an error occurs during file processing, returns a 500 status code with an error message.
    """
    try:
        create_folder_if_not_exists(IMAGE_UPLOAD_DIRECTORY)

        uploaded_files = []

        for file in files:
            file_location = f"/mnt/storage/images/{file.filename}"
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file.filename)

        return {"uploaded_files": uploaded_files, "message": "Files uploaded successfully"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config['fastapi']['host'],
        port=config['fastapi']['port'],
        reload=True
    )
