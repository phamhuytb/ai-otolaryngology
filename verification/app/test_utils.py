import requests 

def  verifying_server (api_url: str, file_path:str) -> dict:
    """
    Send a file to the specified URL and return the server's response.

    Args:
        url (str): The URL to which the file will be sent.
        file_path (str): The path of the file to be sent.

    Returns:
        dict: The JSON response from the server.
    """
    try:
        # Open the file in binary read mode
        with open(file_path, 'rb') as f:
            files = {'files': (file_path, f, 'image/jpeg')}  # Prepare the file for upload
            headers = {
                'accept': 'application/json',  # Accept JSON responses
            }
            # Send POST request to the server
            response = requests.post(api_url, files=files, headers=headers)

        # Print the response for debugging purposes
        print(response)

        # Return the JSON response from the server
        return response.json()

    except Exception as e:
        # Handle exceptions and return an error message
        return {"error": str(e)}


 

