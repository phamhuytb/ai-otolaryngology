# Use an official Python runtime as a parent image
# This sets the base image for the Docker container, which includes the necessary runtime and libraries.
FROM hieudinhpro/base_server_ui_deepmed:v2

# Copy the 'streamlit' directory from the local file system to the '/app/streamlit' directory in the container.
COPY /streamlit /app/streamlit

# Copy the 'utils' directory from the local file system to the '/app/utils' directory in the container.
COPY /utils /app/utils

# Add the 'serving_config.yaml' file from the local 'config' directory to the '/app/config' directory in the container.
ADD /config/serving_config.yaml /app/config/serving_config.yaml

# Change working directory
# This sets the working directory inside the container to '/app'.
WORKDIR /app

# Expose the application port
# This exposes port 8501 on the container, allowing external access to this port.
EXPOSE 8501

# Define the command to run the application
# This specifies the command to run when the container starts, which is to execute 'main.py' in the 'streamlit/ui' directory using Streamlit, and sets the server port to 8501.
CMD ["streamlit", "run", "streamlit/ui/main.py", "--server.port", "8501"]
