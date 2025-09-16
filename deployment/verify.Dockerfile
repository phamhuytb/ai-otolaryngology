# Use an official Python runtime as a parent image
# This sets the base image for the Docker container, which includes the necessary runtime and libraries.
FROM hieudinhpro/base_server_ui_deepmed:v2

# Copy the rest of the application code
# This copies the 'verification' directory from the local file system to the '/app/verification' directory in the container.
COPY /verification /app/verification

# Copy the 'utils' directory from the local file system to the '/app/utils' directory in the container.
COPY /utils /app/utils

# Add the configuration file
# This adds the 'serving_config.yaml' file from the local 'config' directory to the '/app/config' directory in the container.
ADD /config/serving_config.yaml /app/config/serving_config.yaml

# Change working directory
# This sets the working directory inside the container to '/app'.
WORKDIR /app

# Define the command to run the application
# This specifies the command to run when the container starts, which is to execute 'main.py' in the 'verification/verifying_server' directory using Python 3.
RUN python3 verification/app/main.py

CMD ["sleep", "3000"]


