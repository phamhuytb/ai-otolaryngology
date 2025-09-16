# Use the official Python 3.9 slim image as the base image
FROM python:3.9-slim

# Install necessary packages
# Update the package list and install curl, ffmpeg, libsm6, libxext6, bash, wget, nano, git, and build-essential
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    bash \
    wget \
    nano \
    git \
    build-essential \
    # Clean up to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the requirements.txt file from the local 'server' directory to the 'app' directory in the container
COPY /server/requirements.txt app/requirements.txt

# Change working directory to '/app'
WORKDIR /app

# Install Python dependencies from the requirements.txt file
RUN pip install -r requirements.txt

# Define the default command to run when the container starts
CMD ["echo", "running okay"]
