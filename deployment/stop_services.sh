#!/bin/bash

#Stop the container and remove the volume. Note volume removal is a separate step.
docker container stop server server:v0

docker container stop streamlit streamlit:v0

