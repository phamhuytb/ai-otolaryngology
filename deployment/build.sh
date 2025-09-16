#!/bin/bash

#build server
echo "build server image"
docker build -t server:v0  -f server.Dockerfile ../

#build UI
echo "build streamlit image"
docker build -t streamlit:v0 -f streamlit.Dockerfile ../

docker images