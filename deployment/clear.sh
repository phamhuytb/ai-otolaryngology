#!/bin/bash

docker stop $(docker ps -q)

docker rmi streamlit:v0 streamlit server server:v0

docker rm -v -f $(docker ps -qa)

docker system prune -a
# docker rm -v -f $(docker ps -qa)
