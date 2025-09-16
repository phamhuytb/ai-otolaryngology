#!/bin/bash

#run server
echo "AI Server starting ..."
docker run  \
  --name server \
  -d \
  -v  strorage:/mnt/strorage \
  -p 8000:8000 hieudinhpro/ai_server_deepmed_ent:v0

#5s loading model
sleep 10

#run UI
echo "UI service stating ..."
docker run  \
  --name streamlit \
  -d \
  -v  strorage:/mnt/strorage \
  -p 8501:8501 streamlit:v0 

#startting all servicess in the background (detached mode)
docker-compose up -d


