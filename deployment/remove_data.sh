#!/bin/bash


docker container stop server

docker container rm server

docker container stop streamlit

docker container rm streamlit

#List volumes:
docker volume ls


#remove data volume
docker volume rm strorage

