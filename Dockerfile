# Dockerfile

# pull the official docker image
FROM python:3.11.3-slim

# set work directory
WORKDIR /app

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# copy project
COPY . .