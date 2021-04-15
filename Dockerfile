# Copyright (c) Ioannis Antonopoulos
# Based on tensorflow/tensorflow:2.4.1-gpu-jupyter of TensorFlow Development Team.

# Use the official tensorflow/tensorflow:2.4.1 image as a parent image.
ARG BASE_CONTAINER=tensorflow/tensorflow:2.4.1-gpu-jupyter
FROM $BASE_CONTAINER

LABEL maintainer="Ioannis Antonopoulos <ia46@hw.ac.uk>"

# Install phdTools package which does not have a pip or conda package at the moment
RUN pip install git+https://github.com/antongiannis/phd_tools.git 

# Install Python 3 packages
RUN pip install -U \
pandas==1.1.5 \
seaborn==0.11.1 \
scikit-learn==0.24.1


# Change working directory from /tf and create the data directory
RUN rm -r * && mkdir data

# Copy the contents of the notebook folder 
COPY notebooks/* notebooks/

# Copy the utils folder
COPY utils/* utils/