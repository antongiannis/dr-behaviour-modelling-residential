# Copyright (c) Ioannis Antonopoulos
# Based on tensorflow/tensorflow:2.4.1-gpu-jupyter of TensorFlow Development Team.

# Use the official tensorflow/tensorflow:2.4.1 image as a parent image.
ARG BASE_CONTAINER=tensorflow/tensorflow:2.4.1-gpu-jupyter
FROM $BASE_CONTAINER

LABEL maintainer="Ioannis Antonopoulos <anton.ioannis.phys@gmail.com>"


# Install phdTools package which does not have a pip or conda package at the moment
RUN pip install git+https://github.com/antongiannis/phd_tools.git 

# Change working directory from /tf
/bin/bash -c #(nop)  CMD ["bash" "-c" "rm -r *"]

# Copy the contents of the notebook folder 
COPY notebooks/* notebooks/

