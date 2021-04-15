# Copyright (c) Ioannis Antonopoulos
# Based on tensorflow/tensorflow:2.3.1-gpu-jupyter of TensorFlow Development Team.

# Use the official tensorflow/tensorflow:2.3.1 image as a parent image.
ARG BASE_CONTAINER=tensorflow/tensorflow:2.3.1-gpu-jupyter
FROM $BASE_CONTAINER

LABEL maintainer="Ioannis Antonopoulos <ia46@hw.ac.uk>"

# Install phdTools package which does not have a pip or conda package at the moment
RUN pip install git+https://github.com/antongiannis/phd_tools.git 

# Install Python 3 packages
RUN pip install -U \
pandas==1.1.4 \
seaborn==0.11.0 \
category_encoders==2.2.2 \
scikit-learn==0.23.2 \
statsmodels==0.11.1 \
catboost==0.24.1 \
keras-tuner==1.0.1 \
mc4==2.3.0 \
dython==0.6.1 \
shap==0.36.0


# Change working directory from /tf
RUN rm -r *

# Copy the contents of the notebook folder 
COPY notebooks/* notebooks/

# Create the data directory
RUN mkdir notebooks/data

# Copy the utils folder
COPY utils/* utils/