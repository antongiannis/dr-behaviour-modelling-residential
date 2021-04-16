## Repo structure
This repository includes the python scripts and notebooks for reproducing the modelling and analysis included in the paper *Data-driven modelling of energy demand response behaviour based on a large-scale residential trial*, published in Energy & AI.

The `notebooks` directory includes the various Jupyter notebooks. The exploration notebooks need to be run first as they produce files that are used for the subsequent analysis and modelling.
To be able to run the temporal learning notebook you need to run the `exploration-response-dataset` notebook. The modelling of demand response behaviour is in the `modelling-dr-behaviour` notebook.

## Run notebooks using Docker
The easiest way to run the code and the Jupyter notebooks is by building a docker container based on the instructions in the `Dockerfile`. You need to follow the following steps:
1. **Build** the docker image from the `Dockerfile` in the github repo using the following command:
```bash
docker build --tag dr_behaviour https://github.com/antongiannis/dr-behaviour-modelling-residential.git#main
```
2. **Run** the docker container you by typing in the terminal the following command: 
```bash
docker run -p 8888:8888 dr_behaviour
```


