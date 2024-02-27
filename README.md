# Deploying a model using KServe

## Prerequisites
You will need to have a cluster with kserve installed. If you do not have a cluster, you can create one by following 
the guide [here](https://medium.com/towards-data-science/kserve-highly-scalable-machine-learning-deployment-with-kubernetes-aa7af0b71202)

## Python package management
This project uses `poetry` for package management. 
There is also a `requirements.txt` file for those who prefer to use `pip`.

```shell
poetry install
```
## Introduction
This repository contains all the necessary files to deploy a model using kserve. It implements:

1. Continuous integration using GitHub Actions
2. Example unit tests
3. Example integration tests
2. Pre-commit hooks for code quality
3. Pre-commit hooks for poetry `requirements.txt` exports
4. A dummy model wrapped in KServe custom predictor class.
5. Notebook to demonstrate how to deploy the model
