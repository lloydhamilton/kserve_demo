FROM python:3.10 AS build
WORKDIR /app
COPY requirements.txt ./

COPY infra/predictors/segmentation_predictor.py ./
COPY infra/predictors/app_config.ini ./

ENTRYPOINT ["python", "segmentation_predictor.py"]
