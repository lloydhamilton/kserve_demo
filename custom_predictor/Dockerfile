FROM python:3.10 AS build
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r ./requirements.txt

COPY custom_predictor/custom_predictor.py ./
COPY custom_predictor/dummy_model.py ./

ENTRYPOINT ["python", "custom_predictor.py"]
