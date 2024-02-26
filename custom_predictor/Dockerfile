FROM python:3.10 AS build
RUN groupadd -r pythongroup && useradd -r -g pythongroup pythonuser -u 999
WORKDIR /app
COPY custom_predictor/requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r ./requirements.txt

COPY custom_predictor/custom_predictor.py ./
COPY custom_predictor/dummy_model.py ./
USER pythonuser
ENTRYPOINT ["python", "custom_predictor.py"]
