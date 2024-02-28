# Deploying a model using KServe

## Prerequisites
You will need to have a cluster with kserve installed. If you do not have a cluster, you can create one by following 
the guide [here](https://medium.com/towards-data-science/kserve-highly-scalable-machine-learning-deployment-with-kubernetes-aa7af0b71202)

* Docker
* Poetry
* Python 3.10
* kubectl
* pre-commit

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
4. Pre-commit hooks for code quality
5. Pre-commit hooks for poetry `requirements.txt` exports
6. A dummy model wrapped in KServe custom predictor class.
7. Notebook to demonstrate how to deploy the model

## Developing

To build the docker container use:
    
```shell
docker build -f custom_predictor/Dockerfile -t demokserve .
```

Then run the container with:

```shell
docker run -p 8080:8080 demokserve
```

You can send a request to the model using the following code:

```python
import requests
import base64
import io
import uuid
from PIL import Image
from src.data_models import InferenceV2Inputs, InferenceV2

def bytes_to_json_serializable(bytes_data: bytes) -> str:
    return base64.b64encode(bytes_data).decode("utf-8")

# read pil image to bytes
with io.BytesIO() as output:
    with Image.open("cat.jpg") as img:
        img.save(output, format="PNG")
        image_size = img.size
    image_bytes_data = bytes_to_json_serializable(output.getvalue())

inputs = InferenceV2Inputs(
    name="input-0",
    shape=list(image_size),
    datatype="BYTES",
    data=[image_bytes_data]
)

inference_request_payload = InferenceV2(
    id=str(uuid.uuid4()),
    inputs=[inputs]
)

model_name = "kserve-demo-model"
url = f"http://localhost:8080/v2/models/{model_name}/infer"
request_headers = {"Host": f"{model_name}.kserve.example.com"}

response = requests.post(url, data=inference_request_payload.json(), headers=request_headers)
```
