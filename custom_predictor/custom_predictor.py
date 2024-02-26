import base64
import io
import logging

import numpy as np
import torch
from PIL import Image

from kserve.utils.utils import generate_uuid
from kserve import (
    InferInput,
    InferOutput,
    InferRequest,
    InferResponse,
    Model,
    ModelServer,
)
import kserve.constants as constants

from dummy_model import DummyModel

logging.basicConfig(level=constants.KSERVE_LOGLEVEL)
log = logging.getLogger(__name__)


class CustomPredictor(Model):
    """
    Custom predictor class for demo purposes
    """

    def __init__(self, name: str) -> None:
        """Initializes the CustomPredictor.

        Args:
            name: The name of the model.
        """
        super().__init__(name)
        self.model = None
        self.name = name
        self.ready = False
        self.load_model()

    def load_model(self) -> None:
        """
        Loads the dummy model from the file system.
        """
        self.model = DummyModel()
        self.ready = True

    @staticmethod
    def reshape_images(
        image: bytearray | str, height: int = 360, width: int = 640
    ) -> np.array:
        """
        Reads image from byte array and reshapes it to the correct dimensions.

        Args:
            image (bytearray): Image in byte array format
            height (int, optional): Height of the image. Defaults to 360.
            width (int, optional): Width of the image. Defaults to 640.

        Returns:
            reshaped_image (np.array): Reshaped image in numpy array format
        """
        if isinstance(image, str):
            image = base64.b64decode(image)

        image = Image.open(io.BytesIO(image))
        resized_image = image.resize((width, height), resample=Image.LANCZOS)

        # Convert to numpy array and normalize
        np_image = np.array(resized_image)
        normalized_image = np_image.astype(np.float32) / 255.0

        # Transpose and reshape the image
        transposed_image = normalized_image.transpose((2, 0, 1))
        reshaped_image = transposed_image.reshape((1,) + transposed_image.shape)

        return reshaped_image

    def preprocess(
        self, payload: dict | InferRequest, headers: dict[str, str] = None
    ) -> InferRequest:
        log.info(f"Headers: {headers}")
        if "skip_preprocessing" in headers:
            return payload
        logging.info(f"Received request with {len(payload.inputs)} images")
        input_tensors = [
            self.reshape_images(inputs.data[0]) for inputs in payload.inputs
        ]
        self.input_tensor = input_tensors
        input_names = [inputs.name for inputs in payload.inputs]

        processed_inputs = []
        for input_tensor, input_names in zip(input_tensors, input_names):
            logging.info(f"Input tensor shape: {input_tensor.shape}")
            infer_input = InferInput(
                name=f"input_{input_names}",
                datatype="FP32",
                shape=list(input_tensor.shape),
                data=input_tensor.flatten().tolist(),
            )
            processed_inputs.append(infer_input)
        infer_request = InferRequest(
            model_name=self.name, infer_inputs=processed_inputs
        )
        return infer_request

    def predict(
        self, payload: InferRequest, headers: dict[str, str] = None
    ) -> InferResponse:
        """
        This function is called by KFServing to perform inference on a single request.

        Args:
            payload: The input data for inference.
            headers: The headers of the request.
                Defaults to None.

        Returns:
            InferResponse: The output confidence map for inference.
        """
        log.info(f"Reading data with length: {len(payload.inputs)}")

        inference_output = []
        for idx, single_input in enumerate(payload.inputs):
            input_image = np.array(single_input.data).astype(np.float32)
            input_image = input_image.reshape(single_input.shape)
            log.info(f"Input shape: {input_image.shape}")

            log.info(f"Reading data...{input_image.shape}")
            torch_device = torch.device("cpu")
            image_tensor = torch.from_numpy(input_image).to(torch_device)

            log.info("Predicting...")
            with torch.no_grad():
                prediction = self.model(image_tensor)
                prediction = np.squeeze(prediction.cpu().numpy())

            prediction_flat = prediction.flatten().tolist()
            inference_output.append(
                InferOutput(
                    name=f"output-{idx}",
                    shape=list(prediction.shape),
                    datatype="FP32",
                    data=prediction_flat,
                )
            )

        response_id = generate_uuid()
        parameters = {
            "model_name": self.name,
            "model_version": 1,
            "model_registry_name": "demo-model",
        }
        infer_response = InferResponse(
            model_name=self.name,
            infer_outputs=inference_output,
            response_id=response_id,
            parameters=parameters,
        )

        return infer_response

    def postprocess(
        self, response: InferResponse, headers: dict[str, str] = None
    ) -> InferResponse:
        """Postprocess step for the inference request.

        If the return-masked-image header is set to true, apply the mask
        on the image and return the masked image.

        Args:
            response (InferResponse): The response from the inference request.
            headers (Dict[str, str], optional): The headers from the
                inference request. Defaults to None.

        Returns:
            InferResponse: The response from the inference request.
        """
        if headers.get("post-process") == "true":
            logging.info("Post processing")

        return response


if __name__ == "__main__":
    model = CustomPredictor("kserve-demo-model")
    ModelServer().start([model])
