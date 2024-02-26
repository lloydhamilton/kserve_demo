from typing import Any

from pydantic import BaseModel, Field
from enum import Enum


class DataTypes(str, Enum):
    BOOL: str = "BOOL"
    INT8: str = "INT8"
    INT16: str = "INT16"
    INT32: str = "INT32"
    INT64: str = "INT64"
    UINT8: str = "UINT8"
    UINT16: str = "UINT16"
    UINT32: str = "UINT32"
    UINT64: str = "UINT64"
    FP32: str = "FP32"
    FP64: str = "FP64"
    BYTES: str = "BYTES"


class InferenceV2Inputs(BaseModel):
    """
    Data Model for payload from Inference V2
    """

    name: str = Field(example="input-0", description="Name of input")
    shape: list[int] = Field(example=[1, 3, 224, 224], description="Shape of input")
    datatype: DataTypes = Field(example="FP32", description="Datatype of input")
    data: list[Any] = Field(example=[[0.0, 0.0, 0.0]], description="Data of input")


class InferenceV2(BaseModel):
    """
    Data Model for payload from Inference V2
    """

    id: str = Field(
        example="2d1781af-3a4c-4d7c-bd0c-e34b19da4e66", description="ID of event."
    )
    inputs: list[InferenceV2Inputs]
