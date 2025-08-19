from pydantic import BaseModel, validator
from typing import List

EXPECTED_FEATURES = 14

class SinglePredictionRequest(BaseModel):
    # flat list: [f1, f2, ..., f14]
    features: List[float]

    @validator("features")
    def check_length(cls, v):
        if len(v) != EXPECTED_FEATURES:
            raise ValueError(f"features must contain exactly {EXPECTED_FEATURES} values")
        return v

class BatchPredictionRequest(BaseModel):
    # list of rows: [[...14 floats...], [...14 floats...]]
    features: List[List[float]]

    @validator("features")
    def check_batch_length(cls, v):
        for row in v:
            if len(row) != EXPECTED_FEATURES:
                raise ValueError(f"Each feature row must contain exactly {EXPECTED_FEATURES} values")
        return v

class PredictionResponse(BaseModel):
    predicted_label: str
    predicted_class_index: int
    probabilities: List[float]
