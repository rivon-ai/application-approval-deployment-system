# Schema for prediction requests and responses
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float

class PredictionResponse(BaseModel):
    prediction: float
