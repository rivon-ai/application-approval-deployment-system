 # Core logic for handling API endpoints
from fastapi import APIRouter
from .schemas.predict import PredictionRequest, PredictionResponse
from .predict import predict_model

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Call the prediction function
    prediction = await predict_model(request)
    return PredictionResponse(prediction=prediction)


# Endpoint for health check
@router.get("/health")
def health_check():
    return {"status": "Healthy"}