# made additonal predict.py to handle prediction logic
import pickle
from .schemas.predict import PredictionRequest

# Example of loading a trained model (assumed to be saved as a pickle file)
MODEL_PATH = "path_to_trained_model/model.pkl"

async def predict_model(request: PredictionRequest):
    # Load the model if not already loaded (you can optimize by loading the model once)
    model = load_model()
    features = [request.feature1, request.feature2, request.feature3]
    prediction = model.predict([features])
    return prediction[0]

def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model
