# predict.py file contains code of the prediction

from typing import Union
import pandas as pd
from src.processing.data_manager import load_dataset
from src.config import Config
from src.pipeline import build_pipeline


def make_prediction(*, input_data:Union[pd.DataFrame, dict], error=None) -> dict:
    """
    Make predictions using a pre-trained pipeline model.

    Args:
        input_data (Union[pd.DataFrame, dict]): The input data to make predictions on.
        config (Config): The configuration object containing model and preprocessing details.

    Returns:
        dict: A dictionary with the prediction results.
    """
    # Load and preprocess the input data
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    input_data = load_dataset(file_name=input_data)
    
    # Build the model pipeline
    pipeline = build_pipeline(Config)
    
    # Make predictions
    predictions = pipeline.predict(input_data)
    return {"predictions": predictions.tolist()}