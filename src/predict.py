# predict.py file contains code of the prediction

from typing import Union
import pandas as pd
import joblib
from config.core import ConfigLoader
import os


config_path = "config.yml"
path = ConfigLoader(config_path)


def make_prediction(*, input_data:Union[pd.DataFrame, dict], error=None) -> dict:
    """
    Make predictions using a pre-trained pipeline model.

    Args:
        input_data (Union[pd.DataFrame, dict]): The input data to make predictions on.
        config (Config): The configuration object containing model and preprocessing details.

    Returns:
        dict: A dictionary with the prediction results.
    """

    model_directory = path.config.app_config.save_model_directory
    model_name = path.config.mode_config.model_name
    pipeline_name = path.config.mode_config.pipeline_name
    model_path = os.path.join(model_directory, model_name)
    pipeline_path = os.path.join(model_directory, pipeline_name)

    # Load the saved model (ensure the correct file path)
    try:
        if os.path.exists(model_path):
            loaded_model = joblib.load(model_path)
            print("Model loaded successfully.")
    except Exception as e:
        return {"error": f"Error loading model: {str(e)}"}

    # Load the pre-processing pipeline (replace with your actual full_pipeline object)
    try:
        if os.path.exists(pipeline_path):
            full_pipeline = joblib.load(pipeline_path)
            print("Pipeline loaded successfully.")
    except Exception as e:
        return {"error": f"Error loading pipeline: {str(e)}"}
    
    input_df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data

    # Transform the data using the pipeline
    input_transformed = full_pipeline.transform(input_df)

    # Make prediction using the trained model
    try:
        prediction = loaded_model.predict(input_transformed)
    except Exception as e:
        return {"error": f"Error during prediction: {str(e)}"}

    # Return the prediction result in a dictionary
    return {"predictions": prediction.tolist()}