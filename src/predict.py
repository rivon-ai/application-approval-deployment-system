# predict.py file contains code of the prediction

from typing import Union
import pandas as pd


def make_prediction(*, input_data:Union[pd.DataFrame, dict], error=None) -> dict:
    """
    Make predictions using a pre-trained pipeline model.

    Args:
        input_data (Union[pd.DataFrame, dict]): The input data to make predictions on.
        config (Config): The configuration object containing model and preprocessing details.

    Returns:
        dict: A dictionary with the prediction results.
    """
    return {"predictions": predictions.tolist()}