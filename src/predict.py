# predict.py file contains code of the prediction

from typing import Union
import pandas as pd


def make_prediction(*, input_data:Union[pd.DataFrame, dict], error=None) -> dict:
    return {"predictions": "predictions"}