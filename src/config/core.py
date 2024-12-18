# Path setup, and access the config.yml file, datasets folder & trained models
from pydantic import BaseModel

class AppConfig(BaseModel):
    """
    Application-level config.
    """
    pass


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
    pass

class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig