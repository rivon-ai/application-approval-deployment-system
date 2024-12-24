# Path setup, and access the config.yml file, datasets folder & trained models
from pydantic import BaseModel
import yaml
from typing import List

class AppConfig(BaseModel):
    """
    Application-level config.
    """
    data_file: str
    save_model_directory: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
    n_neighbors: int
    model_name: str
    pipeline_name: str
    file_extension: str
    test_size: float
    random_state: int

class Features(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
    num_vars: List[str]
    cat_vars: List[str]
    unused_fields: List[str]

class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    mode_config: ModelConfig
    features: Features

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Config = self.load_config()

    def load_config(self) -> Config:
        """Load and validate the configuration from the YAML file."""
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)

        return Config(**config_data)
    