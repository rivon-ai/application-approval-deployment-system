# Path setup, and access the config.yml file, datasets folder & trained models
from pydantic import BaseModel
import yaml
from typing import Any

class AppConfig(BaseModel):
    """
    Application-level config.
    """
    package_name: str
    pipeline_name: str
    pipeline_save_file: str
    data_file: str
    logging_level: str
    log_file: str
    
    @classmethod
    def from_yaml(cls, config_dict: dict) -> 'AppConfig':
        return cls(
            package_name=config_dict.get('package_name', 'src'),
            pipeline_name=config_dict.get('pipeline_name', 'loan_approval_classifier'),
            pipeline_save_file=config_dict.get('pipeline_save_file', 'loan_approval_classifier_model_output_v1'),
            data_file=config_dict.get('data_file', 'datasets/data.csv'),
            logging_level=config_dict.get('logging', {}).get('level', 'INFO'),
            log_file=config_dict.get('logging', {}).get('log_file', 'logs/pipeline_execution.log')
        )


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
    num_vars: list
    cat_vars: list
    unused_fields: list
    data_preprocessing: dict
    model_type: str
    model_params: dict

    @classmethod
    def from_yaml(cls, config_dict: dict) -> 'ModelConfig':
        return cls(
            num_vars=config_dict.get('features', {}).get('num_vars', []),
            cat_vars=config_dict.get('features', {}).get('cat_vars', []),
            unused_fields=config_dict.get('features', {}).get('unused_fields', []),
            data_preprocessing=config_dict.get('data_preprocessing', {}),
            model_type=config_dict.get('model', {}).get('type', 'LogisticRegression'),
            model_params=config_dict.get('model', {}).get('algorithm_params', {})
        )

class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig

    @classmethod
    def from_yaml(cls, yaml_file_path: str) -> 'Config':
        with open(yaml_file_path, 'r') as file:
            config_dict = yaml.safe_load(file)

        app_config = AppConfig.from_yaml(config_dict)
        model_config = ModelConfig.from_yaml(config_dict)

        return cls(app_config=app_config, model_config=model_config)

# Example usage:
if __name__ == "__main__":
    config = Config.from_yaml("src/config.yml")
    print(config.app_config)
    print(config.model_config)