# pipeline.py file contains code of the pipeline
# pipeline contains missing value imputation and encoding, scaling and outliers removal, feature selection and model

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from .processing.features import FeatureTransformer
from src.config import Config


def build_pipeline(config: Config) -> Pipeline:
    """
    Build the ML pipeline with preprocessing and model.
    The pipeline includes:
    - Imputation of missing values
    - Outlier removal
    - Feature scaling
    - Encoding of categorical variables
    - Logistic Regression model for training

    Args:
        config (Config): The configuration object containing model and preprocessing settings.

    Returns:
        Pipeline: A scikit-learn pipeline object.
    """
    # Define feature transformer
    feature_transformer = FeatureTransformer(
        num_vars=config.model_config.num_vars,
        cat_vars=config.model_config.cat_vars,
        exclude_column=config.model_config.exclude_column,
        n_neighbors=config.model_config.n_neighbors
    )
    
    # Define the pipeline
    pipeline_steps = [
        ('preprocessor', feature_transformer.create_pipeline()),
        ('classifier', LogisticRegression(**config.model_config.model_params))
    ]
    
    return Pipeline(steps=pipeline_steps)