# pipeline.py file contains code of the pipeline
# pipeline contains missing value imputation and encoding, scaling and outliers removal, feature selection and model

from sklearn.pipeline import Pipeline
from config.core import ConfigLoader
from processing.features import MissValImputer, CategoricalEncoder, OutlierHandler, StandardScalerCustom


# Loading Configuration
config_path = "config.yml"
path = ConfigLoader(config_path)

def build_pipeline() -> Pipeline:
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
    pipeline_steps = [
        ('knn_imputer', MissValImputer(numerical_cols=path.config.features.num_vars, categorical_cols=path.config.features.cat_vars, n_neighbors=path.config.mode_config.n_neighbors)),
        ('ohe', CategoricalEncoder(categorical_cols=path.config.features.cat_vars)),
        ('outlier_imputer', OutlierHandler(method='median')),
        ('scaler', StandardScalerCustom(exclude_column=path.config.features.unused_fields))

    ]
    return Pipeline(steps=pipeline_steps)