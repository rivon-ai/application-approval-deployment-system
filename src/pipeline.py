import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.config.core import config
from src.processing.features import MissValImputer
from src.processing.features import OutlierHandler
from src.processing.features import CategoricalEncoder

lac_pipe = Pipeline([
    
    ("miss_val_imputer", MissValImputer(config.modelConfig.num_vars,
                                        config.modelConfig.cat_vars)
     ),
     ##==========Mapper======##
     ("outlier_handler", OutlierHandler(config.modelConfig.num_vars)
      ),
     ##==========Encoding======##
     ("categorical_encoder", CategoricalEncoder(
                                                config.modelConfig.cat_vars
                                                )
     ),
    # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestClassifier(
                                        n_estimators=config.modelConfig.n_estimators, 
                                        max_depth=config.modelConfig.max_depth,
                                        random_state=config.modelConfig.random_state
                                        ))   
     ])

