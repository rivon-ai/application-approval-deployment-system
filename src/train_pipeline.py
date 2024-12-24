# train_pipeline.py file contains code of the training pipeline
from config.core import ConfigLoader
from processing.data_manager import load_dataset, save_pipeline
from pipeline import build_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


config_path = "config.yml"
path = ConfigLoader(config_path)  

def run_training() -> None:
    print("Training the model")

    # Load the Dataset
    print("Loading the dataset...")
    df = load_dataset(file_name=path.config.app_config.data_file)
    print("Dataset loaded")

    # Feature Selection Numerical and Categorical
    df = df[path.config.features.cat_vars + path.config.features.num_vars]

    # Buidling the pipeline
    print("Building the pipeline...")
    pipeline = build_pipeline()
    print("Pipeline built")
    # print pipeline
    print(pipeline)

    # Fitting the pipeline
    print("Fitting the pipeline...")
    df_transformed = pipeline.fit_transform(df)
    print("Pipeline fitted")

    # Split data into features (X) and target (y)
    X = df_transformed.drop(columns=path.config.features.unused_fields)
    y = df_transformed[path.config.features.unused_fields]

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=path.config.mode_config.test_size, 
                                                        random_state=path.config.mode_config.random_state)

    # Logestistic Regression Model
    print("Training the model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Model trained")

    # Predict and Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Save the pipeline
    print("Saving the pipeline...")
    save_pipeline(pipeline_to_persist=pipeline, model_to_persist=model)
    print("Pipeline saved")


if __name__ == "__main__":
    run_training()