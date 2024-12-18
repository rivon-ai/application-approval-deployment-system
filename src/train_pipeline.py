# train_pipeline.py file contains code of the training pipeline
from .processing.data_manager import load_dataset, split_data, save_pipeline, save_model, remove_old_pipelines
from .processing.features import build_pipeline
from .config import Config
from sklearn.metrics import accuracy_score


def train_model(X_train, y_train, pipeline):
    """
    Train the model using the provided training data and pipeline.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        pipeline (Pipeline): The scikit-learn pipeline.
    """
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    print("Model trained successfully.")

def evaluate_model(X_test, y_test, pipeline):
    """
    Evaluate the trained model using the test data and pipeline.

    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        pipeline (Pipeline): The trained scikit-learn pipeline.
    """
    # Make predictions using the test data
    y_pred = pipeline.predict(X_test)

    # Calculate accuracy or any other metric you need
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

def run_training() -> None:
    print("Training the model")

    # Load the dataset
    print("Loading the dataset ...")
    df = load_dataset()
    print("Dataset loaded successfully.")

    # build the pipeline
    print("Building the pipeline ...")
    pipeline = build_pipeline(config=Config)
    print("Pipeline built successfully.")

    # apply the pipeline to the dataset (Fit and Transform in one step)
    print("Applying the pipeline to the entire dataset ...")
    pipeline.fit_transform(df)
    print("Pipeline applied successfully.")

    # save the pipeline
    print("Saving the pipeline ...")
    save_pipeline(pipeline_to_persist=pipeline)
    print("Pipeline saved successfully.")

    # remove old pipelines
    print("Removing old pipelines ...")
    remove_old_pipelines(files_to_keep=["trained_model.pkl"])
    print("Old pipelines removed successfully.")

    # Split the data into train and test sets
    print("Splitting the data into train and test sets ...")
    X_train, X_test, y_train, y_test = split_data(df=df)
    print("Data split successfully.")

    # Train the model
    print("Training the model ...")
    train_model(X_train=X_train, y_train=y_train, pipeline=pipeline)

    # Save the trained model
    print("Saving the model ...")
    save_model(model_to_persist=pipeline)
    print("Model saved successfully.")

    # Evaluate the model
    print("Evaluating the model ...")
    evaluate_model(X_test=X_test, y_test=y_test, pipeline=pipeline)
    print("Model evaluated successfully.")

    # Remove old models
    print("Removing old models ...")
    remove_old_pipelines(files_to_keep=["trained_model.pkl"])
    print("Old models removed successfully.")

    print("Training completed successfully.")
 