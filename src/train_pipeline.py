# train_pipeline.py file contains code of the training pipeline
import pandas as pd
from config.core import ConfigLoader
from processing.data_manager import load_dataset, save_pipeline, get_user_input
from pipeline import build_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from processing.validation import validate_inputs
from predict import make_prediction


config_path = "./config.yml"
path = ConfigLoader(config_path)  

def run_training() -> None:
    print("Training the model")

    # Load the Dataset
    print("Loading the dataset...")
    df = load_dataset(file_name=path.config.app_config.data_file)
    print("Dataset loaded")

    # Define the columns to include
    columns_to_include = list(set(path.config.features.cat_vars + path.config.features.num_vars))
    print(columns_to_include)
    # Select the features (X) and target (y)
    X = df[columns_to_include]  # Correct way to select columns
    y = df[path.config.features.unused_fields]  # Assuming this is the target column

    # Buidling the pipeline
    print("Building the pipeline...")
    pipeline = build_pipeline()
    print("Pipeline built")
    # print pipeline
    print(pipeline)
    
    # Fitting the pipeline
    print("Fitting the pipeline...")
    df_transformed = pipeline.fit_transform(X)
    print("Pipeline fitted")

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(df_transformed, y, test_size=path.config.mode_config.test_size, 
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
    print("Training pipeline completed.")

    # function for User Input
    print("Getting user input...")
    user_input = get_user_input()

    # Convert user input into a DataFrame
    user_input_df = pd.DataFrame([user_input])

    # Validate inputs using the validation function
    print("Validating inputs...")
    validated_df, validation_errors = validate_inputs(input_df=user_input_df)
    print(validated_df)
    if validation_errors:
        print("Validation Errors:", validation_errors)
    else:
        print("No validation errors found.")
        print("Making prediction...")
        result = make_prediction(input_data=validated_df)
        print("Prediction Result:", result)


if __name__ == "__main__":
    run_training()