# Application Approval Deployment System

A comprehensive system for automating the deployment of a machine learning classifier for application approval. This project uses a CI/CD pipeline, Docker, and FastAPI to provide a robust, scalable, and efficient deployment framework.

Features
- End-to-End Machine Learning Pipeline:
    - Train, evaluate, and deploy an ML model for application approval.
- API Deployment:
    - A FastAPI-based RESTful API to serve predictions.
- Containerization:
    - Dockerized environment for consistent and portable deployment.
- CI/CD Pipeline:
    - GitHub Actions workflow for automated testing and deployment.
- Cloud Deployment:
    - Ready for deployment on cloud platforms like AWS.

Project Structure
```
Application-Approval-Deployment-System/
├── Dockerfile                          # Defines the Docker image setup for the project
├── README.md                           # Project documentation
├── requirements/                       # Directory for dependency files
│   ├── requirements.txt                # Core dependencies for running the project
├── src/                                # Core source code for the project
│   ├── __init__.py                     # Initialization file for the src module
│   ├── config/                         # Configuration scripts for the project
│   │   └── core.py                     # Core configurations (e.g., environment variables, paths)
│   ├── config.yml                      # YAML configuration file for the pipeline
│   ├── datasets/                       # Directory containing datasets for training and testing
│   ├── notebooks/                      # Jupyter notebooks for exploratory data analysis and prototyping
│   ├── pipeline.py                     # Orchestrates the ML pipeline (data, training, evaluation)
│   ├── predict.py                      # Script to load the model and perform predictions
│   ├── processing/                     # Directory for data preprocessing and validation
│   │   ├── __init__.py                 # Initialization file for the processing module
│   │   ├── data_manager.py             # Manages data loading and saving
│   │   ├── features.py                 # Feature engineering logic
│   │   └── validation.py               # Input data validation for training and prediction
│   ├── train_pipeline.py               # Script to execute the training pipeline
│   └── trained_models/                 # Directory for saving trained models
├── src_api/                            # Source code for the API server
│   ├── app/                            # API application code
│   │   ├── __init__.py                 # Initialization file for the app module
│   │   ├── api.py                      # Core logic for handling API endpoints
│   │   ├── config.py                   # API-specific configuration (e.g., host, port)
│   │   ├── main.py                     # Main entry point for the FastAPI server
│   │   └── schemas/                    # API request and response schemas
│   │       ├── __init__.py             # Initialization file for the schemas module
│   │       ├── health.py               # Schema for health-check endpoint
│   │       └── predict.py              # Schema for prediction requests and responses
│   ├── dist/                           # Distribution artifacts for the API
│   └── requirements.txt                # Dependencies for running the API
```

### How to Use

1. Clone the Repository

    ```
    git clone https://github.com/your-username/Application-Approval-Deployment-System.git
    cd Application-Approval-Deployment-System
    ```

2. Install Dependencies

    ```
    pip install -r requirements/requirements.txt
    ```

3. Train the Model

    ```
    Run the end-to-end training pipeline using the provided Jupyter notebook or the train_pipeline.py script:
    python src/train_pipeline.py
    ```

4. Start the API: Navigate to the src_api/ directory and run the FastAPI server:
    ```
    cd src_api
    uvicorn app.main:app --reload
    ```

5. Test the API: Access the API documentation and test endpoints:
- Swagger UI: http://127.0.0.1:8000/docs
- Redoc: http://127.0.0.1:8000/redoc

**Key Components**

Model Training:
- Train the classifier using the provided train_pipeline.py script or the included Jupyter notebook.


FastAPI Service:
- Serve predictions via API endpoints with pre-built schemas for request validation.

Docker Support:
- Use the Dockerfile to create a containerized environment for deployment:
    ```
    docker build -t application-approval.
    docker run -p 8000:8000 application-approval
    ```


**CI/CD Workflow**
- The project uses GitHub Actions for Continuous Integration and Continuous Deployment.

License
- This project is licensed under the MIT License.