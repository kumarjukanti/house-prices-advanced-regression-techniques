
# House Prices — End-to-End MLOps Pipeline

## Project Overview

This is a hands-on MLOps learning and portfolio project focused on building an end-to-end machine learning pipeline for predicting residential house prices.

The project uses the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset from Kaggle, based on residential properties in Ames, Iowa.

The project is part of my transition into an MLOps engineering career. My goal is to gain practical experience implementing machine learning workflows, managing datasets and experiments, building reproducible pipelines, deploying models, and automating ML systems.

Rather than focusing only on achieving a high Kaggle score, this project emphasizes the engineering practices required to make machine learning systems reproducible, maintainable, testable, and deployable.

**Project status:** In active development.


## Machine Learning Problem

House price prediction is a supervised regression problem.

The objective is to predict the final sale price of a residential property using information describing the property.

The dataset contains information about many aspects of a house, including:

- Overall quality
- Overall condition
- Living area
- Lot size
- Neighborhood
- Year built
- Year remodeled
- Basement characteristics
- Garage characteristics
- Number of rooms
- Number of bathrooms
- Porch and deck areas
- Sale information

The target variable is:

- `SalePrice` — the property's sale price in dollars

The model produces a continuous numerical prediction representing the expected sale price of a house.

The project also explores how to turn an experimental regression model into a more reliable, reproducible, and production-oriented MLOps system.


## Kaggle Competition

This project is based on the Kaggle:

**House Prices - Advanced Regression Techniques**

Kaggle describes the competition as a practice environment for feature engineering and advanced regression techniques such as Random Forest and Gradient Boosting.

The dataset contains 79 explanatory variables describing residential homes in Ames, Iowa.

The competition objective is to predict the sales price for each property in the test dataset.

### Evaluation Metric

Kaggle evaluates submissions using Root Mean Squared Error (RMSE) between:

- The logarithm of the predicted sale price
- The logarithm of the observed sale price

This means the modeling pipeline needs to account for the logarithmic nature of the competition metric.

The logarithmic evaluation also reduces the effect of simply having larger errors on more expensive houses.

For more information, see the
[Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).


## Technology Stack

Area | Technologies
--- | ---
Programming | Python
Data processing | Pandas, NumPy
Data analysis | Pandas, NumPy
Visualization | Matplotlib, Seaborn
Machine learning | scikit-learn
Regression models | Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting
Environment management | uv
Version control | Git, GitHub
Data versioning | DVC
Cloud storage | AWS S3
Experiment tracking | MLflow
Model serving | FastAPI
Containerization | Docker
Container registry | Amazon ECR
Orchestration | Kubernetes / Amazon EKS
CI/CD | GitHub Actions
Testing | Pytest
Monitoring | To be implemented


## MLOps Architecture

The intended pipeline is:

    Raw Kaggle House Prices Dataset
                |
                v
        Data Versioning with DVC
                |
                v
             AWS S3
                |
                v
        Data Validation & Cleaning
                |
                v
       Exploratory Data Analysis
                |
                v
        Feature Engineering
                |
                v
       Train / Validation Split
                |
                v
    scikit-learn Preprocessing Pipeline
                |
                v
        Model Training & Evaluation
                |
                v
       MLflow Experiment Tracking
                |
                v
          Model Selection
                |
                v
         Model Packaging
                |
                v
       FastAPI Model Serving
                |
                v
            Docker
                |
                v
          Amazon ECR
                |
                v
        Kubernetes / EKS
                |
                v
          GitHub Actions
                |
                v
              CI/CD
                |
                v
      Deployment Verification
                |
                v
       Monitoring / Maintenance

DVC and AWS S3 provide data-versioning and storage capabilities alongside the pipeline.

Git tracks source code, configuration, infrastructure definitions, and workflow files.

MLflow provides experiment tracking and model comparison.

Docker packages the application and model into a reproducible container.

Amazon ECR stores the container images.

Amazon EKS provides the Kubernetes environment for model deployment.

GitHub Actions automates the CI/CD workflow.


## Dataset

The project uses the Kaggle House Prices dataset.

The dataset provides information about residential properties in Ames, Iowa.

### Dataset Files

Dataset | Description
--- | ---
`train.csv` | Training dataset containing house attributes and `SalePrice`
`test.csv` | Test dataset containing house attributes without `SalePrice`
`data_description.txt` | Detailed description of the dataset columns
`sample_submission.csv` | Example Kaggle submission format

The target variable is:

`SalePrice`

The test dataset does not contain the target value. Predictions generated for the test dataset are submitted to Kaggle for evaluation.

The dataset contains a mixture of:

- Numerical variables
- Categorical variables
- Ordinal variables
- Continuous variables
- Discrete variables
- Missing values

This makes the dataset useful for demonstrating realistic machine learning preprocessing and feature-engineering challenges.

Raw datasets are excluded from regular Git tracking.

Data versioning is handled separately through DVC.


## Implementation Progress

### Phase 1 — Data Ingestion and Validation

**In progress**

Planned activities:

- Create a Python project environment.
- Establish a reproducible project directory structure.
- Download the Kaggle dataset.
- Store raw datasets in `data/raw/`.
- Validate required files.
- Validate expected columns.
- Check data types.
- Check duplicate records.
- Analyze missing values.
- Validate the target variable.
- Create reusable data-validation logic.

The objective of this phase is to make sure that invalid or unexpected data is detected before it reaches the model-training pipeline.


### Phase 2 — Exploratory Data Analysis

**Planned**

Perform exploratory analysis of:

- Dataset dimensions
- Data types
- Missing values
- Duplicate records
- Numerical distributions
- Categorical distributions
- Target distribution
- Feature relationships
- Correlations
- Potential outliers
- Highly skewed variables

Special attention will be given to missing values because a missing value can have different meanings depending on the feature.

For example, a missing garage-related value may indicate that the property does not have a garage rather than simply representing an unknown value.

The objective is to understand the data before building the preprocessing and feature-engineering pipeline.


### Phase 3 — Data Cleaning and Feature Engineering

**Planned**

Create reusable data preparation and feature-engineering logic.

Potential transformations include:

- Numerical missing-value imputation
- Categorical missing-value handling
- One-hot encoding
- Ordinal encoding
- Numerical scaling where required
- Log transformation of highly skewed variables
- House age calculations
- Total square footage
- Total bathrooms
- Total porch/deck area
- Garage-related aggregate features
- Basement-related aggregate features

Feature engineering will be based on the actual characteristics of the dataset rather than applying transformations unnecessarily.

The final feature pipeline should be reusable during both model training and prediction.


### Phase 4 — Model Training

**Planned**

Establish baseline and advanced regression models.

Potential baseline models include:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet

Potential tree-based models include:

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost
- Other appropriate boosting models

The initial model will establish a baseline against which more advanced models can be compared.

The preprocessing and estimator will be combined using a scikit-learn `Pipeline` to prevent preprocessing statistics from being learned from validation data.


### Phase 5 — Model Evaluation

**Planned**

Model evaluation will be aligned with the Kaggle evaluation metric.

Metrics and evaluation methods may include:

- RMSE
- Log-transformed RMSE
- Cross-validation
- Training performance
- Validation performance
- Overfitting analysis
- Prediction error analysis

The project will focus primarily on the logarithmic evaluation metric used by Kaggle.

Cross-validation will also be considered because the labeled dataset is relatively small.


### Phase 6 — Experiment Tracking

**Planned**

MLflow will be used to track machine learning experiments.

Experiments will capture information such as:

- Model type
- Hyperparameters
- Feature configuration
- Training configuration
- Validation score
- Cross-validation score
- Model artifacts
- Dataset version
- Git commit

Example:

    Experiment
        |
        +-- Ridge Regression
        |       |
        |       +-- Parameters
        |       +-- Features
        |       +-- CV Score
        |
        +-- Random Forest
        |       |
        |       +-- Parameters
        |       +-- Features
        |       +-- CV Score
        |
        +-- Gradient Boosting
                |
                +-- Parameters
                +-- Features
                +-- CV Score

The objective is to make model comparison reproducible rather than relying on manually recorded experiment results.


### Phase 7 — DVC Pipeline

**Planned**

DVC will be used to version the dataset and ML pipeline.

Potential pipeline stages include:

    data_validation
           |
           v
    data_preparation
           |
           v
    feature_engineering
           |
           v
    train
           |
           v
    evaluate
           |
           v
    model

AWS S3 will be used as the remote storage location for DVC artifacts.

DVC allows the project to associate a particular model with the exact version of the data and pipeline used to produce it.


### Phase 8 — Model Packaging

**Planned**

The selected model and preprocessing pipeline will be packaged into a reproducible model artifact.

The artifact should contain everything necessary for the inference service to:

1. Receive house information.
2. Apply the same preprocessing used during training.
3. Generate the model prediction.
4. Return the predicted sale price.

The goal is to prevent differences between the training environment and prediction environment.


### Phase 9 — FastAPI Model Serving

**Planned**

Create a FastAPI service that exposes the trained model through an HTTP API.

Example endpoint:

    POST /predict

Example request:

    {
        "OverallQual": 7,
        "GrLivArea": 1710,
        "GarageCars": 2,
        "YearBuilt": 2003,
        "FullBath": 2
    }

Example response:

    {
        "predicted_sale_price": 250000
    }

The final API schema will contain the features required by the selected model.

A health endpoint will also be implemented to support deployment verification and Kubernetes health checks.


### Phase 10 — Docker Containerization

**Planned**

Create Docker images for the prediction service.

The container will package:

- Python runtime
- Application code
- Model artifact
- Dependencies
- FastAPI service
- Configuration

The objective is to ensure that the same application can run consistently across:

- Local development
- CI
- Docker
- Kubernetes
- Production-like environments


### Phase 11 — Kubernetes / Amazon EKS Deployment

**Planned**

Deploy the prediction service to Kubernetes.

The Kubernetes deployment will include:

- Deployment
- Service
- Container image
- Replica configuration
- Health checks
- Resource configuration

Amazon EKS will provide the managed Kubernetes environment.

Amazon ECR will store the Docker images.

The deployment should support multiple replicas so that the service can continue operating if an individual pod becomes unavailable.


### Phase 12 — CI/CD with GitHub Actions

**Planned**

The CI/CD pipeline will automate:

    Git Push
       |
       v
    GitHub Actions
       |
       +---- Run Tests
       |
       +---- Validate Code
       |
       +---- Build Docker Image
       |
       +---- Authenticate to AWS
       |
       +---- Push Image to ECR
       |
       +---- Deploy to EKS
       |
       +---- Verify Deployment
       |
       v
    Deployed ML Service

GitHub Actions will use AWS OIDC authentication instead of storing long-lived AWS access keys in GitHub.

Docker images will be tagged using the Git commit SHA.

This provides traceability between:

    Git Commit
         |
         v
    Docker Image
         |
         v
       ECR
         |
         v
    Kubernetes
         |
         v
    Deployed Model


## Key Data Quality Challenges

The House Prices dataset contains a large number of variables with different data types and missing-value patterns.

One of the major challenges is understanding what a missing value actually means.

For example:

Issue | Possible Interpretation
--- | ---
Missing garage information | Property may not have a garage
Missing basement information | Property may not have a basement
Missing pool information | Property may not have a pool
Missing categorical value | May represent absence rather than unknown
Missing numerical value | May require imputation based on feature meaning

This highlights an important machine learning engineering principle:

**Missing data should not automatically be treated as bad data.**

The meaning of the missing value needs to be understood before deciding how it should be handled.

The project will document the cleaning and preprocessing rules so that data preparation can be reproduced.


## Data Leakage Considerations

Data leakage is an important consideration for this project.

The Kaggle test dataset does not contain `SalePrice`.

Therefore, it cannot be used to calculate local model performance.

The labeled training dataset must be divided appropriately for model development.

The preprocessing pipeline must also be fitted only on the training data.

The general workflow is:

    Labeled Training Data
             |
             +----------------------+
             |                      |
             v                      v
        Training Data         Validation Data
             |                      |
             v                      |
       Fit preprocessing            |
             |                      |
             +----------+-----------+
                        |
                        v
                 Evaluate Model

The Kaggle test dataset remains separate and is used only for generating the final Kaggle submission.

This separation helps prevent information from the validation or test data from influencing the training process.


## Project Structure

The project structure is designed to separate data, source code, tests, models, deployment configuration, and MLOps configuration.

    house-prices-mlops/
    │
    ├── .github/
    │   └── workflows/
    │       ├── ci.yaml
    │       └── cd.yaml
    │
    ├── data/
    │   ├── raw/
    │   │   ├── train.csv
    │   │   ├── test.csv
    │   │   ├── data_description.txt
    │   │   └── sample_submission.csv
    │   │
    │   ├── interim/
    │   │
    │   └── processed/
    │
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   └── 02_model_experiments.ipynb
    │
    ├── src/
    │   ├── data/
    │   │   ├── validate.py
    │   │   └── prepare.py
    │   │
    │   ├── features/
    │   │   └── build_features.py
    │   │
    │   ├── models/
    │   │   ├── train.py
    │   │   ├── evaluate.py
    │   │   └── predict.py
    │   │
    │   └── api/
    │       └── main.py
    │
    ├── tests/
    │   ├── test_data.py
    │   ├── test_features.py
    │   ├── test_model.py
    │   └── test_api.py
    │
    ├── artifacts/
    │   ├── feature_schema.json
    │   └── model_metadata.json
    │
    ├── models/
    │
    ├── deployment/
    │   └── kubernetes/
    │       ├── deployment.yaml
    │       └── service.yaml
    │
    ├── dvc.yaml
    ├── dvc.lock
    ├── params.yaml
    ├── Dockerfile
    ├── docker-compose.yaml
    ├── requirements.txt
    ├── pyproject.toml
    ├── .gitignore
    ├── .dvcignore
    └── README.md

The structure will evolve as additional MLOps components are implemented.


## Getting Started

### 1. Clone the repository

    git clone <repository-url>
    cd house-prices-mlops


### 2. Set up the Python environment

Using `uv`:

    uv sync

Or using a Python virtual environment:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt


### 3. Prepare the dataset

Download the House Prices dataset from Kaggle:

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

Place the required files in:

    data/raw/

The raw dataset should not be committed directly to Git.

DVC will be used for dataset versioning.


### 4. Pull DVC data

    dvc pull

This restores the versioned dataset from the configured DVC remote.

AWS S3 is used as the DVC remote storage.


### 5. Validate the dataset

    uv run python src/data/validate.py

This step validates the input dataset before it reaches the machine learning pipeline.


### 6. Build the feature dataset

    uv run python src/features/build_features.py

The feature-building process creates the dataset used for model training.


### 7. Train the baseline model

    uv run python src/models/train.py

The training process will:

1. Load the processed dataset.
2. Separate features and target.
3. Build the preprocessing pipeline.
4. Train the baseline regression model.
5. Evaluate the model.
6. Save the model artifact.
7. Log experiment information to MLflow.


### 8. Run tests

    pytest

Automated tests will validate important parts of the pipeline.


### 9. Run the API locally

    uv run uvicorn src.api.main:app --reload

The API can then be tested locally before containerization.


### 10. Build the Docker image

    docker build -t house-prices-api .


### 11. Run the Docker container

    docker run -p 8000:8000 house-prices-api


### 12. Deploy to Kubernetes

    kubectl apply -f deployment/kubernetes/deployment.yaml
    kubectl apply -f deployment/kubernetes/service.yaml

Verify the deployment:

    kubectl get pods
    kubectl get deployments
    kubectl get services


## Engineering Lessons

This project focuses on learning important MLOps engineering practices:

- Why dataset versioning matters.
- Why validation should happen before model training.
- Why raw datasets should not be treated as source code.
- How DVC tracks datasets independently from Git.
- How AWS S3 can be used as a DVC remote.
- Why preprocessing must be learned from training data only.
- How to prevent data leakage.
- Why missing values need to be interpreted based on feature meaning.
- Why feature engineering is important for tabular regression.
- Why model evaluation must match the actual business or competition metric.
- Why cross-validation can be useful when working with smaller datasets.
- How MLflow helps compare experiments.
- How models can be packaged for deployment.
- How FastAPI can expose an ML model through an API.
- How Docker provides reproducible application environments.
- How Kubernetes manages deployed ML services.
- How Amazon ECR stores container images.
- How GitHub Actions automates CI/CD.
- How Git commit SHA provides deployment traceability.
- How AWS IAM permissions affect real-world ML pipelines.
- How to troubleshoot failures across GitHub, AWS, Docker, and Kubernetes.


## Current Limitations

The project is a learning implementation and is not yet a production-ready house-price valuation system.

In particular:

- The dataset represents historical residential properties in Ames, Iowa.
- The Kaggle test dataset does not contain the actual `SalePrice` values.
- Kaggle leaderboard performance should not be interpreted as real-world property valuation performance.
- The dataset is relatively small compared with many production machine learning datasets.
- Model performance can be sensitive to feature engineering choices.
- Hyperparameter tuning requires careful control to avoid overfitting.
- External real-estate, geographic, economic, and market information is not included.
- Model monitoring and automated retraining are future work.
- Production infrastructure and cloud deployment will require additional validation.
- The deployed model should not be considered a replacement for a professional property valuation.


## Future Enhancements

Potential future improvements include:

- Automated data-quality validation
- Advanced feature engineering
- Hyperparameter optimization
- Cross-validation automation
- Model ensemble/blending
- MLflow Model Registry
- Automated model promotion
- Model performance monitoring
- Data-drift monitoring
- Prediction monitoring
- Automated retraining
- Scheduled training pipelines
- Infrastructure-as-Code using Terraform
- Container vulnerability scanning
- Security scanning
- Kubernetes health checks
- Horizontal pod autoscaling
- Production monitoring and alerting


## Project Goal

The goal of this project is not simply to achieve a strong Kaggle score.

The larger goal is to demonstrate practical proficiency across the complete MLOps lifecycle:

    Data
      ↓
    Data Versioning
      ↓
    Data Validation
      ↓
    Feature Engineering
      ↓
    Model Training
      ↓
    Model Evaluation
      ↓
    Experiment Tracking
      ↓
    Model Packaging
      ↓
    API
      ↓
    Docker
      ↓
    Container Registry
      ↓
    Kubernetes
      ↓
    CI/CD
      ↓
    Monitoring

By completing this project, I aim to demonstrate that I can take a machine learning problem from a raw dataset through reproducible development and into an automated, deployable ML system.

The final project will serve as a technical portfolio project and a reference for demonstrating practical MLOps engineering skills.