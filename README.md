MLOps Assignment - Development & Deployment
Overview
--------
This repository implements an end-to-end MLOps pipeline for training and deploying machine learning models. The project follows best practices in model productization, including:

Standard ML project folder structure
Poetry for dependency management
Hydra for configuration management
DVC for data version control
GitHub for collaboration
Flask-based web application for model deployment
Project Structure
The main repository consists of a central mlops-assignment directory with the following subfolders:

xinrui/ - Wheat seeds dataset
dekai/ - Melbourne residential dataset
jet/ - Used car prices dataset
common/ - Shared resources, including dvc files, pyproject.toml, and this README.
Each person's folder (xinrui, dekai, jet) contains:

data/ - Raw datasets (data/raw/)
notebooks/ - Task 1 & 2 Jupyter notebooks
src/ - Includes train.py, preprocess.py, and config.yaml
models/ - Stores trained model .pkl files
processed/ - Processed train/test datasets
plot/ (optional) - Visualization outputs
app/ - Task 3 Flask app (contains app.py, model .pkl files, and templates/ folder for UI)
Deployment Scripts - Each folder contains Python scripts for individual cloud deployment.
Development & Deployment Workflow
Data Preparation & Preprocessing

Raw data stored in data/raw/
Processed data versioned with DVC
Model Training & Evaluation

Models trained using PyCaret, managed via Hydra
Hyperparameter tuning for optimal performance
Model Deployment

Flask-based web app to serve predictions
Docker containerization for cloud deployment
Individual model deployment scripts included in xinrui/, dekai/, and jet/ folders
Version Control & Collaboration

Poetry for dependency management
GitHub for collaboration
Centralized & Individual Model Deployment
The app/ folder in the main directory serves all three models in a single deployment:

Running app.py from the main directory app folder provides a UI for selecting and using different models.
The templates/ folder contains:
Three model-specific HTML pages
index.html with a navigation bar for model selection.

Individual Cloud Deployments:
Each person's folder (xinrui/, dekai/, jet/) contains Python scripts for deploying their model separately to the cloud.

Deployment Links:
Dekai's App: XXXXXXXXXXXXXXXX
Jet's App: XXXXXXXXXXXXXXXX
Xinrui's App (Wheat Seed Classification App - By streamlit): https://mlops-assignment-bjvgx5bo3vmtuggmwxadox.streamlit.app/
This structure ensures modular development while enabling both unified deployment and individual cloud-hosted models.
