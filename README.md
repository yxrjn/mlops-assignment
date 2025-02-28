MLOps Assignment - Development & Deployment<br>
Overview<br>
--------<br>
This repository implements an end-to-end MLOps pipeline for training and deploying machine learning models. The project follows best practices in model productization, including:<br>

Standard ML project folder structure<br>
Poetry for dependency management<br>
Hydra for configuration management<br>
DVC for data version control<br>
GitHub for collaboration<br>
Flask-based web application for model deployment<br>
Project Structure<br>
The main repository consists of a central mlops-assignment directory with the following subfolders:<br>
app/ - contains our integrated flask for all data models <br>
xinrui/ - Wheat seeds dataset<br>
dekai/ - Melbourne residential dataset<br>
jet/ - Used car prices dataset<br>
common/ - Shared resources, including dvc files, pyproject.toml, and poetry lock file.<br>
Each person's folder (xinrui, dekai, jet) contains:<br>

data/ - Raw datasets (data/raw/)<br>
notebooks/ - Task 1 & 2 Jupyter notebooks and inside would also include those files that is being generated after running task 1 and task 2 files.<br>
src/ - Includes train.py, preprocess.py, and config.yaml<br>
models/ - Stores trained model .pkl files<br>
processed/ - Processed train/test datasets<br>
plot/ (optional) - Visualization outputs<br>
app/ - Task 3 Flask app (contains app.py, model .pkl files, and templates/ folder for UI)<br>
Deployment Scripts - Each folder contains Python scripts for individual cloud deployment.<br>
Development & Deployment Workflow<br>
Data Preparation & Preprocessing<br>

Raw data stored in data/raw/<br>
Processed data versioned with DVC<br>
Model Training & Evaluation<br>

Models trained using PyCaret, managed via Hydra<br>
Hyperparameter tuning for optimal performance<br>
Model Deployment<br>

Flask-based web app to serve predictions<br>
Docker containerization for cloud deployment<br>
Individual model deployment scripts included in xinrui/, dekai/, and jet/ folders<br>
Version Control & Collaboration

Poetry for dependency management<br>
GitHub for collaboration<br>
Centralized & Individual Model Deployment<br>
The app/ folder in the main directory serves all three models in a single deployment:<br>

Running app.py from the main directory app folder provides a UI for selecting and using different models.<br>
The templates/ folder contains:<br>
Three model-specific HTML pages<br>
index.html with a navigation bar for model selection.<br>

Individual Cloud Deployments:<br>
Each person's folder (xinrui/, dekai/, jet/) contains Python scripts for deploying their model separately to the cloud.<br>

Deployment Links:<br>
Dekai's App (Melbourne Residential Price Prediction): https://mlops-assignment-3ed4lxwbpn6szrtnattjbe.streamlit.app/ <br>
Jet's App: (Used Car Price Prediction): https://mlops-assignment-baana7cmgf6mwcf6wkxnh6.streamlit.app/ <br>
Xinrui's App (Wheat Seed Classification App - By streamlit): https://mlops-assignment-bjvgx5bo3vmtuggmwxadox.streamlit.app/<br>
The group's deloyed model  to the cloud : https://mlops-assignment-exjnfjzf7hzv87awjqoyw9.streamlit.app/<br>
This structure ensures modular development while enabling both unified deployment and individual cloud-hosted models.<br>
