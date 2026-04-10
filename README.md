

# Robust Anomaly Detection for Credit Card Transactions

This repository contains an AI system  focused on detecting anomalies in transaction data using various algorithms and techniques.

## By  [ Prince Appiah & Nikhila Kukkala ]


## Overview

The goal of this project is to identify fraudulent or unusual transactions that deviate from normal behaviour. The project involves data cleaning, feature engineering and selection, model training, evaluation, and visualization.
The project is divided into four Milestones. 
## Project Link Here
You can access the project and visualizations in the Jupyter notebook in this repository.
## Project Features
The project includes the following components:
- Data Exploration and Visualization: Initial exploration and visualization of the transaction data.
- Feature Engineering: Creation of new features to improve model performance.
- Ensemble Model (Random Forest, XGBoost, Logistic Regression and Autoencoders)
- Model Training: Training multiple models, including Random Forest, XGBoost, Logistic Regression and Autoencoders.
- Model Evaluation: Evaluation of the models using precision, recall, F1-score, and ROC-AUC metrics.
- Robustness and System Monitoring 
- Continual Learning and Human-In-The-Loop
- Results and Discussion: Summarizing the performance of the models and discussing the findings.
- Model Deployment!!

## Dataset
The data used in this project is sourced from creditcard.csv. It includes transaction records with various features such as transaction amount, V1-V29, and Time.

## How to Use
To explore the project:

### Step 1 - Do the following ;
      1. Clone or download this repository to your local machine
      2. Create a virtual environment and activate it
      3. Install the required packages (requirements.txt)
      4. Ensure the dataset is available in the data directory (creditcard.csv)
      5. Open and run the Jupyter notebook (anomaly.ipynb). Run all cells top-to-bottom.



### Steps to run the system 
Run all cells top-to-bottom. Requires creditcard.csv. 
This produces model_random_forest.pkl, model_xgboost.pkl, model_logistic.pkl, model_deep_learning.keras, feature_engineer.pkl, scaler.pkl, and model_config.json in output.
Skip this step if output already contains the trained model files.

### Step 2 — Start the Flask REST API
cd e:\_Anomaly\output
python fraud_api.py

The API starts at http://localhost:5000. Verify it's running:

curl http://localhost:5000/health

### Step 3 — Launch the Streamlit Dashboard

  Open a second terminal, activate the venv, then:
  cd e:\_Anomaly\fraud\fraud-detection-streamlit
  streamlit run app.py
  Opens at http://localhost:8501. Has two tabs — single transaction prediction and batch CSV upload.


### Step 4 — (Optional) Run Drift Monitoring
    cd e:\_Anomaly
    python output\monitoring\fraud_monitoring.py --reference output\monitoring_reference.csv --outdir output\monitoring
    Generates drift reports (CSVs, PNG dashboard, JSON summary) in monitoring.

### Step 5 — (Optional) Docker Deployment
Requires Docker Desktop running:

    cd e:\_Anomaly\output
    ### Create a requirements.txt for Docker build first
    Copy-Item ..\deployment\requirements_deploy.txt .\requirements.txt
    docker-compose up

API runs on http://localhost:5000 inside the container.


### Quick Reference: Ports & URLs
### Service	URL
    - Flask API	http://localhost:5000
    - API Health	http://localhost:5000/health
    - API Predict	POST http://localhost:5000/predict
    - API Batch	POST http://localhost:5000/batch_predict
    - Streamlit UI	http://localhost:8501

Minimum to see the UI working: complete Steps 1 → 2 → 3 (in order, Steps 2 and 3 in separate terminals simultaneously).
Run this from workspace root (after activating your venv):
locust -f output\stress_test.py --host http://localhost:5000
locust -f output\stress_test.py --host http://localhost:5000














