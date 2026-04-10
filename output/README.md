# Fraud Detection API

This project implements a Flask-based API for fraud detection using various machine learning models. The API provides endpoints for health checks and transaction predictions.

## Project Structure

- `fraud_api.py`: The main Flask application that handles requests and predictions.
- `requirements.txt`: Lists the required Python packages for the project.
- `Dockerfile`: Contains instructions for building a Docker image for the application.
- `docker-compose.yml`: Defines services and configurations for running the application in Docker.
- Model files (`model_random_forest.pkl`, `model_xgboost.pkl`, `model_logistic.pkl`, `model_deep_learning.keras`): Serialized models used for making predictions.
- `feature_engineer.pkl`: Serialized object for feature engineering.
- `scaler.pkl`: Serialized scaler for normalizing input features.
- `model_config.json`: Configuration file containing model settings and thresholds.
- `README.md`: Documentation for the project.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fraud-api-deployment
   ```

2. **Install dependencies**:
   You can install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   You can run the Flask application directly:
   ```bash
   python fraud_api.py
   ```
   Alternatively, you can use Docker to run the application:
   ```bash
   docker-compose up
   ```

## API Endpoints

- **Health Check**: 
  - `GET /health`
  - Returns the status of the API and loaded models.

- **Predict**:
  - `POST /predict`
  - Accepts a JSON payload with transaction details and returns a risk score and recommended action.

- **Batch Predict**:
  - `POST /batch_predict`
  - Accepts a JSON payload with multiple transactions and returns risk scores for each.

## Usage

To make a prediction, send a POST request to the `/predict` endpoint with the required fields in JSON format. Example:

```json
{
  "transaction_id": "12345",
  "Time": 1610000000,
  "Amount": 100.0,
  "V1": 0.1,
  "V2": 0.2,
  ...
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.