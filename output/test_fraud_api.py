# test_fraud_api.py
import pytest
import json
from unittest.mock import patch, MagicMock
from fraud_api import app  # Import the Flask app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@patch('fraud_api.joblib.load')
@patch('fraud_api.keras.models.load_model')
@patch('fraud_api.json.load')
def test_health_check(mock_json, mock_keras, mock_joblib, client):
    # Mock the loaded models and config
    mock_json.return_value = {'optimal_threshold': 0.5, 'model_weights': {'random_forest': 0.25, 'xgboost': 0.25, 'logistic': 0.25, 'deep_learning': 0.25}}
    mock_joblib.return_value = MagicMock()  # Mock for joblib loads
    mock_keras.return_value = MagicMock()  # Mock for Keras model
    
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert 'models_loaded' in data

@patch('fraud_api.joblib.load')
@patch('fraud_api.keras.models.load_model')
@patch('fraud_api.json.load')
def test_predict_valid_input(mock_json, mock_keras, mock_joblib, client):
    # Mock models and config
    mock_json.return_value = {'optimal_threshold': 0.5, 'model_weights': {'random_forest': 0.25, 'xgboost': 0.25, 'logistic': 0.25, 'deep_learning': 0.25}}
    mock_rf = MagicMock()
    mock_rf.predict_proba.return_value = [[0.1, 0.9]]
    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = [[0.2, 0.8]]
    mock_log = MagicMock()
    mock_log.predict_proba.return_value = [[0.3, 0.7]]
    mock_dl = MagicMock()
    mock_dl.predict.return_value = ([0.4], None)
    mock_joblib.side_effect = [MagicMock(), MagicMock(), mock_rf, mock_xgb, mock_log, MagicMock(), MagicMock()]  # Mocks for fe, scaler, models
    mock_keras.return_value = mock_dl
    
    # Sample input data
    input_data = {
        'transaction_id': '12345',
        'Time': 1000,
        'Amount': 50.0,
        **{f'V{i}': 0.0 for i in range(1, 29)}
    }
    response = client.post('/predict', json=input_data)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'risk_score' in data
    assert 'recommended_action' in data

@patch('fraud_api.joblib.load')
@patch('fraud_api.keras.models.load_model')
@patch('fraud_api.json.load')
def test_predict_missing_fields(mock_json, mock_keras, mock_joblib, client):
    # Mock as above
    mock_json.return_value = {'optimal_threshold': 0.5, 'model_weights': {'random_forest': 0.25, 'xgboost': 0.25, 'logistic': 0.25, 'deep_learning': 0.25}}
    mock_joblib.return_value = MagicMock()
    mock_keras.return_value = MagicMock()
    
    # Incomplete input
    input_data = {'transaction_id': '12345', 'Time': 1000}
    response = client.post('/predict', json=input_data)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

@patch('fraud_api.joblib.load')
@patch('fraud_api.keras.models.load_model')
@patch('fraud_api.json.load')
def test_batch_predict(mock_json, mock_keras, mock_joblib, client):
    # Mock as above
    mock_json.return_value = {'optimal_threshold': 0.5, 'model_weights': {'random_forest': 0.25, 'xgboost': 0.25, 'logistic': 0.25, 'deep_learning': 0.25}}
    mock_rf = MagicMock()
    mock_rf.predict_proba.return_value = [[0.1, 0.9]]
    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = [[0.2, 0.8]]
    mock_log = MagicMock()
    mock_log.predict_proba.return_value = [[0.3, 0.7]]
    mock_dl = MagicMock()
    mock_dl.predict.return_value = ([0.4], None)
    mock_joblib.side_effect = [MagicMock(), MagicMock(), mock_rf, mock_xgb, mock_log, MagicMock(), MagicMock()]
    mock_keras.return_value = mock_dl
    
    # Sample batch input
    input_data = {
        'transactions': [
            {'transaction_id': '1', 'Time': 1000, 'Amount': 50.0, **{f'V{i}': 0.0 for i in range(1, 29)}},
            {'transaction_id': '2', 'Time': 2000, 'Amount': 100.0, **{f'V{i}': 0.0 for i in range(1, 29)}}
        ]
    }
    response = client.post('/batch_predict', json=input_data)
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'results' in data
    assert len(data['results']) == 2