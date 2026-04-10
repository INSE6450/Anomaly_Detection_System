from locust import HttpUser, task, between
import json
import numpy as np
import random
import logging

# Set up logging for custom metrics (e.g., false positives/negatives)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudDetectionUser(HttpUser):
    wait_time = between(1, 3)

    def generate_base_transaction(self, transaction_id="test-001"):
        return {
            "transaction_id": transaction_id,
            "Time": random.randint(0, 172792),
            "Amount": random.uniform(1, 1000),
            **{f"V{i}": random.gauss(0, 1) for i in range(1, 29)}  # Normal distribution
        }

    def log_model_metrics(self, response, scenario_name):
        """Log custom model metrics based on response and simulated ground truth."""
        if response.status_code != 200:
            return  # Skip if API failed
        
        data = response.json()
        risk_score = data.get('risk_score', 0)
        is_fraud_pred = risk_score > 0.5  # Assuming threshold 0.5
        
        # Simulate ground truth based on scenario
        if "corrupted" in scenario_name or "partial" in scenario_name:
            is_fraud_true = random.random() < 0.1  # Low fraud rate for noise/partial tests
        elif "ood" in scenario_name:
            is_fraud_true = random.random() < 0.8  # High fraud rate for OOD
        elif "rarity" in scenario_name:
            is_fraud_true = random.random() < 0.05  # Very rare fraud
        else:
            is_fraud_true = random.random() < 0.001  # Baseline legit
        
        # Log metrics
        if is_fraud_pred and not is_fraud_true:
            logger.info(f"False Positive in {scenario_name}: Risk Score {risk_score}")
        elif not is_fraud_pred and is_fraud_true:
            logger.info(f"False Negative in {scenario_name}: Risk Score {risk_score}")
        else:
            logger.info(f"Correct Prediction in {scenario_name}: Risk Score {risk_score}, True Fraud: {is_fraud_true}")

    @task(3)
    def predict_single(self):
        data = self.generate_base_transaction()
        response = self.client.post("/predict", json=data, headers={"Content-Type": "application/json"}, name="predict_single")
        self.log_model_metrics(response, "single")

    @task(1)
    def predict_batch(self):
        transactions = [self.generate_base_transaction(f"batch-{i}") for i in range(10)]
        response = self.client.post("/batch_predict", json={"transactions": transactions}, headers={"Content-Type": "application/json"}, name="predict_batch")
        # For batch, log once (simplified; could loop over results if needed)
        self.log_model_metrics(response, "batch")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="health_check")

    # Corrupted Inputs
    @task(2)
    def predict_corrupted_noise(self):
        data = self.generate_base_transaction("corrupted-noise")
        for i in range(1, 29):
            data[f"V{i}"] += np.random.normal(0, 2)
        response = self.client.post("/predict", json=data, headers={"Content-Type": "application/json"}, name="predict_corrupted_noise")
        self.log_model_metrics(response, "corrupted_noise")

    @task(2)
    def predict_corrupted_blur(self):
        data = self.generate_base_transaction("corrupted-blur")
        v_vals = [data[f"V{i}"] for i in range(1, 29)]
        blurred = [(v_vals[i-1] + v_vals[i] + v_vals[(i+1)%28]) / 3 for i in range(28)]
        for i in range(1, 29):
            data[f"V{i}"] = blurred[i-1]
        response = self.client.post("/predict", json=data, headers={"Content-Type": "application/json"}, name="predict_corrupted_blur")
        self.log_model_metrics(response, "corrupted_blur")

    @task(2)
    def predict_corrupted_masking(self):
        data = self.generate_base_transaction("corrupted-masking")
        mask_indices = random.sample(range(1, 29), 6)
        for i in mask_indices:
            data[f"V{i}"] = 0.0
        response = self.client.post("/predict", json=data, headers={"Content-Type": "application/json"}, name="predict_corrupted_masking")
        self.log_model_metrics(response, "corrupted_masking")

    @task(2)
    def predict_corrupted_dropout(self):
        data = self.generate_base_transaction("corrupted-dropout")
        dropout_indices = random.sample(range(1, 29), 3)
        for i in dropout_indices:
            data[f"V{i}"] = None
        response = self.client.post("/predict", json=data, headers={"Content-Type": "application/json"}, name="predict_corrupted_dropout")
        self.log_model_metrics(response, "corrupted_dropout")

    # Partial Feature Loss
    @task(2)
    def predict_partial_loss(self):
        data = self.generate_base_transaction("partial-loss")
        loss_indices = random.sample(range(1, 29), 9)
        for i in loss_indices:
            data[f"V{i}"] = 0.0
        response = self.client.post("/predict", json=data, headers={"Content-Type": "application/json"}, name="predict_partial_loss")
        self.log_model_metrics(response, "partial_loss")

    # OOD Samples
    @task(2)
    def predict_ood_extreme_amount(self):
        data = self.generate_base_transaction("ood-extreme")
        data["Amount"] = random.uniform(10000, 50000)
        response = self.client.post("/predict", json=data, headers={"Content-Type": "application/json"}, name="predict_ood_extreme_amount")
        self.log_model_metrics(response, "ood_extreme_amount")

    @task(2)
    def predict_ood_unusual_pattern(self):
        data = self.generate_base_transaction("ood-pattern")
        for i in range(1, 29):
            data[f"V{i}"] += 5
        response = self.client.post("/predict", json=data, headers={"Content-Type": "application/json"}, name="predict_ood_unusual_pattern")
        self.log_model_metrics(response, "ood_unusual_pattern")

    # Class Rarity Scenarios
    @task(2)
    def predict_rarity_low_amount_fraud(self):
        data = self.generate_base_transaction("rarity-low")
        data["Amount"] = random.uniform(0.1, 1)
        for i in range(1, 29):
            data[f"V{i}"] += random.gauss(2, 0.5)
        response = self.client.post("/predict", json=data, headers={"Content-Type": "application/json"}, name="predict_rarity_low_amount_fraud")
        self.log_model_metrics(response, "rarity_low_amount_fraud")

    @task(2)
    def predict_rarity_high_frequency(self):
        data = self.generate_base_transaction("rarity-freq")
        data["Time"] = random.randint(0, 100)
        data["Amount"] = random.uniform(500, 1000)
        response = self.client.post("/predict", json=data, headers={"Content-Type": "application/json"}, name="predict_rarity_high_frequency")
        self.log_model_metrics(response, "rarity_high_frequency")