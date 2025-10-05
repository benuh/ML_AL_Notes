# MLOps Best Practices

Production-ready ML: Build, deploy, and maintain machine learning systems at scale.

## Table of Contents
1. [MLOps Fundamentals](#mlops-fundamentals)
2. [Experiment Tracking](#experiment-tracking)
3. [Model Versioning and Registry](#model-versioning-and-registry)
4. [Data Versioning](#data-versioning)
5. [CI/CD for ML](#cicd-for-ml)
6. [Model Deployment](#model-deployment)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Feature Stores](#feature-stores)
9. [Production Patterns](#production-patterns)

---

## MLOps Fundamentals

### What is MLOps?

**MLOps** = Machine Learning + DevOps + Data Engineering

**Key Principles:**
1. **Reproducibility** - Same code + data → same results
2. **Automation** - CI/CD pipelines for ML
3. **Monitoring** - Track model performance in production
4. **Versioning** - Code, data, models, and experiments
5. **Collaboration** - Between data scientists, engineers, and ops

### MLOps Maturity Model

**Level 0: Manual** - Notebooks, ad-hoc scripts
**Level 1: Automated Training** - Pipelines, version control
**Level 2: Automated Deployment** - CI/CD, serving infrastructure
**Level 3: Full MLOps** - Automated retraining, monitoring, feedback loops

---

## Experiment Tracking

### MLflow

**Best for:** Experiment tracking, model registry, deployment.

```python
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Start MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("classification_experiments")

def train_with_mlflow(X_train, y_train, X_val, y_val, config):
    with mlflow.start_run(run_name=f"lr_{config['learning_rate']}"):
        # Log parameters
        mlflow.log_params(config)

        # Log tags
        mlflow.set_tag("model_type", "neural_network")
        mlflow.set_tag("framework", "pytorch")

        # Train model
        model = create_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        for epoch in range(config['epochs']):
            # Training
            train_loss = train_epoch(model, X_train, y_train, optimizer)

            # Validation
            val_loss, y_pred = evaluate(model, X_val, y_val)

            # Log metrics per epoch
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, step=epoch)

        # Final metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')

        mlflow.log_metrics({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        })

        # Log model
        mlflow.pytorch.log_model(model, "model")

        # Log artifacts (plots, configs, etc.)
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("config.yaml")

        # Log dataset info
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))

        return model

# Usage
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'hidden_size': 128,
    'dropout': 0.3
}

model = train_with_mlflow(X_train, y_train, X_val, y_val, config)
```

**Query and Compare Experiments:**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all runs from experiment
experiment = client.get_experiment_by_name("classification_experiments")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

# Print top runs
for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {run.data.metrics['accuracy']:.4f}")
    print(f"Config: {run.data.params}")
    print("-" * 50)

# Load best model
best_run = runs[0]
model_uri = f"runs:/{best_run.info.run_id}/model"
loaded_model = mlflow.pytorch.load_model(model_uri)
```

---

### Weights & Biases (W&B)

**Best for:** Rich visualizations, hyperparameter sweeps, collaboration.

```python
import wandb

# Initialize
wandb.init(
    project="my-ml-project",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "architecture": "resnet50",
        "dataset": "imagenet"
    }
)

# Access config
config = wandb.config

# Train model
model = create_model(config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

for epoch in range(config.epochs):
    # Training
    train_loss = 0
    for X_batch, y_batch in train_loader:
        loss = train_step(model, X_batch, y_batch, optimizer)
        train_loss += loss

    # Validation
    val_loss, val_acc = validate(model, val_loader)

    # Log metrics
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss / len(train_loader),
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    # Log images (e.g., predictions)
    if epoch % 10 == 0:
        wandb.log({
            "predictions": wandb.Image(prediction_image),
            "attention_map": wandb.Image(attention_visualization)
        })

# Log model
wandb.log_artifact(model, name="trained_model", type="model")

# Save model to W&B
wandb.save("model.pth")

# Finish run
wandb.finish()
```

**Hyperparameter Sweeps with W&B:**
```yaml
# sweep.yaml
program: train.py
method: bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform
    min: 0.00001
    max: 0.1
  batch_size:
    values: [16, 32, 64, 128]
  dropout:
    distribution: uniform
    min: 0.1
    max: 0.5
  optimizer:
    values: ['adam', 'sgd', 'adamw']
```

```python
# Run sweep
import wandb

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project='my-project')

# Agent function
def train():
    wandb.init()
    config = wandb.config

    # Train with config
    model = train_model(config)

    # Log results
    wandb.log({'val_accuracy': accuracy})

# Run sweep agents
wandb.agent(sweep_id, function=train, count=50)
```

---

## Model Versioning and Registry

### Model Registry Pattern

```python
import mlflow
from mlflow.tracking import MlflowClient

class ModelRegistry:
    """Model registry for versioning and deployment"""

    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(self, run_id, model_name, description=""):
        """Register model from run"""
        model_uri = f"runs:/{run_id}/model"

        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        # Update description
        self.client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description
        )

        return model_version

    def promote_to_production(self, model_name, version):
        """Promote model to production"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )

        print(f"Model {model_name} v{version} promoted to Production")

    def get_production_model(self, model_name):
        """Get current production model"""
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pytorch.load_model(model_uri)
        return model

    def get_model_version_info(self, model_name, version):
        """Get model version metadata"""
        version_info = self.client.get_model_version(
            name=model_name,
            version=version
        )

        return {
            'name': version_info.name,
            'version': version_info.version,
            'stage': version_info.current_stage,
            'description': version_info.description,
            'run_id': version_info.run_id,
            'creation_timestamp': version_info.creation_timestamp,
        }

    def compare_models(self, model_name, version1, version2):
        """Compare two model versions"""
        v1_info = self.get_model_version_info(model_name, version1)
        v2_info = self.get_model_version_info(model_name, version2)

        # Get metrics from runs
        run1 = self.client.get_run(v1_info['run_id'])
        run2 = self.client.get_run(v2_info['run_id'])

        comparison = {
            'version1': {
                'version': version1,
                'metrics': run1.data.metrics,
                'params': run1.data.params
            },
            'version2': {
                'version': version2,
                'metrics': run2.data.metrics,
                'params': run2.data.params
            }
        }

        return comparison

# Usage
registry = ModelRegistry()

# Register model from training run
model_version = registry.register_model(
    run_id="abc123",
    model_name="fraud_detector",
    description="XGBoost model trained on 2024-01 data"
)

# Promote to production after validation
registry.promote_to_production("fraud_detector", version=3)

# Load production model
production_model = registry.get_production_model("fraud_detector")

# Compare versions
comparison = registry.compare_models("fraud_detector", version1=2, version2=3)
```

---

## Data Versioning

### DVC (Data Version Control)

```bash
# Initialize DVC
dvc init

# Add data to DVC
dvc add data/train.csv
git add data/train.csv.dvc .gitignore
git commit -m "Add training data"

# Configure remote storage
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc remote modify myremote region us-west-2

# Push data to remote
dvc push

# Pull data from remote
dvc pull
```

**DVC Pipeline:**
```yaml
# dvc.yaml
stages:
  preprocess:
    cmd: python preprocess.py
    deps:
      - data/raw/train.csv
      - preprocess.py
    outs:
      - data/processed/train.pkl

  train:
    cmd: python train.py
    deps:
      - data/processed/train.pkl
      - train.py
    params:
      - train.learning_rate
      - train.epochs
    outs:
      - models/model.pkl
    metrics:
      - metrics/train_metrics.json:
          cache: false

  evaluate:
    cmd: python evaluate.py
    deps:
      - models/model.pkl
      - data/processed/test.pkl
      - evaluate.py
    metrics:
      - metrics/test_metrics.json:
          cache: false
```

```bash
# Run pipeline
dvc repro

# See metrics
dvc metrics show

# Compare experiments
dvc metrics diff
```

---

### Data Validation with Great Expectations

```python
import great_expectations as ge
import pandas as pd

# Create expectation suite
df = ge.read_csv('data/train.csv')

# Define expectations
df.expect_column_values_to_be_between('age', 0, 120)
df.expect_column_values_to_not_be_null('user_id')
df.expect_column_values_to_be_in_set('status', ['active', 'inactive', 'pending'])
df.expect_column_mean_to_be_between('revenue', 0, 10000)

# Save suite
df.save_expectation_suite('data_validation_suite.json')

# Validate new data
def validate_data(df_path):
    df = ge.read_csv(df_path)
    results = df.validate(expectation_suite='data_validation_suite.json')

    if not results['success']:
        print("Data validation failed!")
        for result in results['results']:
            if not result['success']:
                print(f"Failed: {result['expectation_config']['expectation_type']}")

        raise ValueError("Data validation failed")

    print("Data validation passed ✓")
    return True

# Use in pipeline
validate_data('data/new_batch.csv')
```

---

## CI/CD for ML

### GitHub Actions for ML

```yaml
# .github/workflows/ml_pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run unit tests
        run: |
          pytest tests/

      - name: Run data validation
        run: |
          python scripts/validate_data.py

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Pull data from DVC
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          dvc pull

      - name: Train model
        run: |
          python train.py

      - name: Evaluate model
        run: |
          python evaluate.py

      - name: Check model performance
        run: |
          python scripts/check_model_performance.py

  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2

      - name: Deploy to production
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          python deploy.py --env production
```

### Model Performance Gating

```python
# scripts/check_model_performance.py
import json

def check_model_performance(metrics_path, thresholds):
    """Gate deployment based on performance thresholds"""

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Check all thresholds
    failed_checks = []

    for metric, threshold in thresholds.items():
        if metric not in metrics:
            failed_checks.append(f"Missing metric: {metric}")
        elif metrics[metric] < threshold:
            failed_checks.append(
                f"{metric} ({metrics[metric]:.4f}) below threshold ({threshold})"
            )

    if failed_checks:
        print("❌ Model performance checks failed:")
        for check in failed_checks:
            print(f"  - {check}")
        exit(1)

    print("✅ All performance checks passed!")
    print(f"Metrics: {metrics}")
    exit(0)

# Usage
thresholds = {
    'accuracy': 0.90,
    'precision': 0.85,
    'recall': 0.85,
    'f1_score': 0.85
}

check_model_performance('metrics/test_metrics.json', thresholds)
```

---

## Model Deployment

### FastAPI Model Serving

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import mlflow
from typing import List

app = FastAPI(title="ML Model API")

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    # Load from MLflow registry
    model_uri = "models:/fraud_detector/Production"
    model = mlflow.pytorch.load_model(model_uri)
    print("Model loaded successfully")

# Request/Response models
class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Preprocess
        features = torch.tensor([request.features], dtype=torch.float32)

        # Predict
        with torch.no_grad():
            output = model(features)
            probability = torch.softmax(output, dim=1)[0, 1].item()
            prediction = int(probability > 0.5)

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_version="1.0.0"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/metrics")
async def metrics():
    # Prometheus metrics
    return {
        "predictions_total": 12345,
        "prediction_latency_ms": 15.3,
        "model_version": "1.0.0"
    }
```

**Docker Deployment:**
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY models/ models/

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

---

### Batch Prediction Pipeline

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

class LoadModel(beam.DoFn):
    """Load model once per worker"""

    def setup(self):
        import mlflow
        self.model = mlflow.pytorch.load_model("models:/fraud_detector/Production")

    def process(self, element):
        # Predict
        features = element['features']
        prediction = self.model.predict([features])[0]

        yield {
            'id': element['id'],
            'prediction': int(prediction),
            'timestamp': element['timestamp']
        }

def run_batch_prediction(input_path, output_path):
    """Batch prediction pipeline"""

    pipeline_options = PipelineOptions()

    with beam.Pipeline(options=pipeline_options) as p:
        predictions = (
            p
            | 'Read Input' >> beam.io.ReadFromText(input_path)
            | 'Parse JSON' >> beam.Map(lambda x: json.loads(x))
            | 'Predict' >> beam.ParDo(LoadModel())
            | 'Format Output' >> beam.Map(lambda x: json.dumps(x))
            | 'Write Output' >> beam.io.WriteToText(output_path)
        )

# Run
run_batch_prediction(
    input_path='gs://my-bucket/input/*.json',
    output_path='gs://my-bucket/output/predictions'
)
```

---

## Monitoring and Observability

### Model Performance Monitoring

```python
import prometheus_client as prom
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
prediction_counter = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['model_version', 'prediction_class']
)

prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency',
    ['model_version']
)

feature_drift = Gauge(
    'feature_drift_score',
    'Feature drift score',
    ['feature_name']
)

class ModelMonitor:
    """Monitor model predictions in production"""

    def __init__(self, model, model_version):
        self.model = model
        self.model_version = model_version
        self.prediction_history = []

    def predict_with_monitoring(self, features):
        # Time prediction
        start_time = time.time()

        # Predict
        prediction = self.model.predict([features])[0]

        # Record latency
        latency = time.time() - start_time
        prediction_latency.labels(model_version=self.model_version).observe(latency)

        # Count prediction
        prediction_counter.labels(
            model_version=self.model_version,
            prediction_class=str(prediction)
        ).inc()

        # Store for drift detection
        self.prediction_history.append({
            'features': features,
            'prediction': prediction,
            'timestamp': time.time()
        })

        return prediction

    def detect_feature_drift(self, reference_data, window_size=1000):
        """Detect feature drift using KL divergence"""
        from scipy.stats import entropy

        if len(self.prediction_history) < window_size:
            return

        # Recent predictions
        recent_features = [p['features'] for p in self.prediction_history[-window_size:]]
        recent_features = np.array(recent_features)

        # Compare distributions
        for i, feature_name in enumerate(['feature_0', 'feature_1', ...]):
            ref_hist, _ = np.histogram(reference_data[:, i], bins=50, density=True)
            curr_hist, _ = np.histogram(recent_features[:, i], bins=50, density=True)

            # KL divergence
            drift_score = entropy(ref_hist + 1e-10, curr_hist + 1e-10)

            # Update metric
            feature_drift.labels(feature_name=feature_name).set(drift_score)

            # Alert if high drift
            if drift_score > 0.5:
                print(f"⚠️ High drift detected in {feature_name}: {drift_score:.4f}")

# Usage
monitor = ModelMonitor(model, model_version="1.0.0")

# In production
prediction = monitor.predict_with_monitoring(features)

# Periodic drift detection
monitor.detect_feature_drift(reference_data, window_size=1000)
```

---

### Alerting with Evidently AI

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Reference and current data
reference_data = pd.read_csv('reference_data.csv')
current_data = pd.read_csv('current_data.csv')

# Column mapping
column_mapping = ColumnMapping(
    target='target',
    prediction='prediction',
    numerical_features=['age', 'income', 'score'],
    categorical_features=['category', 'region']
)

# Create report
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset()
])

report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

# Save report
report.save_html('drift_report.html')

# Get drift results
drift_results = report.as_dict()

# Alert if drift detected
if drift_results['metrics'][0]['result']['dataset_drift']:
    print("⚠️ Data drift detected!")
    # Send alert (email, Slack, PagerDuty, etc.)
    send_alert("Data drift detected in production model")
```

---

## Feature Stores

### Feast Feature Store

```yaml
# feature_repo/feature_store.yaml
project: fraud_detection
registry: s3://my-bucket/feast/registry.db
provider: aws
online_store:
  type: dynamodb
  region: us-west-2
offline_store:
  type: redshift
  cluster_id: my-cluster
  database: analytics
```

```python
# feature_repo/features.py
from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import RedshiftSource

# Define entity
user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="User ID"
)

# Define data source
user_stats_source = RedshiftSource(
    query="SELECT * FROM user_stats",
    timestamp_field="event_timestamp"
)

# Define feature view
user_stats_fv = FeatureView(
    name="user_stats",
    entities=["user_id"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="total_transactions", dtype=ValueType.INT64),
        Feature(name="avg_transaction_amount", dtype=ValueType.DOUBLE),
        Feature(name="days_since_last_transaction", dtype=ValueType.INT64),
    ],
    online=True,
    batch_source=user_stats_source,
)
```

```python
# Training: Get historical features
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

# Entity dataframe
entity_df = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5],
    "event_timestamp": pd.to_datetime([
        "2024-01-01", "2024-01-01", "2024-01-02",
        "2024-01-02", "2024-01-03"
    ])
})

# Get historical features
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_stats:total_transactions",
        "user_stats:avg_transaction_amount",
        "user_stats:days_since_last_transaction",
    ]
).to_df()

# Inference: Get online features
features = store.get_online_features(
    features=[
        "user_stats:total_transactions",
        "user_stats:avg_transaction_amount",
    ],
    entity_rows=[{"user_id": 123}]
).to_dict()
```

---

## Production Patterns

### Shadow Mode Deployment

```python
class ShadowModePredictor:
    """Deploy new model in shadow mode"""

    def __init__(self, production_model, shadow_model):
        self.production_model = production_model
        self.shadow_model = shadow_model

    def predict(self, features):
        # Production prediction (served to user)
        prod_prediction = self.production_model.predict(features)

        # Shadow prediction (logged, not served)
        try:
            shadow_prediction = self.shadow_model.predict(features)

            # Log comparison
            self.log_shadow_comparison(
                features=features,
                prod_prediction=prod_prediction,
                shadow_prediction=shadow_prediction
            )
        except Exception as e:
            print(f"Shadow model error: {e}")

        # Always return production prediction
        return prod_prediction

    def log_shadow_comparison(self, features, prod_prediction, shadow_prediction):
        """Log shadow vs production predictions"""
        log_data = {
            'timestamp': time.time(),
            'features': features,
            'production_prediction': prod_prediction,
            'shadow_prediction': shadow_prediction,
            'agreement': prod_prediction == shadow_prediction
        }

        # Log to database/file
        with open('shadow_logs.jsonl', 'a') as f:
            f.write(json.dumps(log_data) + '\n')
```

---

### A/B Testing Framework

```python
import hashlib

class ABTestingFramework:
    """A/B testing for ML models"""

    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio

    def get_model_for_user(self, user_id):
        """Consistent model assignment based on user_id"""
        hash_value = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        assignment = (hash_value % 100) / 100.0

        if assignment < self.split_ratio:
            return 'model_a', self.model_a
        else:
            return 'model_b', self.model_b

    def predict(self, user_id, features):
        variant, model = self.get_model_for_user(user_id)

        # Predict
        prediction = model.predict([features])[0]

        # Log for analysis
        self.log_prediction(
            user_id=user_id,
            variant=variant,
            features=features,
            prediction=prediction
        )

        return prediction

    def log_prediction(self, user_id, variant, features, prediction):
        """Log prediction for A/B analysis"""
        log_data = {
            'user_id': user_id,
            'variant': variant,
            'features': features,
            'prediction': prediction,
            'timestamp': time.time()
        }

        # Log to analytics platform
        # analytics.track('model_prediction', log_data)

    def analyze_results(self, outcome_data):
        """Analyze A/B test results"""
        from scipy import stats

        # Split by variant
        variant_a = outcome_data[outcome_data['variant'] == 'model_a']
        variant_b = outcome_data[outcome_data['variant'] == 'model_b']

        # Statistical test
        t_stat, p_value = stats.ttest_ind(
            variant_a['outcome'],
            variant_b['outcome']
        )

        result = {
            'model_a_mean': variant_a['outcome'].mean(),
            'model_b_mean': variant_b['outcome'].mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        return result
```

---

### Canary Deployment

```python
class CanaryDeployment:
    """Gradual rollout of new model"""

    def __init__(self, old_model, new_model, initial_traffic=0.05):
        self.old_model = old_model
        self.new_model = new_model
        self.traffic_percentage = initial_traffic
        self.error_rate_threshold = 0.05

    def predict(self, features):
        # Route traffic based on percentage
        if random.random() < self.traffic_percentage:
            model = self.new_model
            variant = 'canary'
        else:
            model = self.old_model
            variant = 'stable'

        # Predict
        try:
            prediction = model.predict([features])[0]

            # Log success
            self.log_prediction(variant, success=True)

            return prediction

        except Exception as e:
            # Log error
            self.log_prediction(variant, success=False, error=str(e))

            # Fallback to stable model
            if variant == 'canary':
                return self.old_model.predict([features])[0]
            else:
                raise

    def log_prediction(self, variant, success, error=None):
        # Log to monitoring system
        pass

    def increase_traffic(self, increment=0.1):
        """Gradually increase canary traffic"""
        canary_error_rate = self.get_canary_error_rate()

        if canary_error_rate < self.error_rate_threshold:
            self.traffic_percentage = min(1.0, self.traffic_percentage + increment)
            print(f"Increased canary traffic to {self.traffic_percentage*100}%")
        else:
            print(f"⚠️ Canary error rate too high: {canary_error_rate:.2%}")
            self.rollback()

    def rollback(self):
        """Rollback canary deployment"""
        self.traffic_percentage = 0
        print("❌ Canary deployment rolled back")

    def get_canary_error_rate(self):
        # Query monitoring system for canary error rate
        # Return error_rate
        return 0.02  # Placeholder
```

---

## Summary: MLOps Stack

| Component | Tools | Purpose |
|-----------|-------|---------|
| **Experiment Tracking** | MLflow, W&B | Track experiments, compare models |
| **Model Registry** | MLflow, W&B | Version and manage models |
| **Data Versioning** | DVC, LakeFS | Version datasets |
| **Data Validation** | Great Expectations | Validate data quality |
| **CI/CD** | GitHub Actions, GitLab CI | Automate testing and deployment |
| **Serving** | FastAPI, TorchServe, TensorFlow Serving | Serve models |
| **Monitoring** | Prometheus, Evidently AI | Monitor performance and drift |
| **Feature Store** | Feast, Tecton | Manage features |
| **Orchestration** | Airflow, Prefect, Kubeflow | Orchestrate pipelines |

---

## Key Takeaways

1. **Version Everything** - Code, data, models, experiments
2. **Automate Testing** - Unit tests, integration tests, model performance tests
3. **Monitor Continuously** - Latency, throughput, drift, errors
4. **Deploy Gradually** - Shadow mode → Canary → Full rollout
5. **Track Experiments** - MLflow or W&B for reproducibility
6. **Use Feature Stores** - Consistent features across training and serving
7. **Validate Data** - Catch issues before they affect models
8. **Set Up Alerts** - Drift, errors, performance degradation

**Production Checklist:**
- [ ] Experiment tracking configured
- [ ] Model versioning in place
- [ ] Data versioning set up
- [ ] CI/CD pipeline automated
- [ ] Monitoring and alerting active
- [ ] Gradual deployment strategy
- [ ] Rollback plan documented
- [ ] Feature store (if applicable)
- [ ] A/B testing framework
- [ ] Documentation updated

**Next Steps:**
- Set up MLflow for experiment tracking
- Implement CI/CD pipeline for your model
- Add monitoring and alerting
- Deploy with canary or shadow mode
- Build feature store for real-time features
