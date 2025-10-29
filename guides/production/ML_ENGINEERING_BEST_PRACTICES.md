# ML Engineering Best Practices

**A Comprehensive Guide to Production Machine Learning Engineering**

*Last Updated: October 26, 2025*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure & Organization](#1-project-structure--organization)
3. [Code Quality & Testing](#2-code-quality--testing)
4. [Data Management](#3-data-management)
5. [Model Development Lifecycle](#4-model-development-lifecycle)
6. [Experiment Tracking & Reproducibility](#5-experiment-tracking--reproducibility)
7. [Production Deployment](#6-production-deployment)
8. [Monitoring & Maintenance](#7-monitoring--maintenance)
9. [Performance Optimization](#8-performance-optimization)
10. [Security & Compliance](#9-security--compliance)
11. [Collaboration & Documentation](#10-collaboration--documentation)
12. [Common Pitfalls & Solutions](#11-common-pitfalls--solutions)
13. [Checklist for Production ML](#12-checklist-for-production-ml)
14. [Interview Questions](#13-interview-questions)

---

## Introduction

This guide synthesizes best practices from industry-leading ML engineering teams (Google, Meta, Netflix, Uber) and provides actionable guidance for building production ML systems.

**Key Principles:**
1. **Simplicity First** - Start simple, add complexity only when needed
2. **Measure Everything** - You can't improve what you don't measure
3. **Fail Fast** - Catch issues early in development
4. **Automate Relentlessly** - Reduce manual work and human error
5. **Design for Failure** - Systems will fail; plan for it
6. **Iterate Quickly** - Ship fast, learn, improve

**Target Audience:**
- ML Engineers transitioning to production roles
- Data Scientists moving into ML Engineering
- Software Engineers entering ML
- Anyone building production ML systems

---

## 1. Project Structure & Organization

### 1.1 Standard ML Project Structure

A well-organized project structure is critical for maintainability and collaboration.

**Recommended Structure:**

```
ml-project/
├── README.md                 # Project overview, setup instructions
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── .gitignore               # Git ignore rules
├── .env.example             # Environment variables template
│
├── configs/                 # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── deployment_config.yaml
│
├── data/                    # Data directory (add to .gitignore)
│   ├── raw/                # Original, immutable data
│   ├── processed/          # Cleaned, transformed data
│   └── features/           # Feature store or engineered features
│
├── notebooks/              # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── data/              # Data processing
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── preprocessing.py
│   │   └── validation.py
│   │
│   ├── features/          # Feature engineering
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   └── feature_store.py
│   │
│   ├── models/            # Model definitions
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── neural_network.py
│   │   └── ensemble.py
│   │
│   ├── training/          # Training logic
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── serving/           # Model serving
│   │   ├── __init__.py
│   │   ├── api.py
│   │   └── predictor.py
│   │
│   └── utils/             # Utility functions
│       ├── __init__.py
│       ├── logging.py
│       └── metrics.py
│
├── tests/                 # Unit and integration tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
│
├── models/                # Saved model artifacts (add to .gitignore)
│   └── .gitkeep
│
├── scripts/              # Standalone scripts
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── deploy_model.py
│
├── docker/               # Docker configurations
│   ├── Dockerfile.training
│   ├── Dockerfile.serving
│   └── docker-compose.yml
│
├── deployment/           # Deployment configurations
│   ├── kubernetes/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── terraform/
│       └── main.tf
│
├── monitoring/           # Monitoring dashboards
│   └── grafana/
│       └── dashboard.json
│
└── docs/                # Documentation
    ├── architecture.md
    ├── api_reference.md
    └── deployment_guide.md
```

### 1.2 Configuration Management

**Best Practice: Use configuration files, not hard-coded values**

```python
# configs/training_config.yaml
model:
  name: "recommendation_model"
  architecture: "collaborative_filtering"
  embedding_dim: 128
  hidden_layers: [256, 128, 64]

training:
  batch_size: 256
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10

data:
  train_path: "data/processed/train.csv"
  val_path: "data/processed/val.csv"
  test_path: "data/processed/test.csv"

experiment:
  name: "experiment_v1"
  tracking_uri: "http://localhost:5000"
```

```python
# src/utils/config.py
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration loader with validation"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_config(self):
        """Validate required fields"""
        required_fields = ['model', 'training', 'data']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

    def get(self, key: str, default=None):
        """Get config value with dot notation (e.g., 'model.name')"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value

# Usage
config = Config('configs/training_config.yaml')
batch_size = config.get('training.batch_size')  # 256
model_name = config.get('model.name')  # "recommendation_model"
```

### 1.3 Environment Management

**Use .env files for secrets and environment-specific configs:**

```bash
# .env.example (commit this)
MLFLOW_TRACKING_URI=http://localhost:5000
AWS_REGION=us-west-2
MODEL_BUCKET=my-ml-models
DATABASE_URL=postgresql://user:pass@localhost:5432/mldb
API_KEY=your-api-key-here

# .env (DO NOT commit this - add to .gitignore)
MLFLOW_TRACKING_URI=http://mlflow.company.com
AWS_REGION=us-west-2
MODEL_BUCKET=prod-ml-models
DATABASE_URL=postgresql://user:secretpass@prod-db:5432/mldb
API_KEY=actual-secret-key
```

```python
# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI')
AWS_REGION = os.getenv('AWS_REGION', 'us-west-2')  # Default value
```

---

## 2. Code Quality & Testing

### 2.1 Code Quality Standards

**Follow Python best practices:**

```python
# ✅ GOOD: Clean, documented, type-hinted code
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

def preprocess_features(
    df: pd.DataFrame,
    numeric_columns: List[str],
    categorical_columns: List[str],
    fill_missing: bool = True
) -> pd.DataFrame:
    """
    Preprocess features for ML model.

    Args:
        df: Input dataframe
        numeric_columns: List of numeric column names
        categorical_columns: List of categorical column names
        fill_missing: Whether to fill missing values

    Returns:
        Preprocessed dataframe

    Raises:
        ValueError: If required columns are missing
    """
    # Validate inputs
    missing_cols = set(numeric_columns + categorical_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Process numeric columns
    if fill_missing:
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Process categorical columns
    df[categorical_columns] = df[categorical_columns].fillna('unknown')

    return df


# ❌ BAD: No types, no docs, unclear logic
def process(d, n, c):
    d[n] = d[n].fillna(d[n].median())
    d[c] = d[c].fillna('unknown')
    return d
```

**Use linting and formatting tools:**

```bash
# Install tools
pip install black isort flake8 mypy pylint

# Format code
black src/
isort src/

# Check code quality
flake8 src/
mypy src/
pylint src/
```

**Pre-commit hooks (`.pre-commit-config.yaml`):**

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100', '--ignore=E203,W503']
```

### 2.2 Testing Strategy

**Test Pyramid:**
- **70%** Unit tests (fast, isolated)
- **20%** Integration tests (moderate speed)
- **10%** End-to-end tests (slow, full system)

**Unit Testing Example:**

```python
# tests/test_preprocessing.py
import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import preprocess_features

class TestPreprocessing:
    """Test suite for data preprocessing"""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe for testing"""
        return pd.DataFrame({
            'age': [25, 30, np.nan, 40],
            'salary': [50000, 60000, 70000, np.nan],
            'city': ['NYC', 'LA', np.nan, 'SF']
        })

    def test_numeric_missing_filled_with_median(self, sample_data):
        """Test that numeric missing values are filled with median"""
        result = preprocess_features(
            sample_data,
            numeric_columns=['age', 'salary'],
            categorical_columns=['city'],
            fill_missing=True
        )

        # Check no missing values in numeric columns
        assert result[['age', 'salary']].isna().sum().sum() == 0

        # Check median was used
        assert result['age'].iloc[2] == 30.0  # median of [25, 30, 40]

    def test_categorical_missing_filled_with_unknown(self, sample_data):
        """Test that categorical missing values filled with 'unknown'"""
        result = preprocess_features(
            sample_data,
            numeric_columns=['age', 'salary'],
            categorical_columns=['city']
        )

        assert result['city'].iloc[2] == 'unknown'

    def test_raises_error_on_missing_columns(self, sample_data):
        """Test that error is raised when columns are missing"""
        with pytest.raises(ValueError, match="Missing columns"):
            preprocess_features(
                sample_data,
                numeric_columns=['age', 'nonexistent_col'],
                categorical_columns=['city']
            )
```

**Model Testing Example:**

```python
# tests/test_model.py
import pytest
import torch
from src.models.neural_network import RecommendationModel

class TestRecommendationModel:
    """Test suite for recommendation model"""

    @pytest.fixture
    def model(self):
        """Create model instance"""
        return RecommendationModel(
            num_users=1000,
            num_items=5000,
            embedding_dim=64
        )

    def test_model_forward_pass_shape(self, model):
        """Test output shape of forward pass"""
        batch_size = 32
        user_ids = torch.randint(0, 1000, (batch_size,))
        item_ids = torch.randint(0, 5000, (batch_size,))

        output = model(user_ids, item_ids)

        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()

    def test_model_embeddings_initialized(self, model):
        """Test that embeddings are properly initialized"""
        user_emb = model.user_embedding.weight
        item_emb = model.item_embedding.weight

        # Check shapes
        assert user_emb.shape == (1000, 64)
        assert item_emb.shape == (5000, 64)

        # Check not all zeros or ones
        assert not torch.all(user_emb == 0)
        assert not torch.all(user_emb == 1)

    def test_model_trainable_parameters(self, model):
        """Test that model has trainable parameters"""
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        assert trainable_params > 0
```

**Integration Testing (API):**

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.serving.api import app

class TestPredictionAPI:
    """Test suite for prediction API"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_prediction_endpoint_valid_input(self, client):
        """Test prediction with valid input"""
        payload = {
            "user_id": 123,
            "item_ids": [1, 2, 3, 4, 5],
            "top_k": 3
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        assert all("item_id" in p and "score" in p for p in data["predictions"])

    def test_prediction_endpoint_invalid_input(self, client):
        """Test prediction with invalid input"""
        payload = {
            "user_id": -1,  # Invalid user ID
            "item_ids": [],
            "top_k": 3
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
```

**Run tests:**

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::TestRecommendationModel::test_model_forward_pass_shape

# Run with verbose output
pytest tests/ -v
```

### 2.3 Continuous Integration

**GitHub Actions CI/CD (`.github/workflows/ci.yml`):**

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov black flake8

    - name: Check code formatting
      run: |
        black --check src/

    - name: Lint with flake8
      run: |
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics

    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

---

## 3. Data Management

### 3.1 Data Versioning

**Use DVC (Data Version Control) for data versioning:**

```bash
# Install DVC
pip install dvc dvc-s3

# Initialize DVC
dvc init

# Add data to DVC
dvc add data/raw/dataset.csv

# Push to remote storage
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc push

# Pull data
dvc pull

# Switch to different data version
git checkout v1.0.0
dvc checkout
```

**Track data lineage:**

```python
# src/data/versioning.py
import hashlib
import json
from pathlib import Path
from typing import Dict, Any

class DataVersionTracker:
    """Track data versions and lineage"""

    def __init__(self, tracking_dir: str = "data/versions"):
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

    def compute_hash(self, file_path: str) -> str:
        """Compute file hash for versioning"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def track_dataset(
        self,
        dataset_name: str,
        file_path: str,
        metadata: Dict[str, Any]
    ):
        """Track dataset version"""
        file_hash = self.compute_hash(file_path)

        version_info = {
            'dataset_name': dataset_name,
            'file_path': str(file_path),
            'file_hash': file_hash,
            'metadata': metadata,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Save version info
        version_file = self.tracking_dir / f"{dataset_name}_{file_hash[:8]}.json"
        with open(version_file, 'w') as f:
            json.dump(version_info, f, indent=2)

        return file_hash

# Usage
tracker = DataVersionTracker()
version_hash = tracker.track_dataset(
    dataset_name="train_data",
    file_path="data/processed/train.csv",
    metadata={
        'preprocessing': 'v2',
        'feature_engineering': 'v1',
        'num_samples': 100000
    }
)
```

### 3.2 Data Validation

**Validate data quality with Great Expectations:**

```python
# src/data/validation.py
import great_expectations as ge
import pandas as pd
from typing import Dict, List

class DataValidator:
    """Validate data quality"""

    def __init__(self):
        self.expectations = []

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        expectations: List[Dict]
    ) -> Dict:
        """
        Validate dataframe against expectations

        Args:
            df: Input dataframe
            expectations: List of expectation configs

        Returns:
            Validation results
        """
        # Convert to GE dataframe
        ge_df = ge.from_pandas(df)

        # Apply expectations
        results = []
        for exp in expectations:
            exp_type = exp['type']
            exp_kwargs = exp.get('kwargs', {})

            if exp_type == 'expect_column_values_to_not_be_null':
                result = ge_df.expect_column_values_to_not_be_null(**exp_kwargs)
            elif exp_type == 'expect_column_values_to_be_between':
                result = ge_df.expect_column_values_to_be_between(**exp_kwargs)
            elif exp_type == 'expect_column_values_to_be_in_set':
                result = ge_df.expect_column_values_to_be_in_set(**exp_kwargs)

            results.append(result)

        # Summary
        all_passed = all(r.success for r in results)

        return {
            'all_passed': all_passed,
            'num_expectations': len(results),
            'num_passed': sum(r.success for r in results),
            'results': results
        }

# Usage
validator = DataValidator()

expectations = [
    {
        'type': 'expect_column_values_to_not_be_null',
        'kwargs': {'column': 'user_id'}
    },
    {
        'type': 'expect_column_values_to_be_between',
        'kwargs': {'column': 'age', 'min_value': 0, 'max_value': 120}
    },
    {
        'type': 'expect_column_values_to_be_in_set',
        'kwargs': {'column': 'status', 'value_set': ['active', 'inactive', 'pending']}
    }
]

validation_results = validator.validate_dataframe(df, expectations)

if not validation_results['all_passed']:
    raise ValueError("Data validation failed!")
```

### 3.3 Data Privacy & Security

**Implement PII detection and masking:**

```python
# src/data/privacy.py
import re
from typing import List, Dict
import pandas as pd

class PIIDetector:
    """Detect and mask PII in data"""

    # Regex patterns for PII
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
    }

    def detect_pii_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect columns containing PII"""
        pii_columns = {}

        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                sample = df[col].dropna().astype(str).head(100)

                for pii_type, pattern in self.PATTERNS.items():
                    if sample.str.contains(pattern, regex=True).any():
                        if pii_type not in pii_columns:
                            pii_columns[pii_type] = []
                        pii_columns[pii_type].append(col)

        return pii_columns

    def mask_pii(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Mask PII in specified columns"""
        df_masked = df.copy()

        for col in columns:
            if col in df_masked.columns:
                # Email: show first 2 chars + domain
                df_masked[col] = df_masked[col].apply(self._mask_email)

        return df_masked

    def _mask_email(self, email: str) -> str:
        """Mask email address"""
        if pd.isna(email) or '@' not in email:
            return email

        local, domain = email.split('@')
        if len(local) <= 2:
            masked_local = '*' * len(local)
        else:
            masked_local = local[:2] + '*' * (len(local) - 2)

        return f"{masked_local}@{domain}"

# Usage
detector = PIIDetector()
pii_cols = detector.detect_pii_columns(df)
print(f"PII detected: {pii_cols}")

# Mask PII before logging/sharing
df_safe = detector.mask_pii(df, pii_cols.get('email', []))
```

---

## 4. Model Development Lifecycle

### 4.1 Start Simple, Add Complexity

**Always start with a simple baseline:**

```python
# Step 1: Simple baseline (Day 1)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Predict most frequent class
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
print(f"Baseline accuracy: {baseline_acc:.3f}")

# Step 2: Simple linear model (Day 1-2)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
print(f"Logistic Regression accuracy: {lr_acc:.3f}")

# Step 3: Tree-based model (Day 3-5)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest accuracy: {rf_acc:.3f}")

# Step 4: Advanced model (Week 2+)
import xgboost as xgb

xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"XGBoost accuracy: {xgb_acc:.3f}")

# Only proceed to neural networks if significant improvement expected
```

**Rule of Thumb:**
- If linear model achieves 90% of ideal performance → use it
- If tree-based model achieves 95% → use it
- Only use deep learning if simpler models fail

### 4.2 Feature Engineering Pipeline

```python
# src/features/feature_engineering.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer"""

    def __init__(self, create_interactions=True):
        self.create_interactions = create_interactions
        self.feature_names_ = None

    def fit(self, X, y=None):
        """Learn feature names"""
        self.feature_names_ = X.columns.tolist()
        return self

    def transform(self, X):
        """Apply feature engineering"""
        X = X.copy()

        # Create interaction features
        if self.create_interactions:
            X['age_income_interaction'] = X['age'] * X['income']
            X['age_squared'] = X['age'] ** 2

        # Create binned features
        X['age_group'] = pd.cut(
            X['age'],
            bins=[0, 25, 45, 65, 100],
            labels=['young', 'adult', 'senior', 'elderly']
        )

        # Create aggregation features
        X['total_spending'] = X[['groceries', 'utilities', 'entertainment']].sum(axis=1)

        return X

# Build complete pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['city', 'occupation']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipeline = Pipeline([
    ('feature_eng', FeatureEngineering()),
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### 4.3 Hyperparameter Tuning

**Use Bayesian optimization for efficiency:**

```python
# src/training/hyperparameter_tuning.py
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def objective(trial):
    """Optuna objective function"""
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    }

    # Train model with CV
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='f1_weighted', n_jobs=-1
    ).mean()

    return score

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=3600)

print(f"Best score: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Train final model with best params
best_model = RandomForestClassifier(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)
```

---

## 5. Experiment Tracking & Reproducibility

### 5.1 MLflow Integration

```python
# src/training/train_with_mlflow.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_model_with_tracking(params, X_train, y_train, X_test, y_test):
    """Train model with MLflow tracking"""

    with mlflow.start_run(run_name="rf_experiment"):
        # Log parameters
        mlflow.log_params(params)

        # Log dataset info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log confusion matrix as artifact
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')

        # Log feature importance
        import pandas as pd
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')

        return model, metrics

# Usage
params = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

model, metrics = train_model_with_tracking(params, X_train, y_train, X_test, y_test)
```

### 5.2 Reproducibility Checklist

**Ensure reproducible experiments:**

```python
# Set all random seeds
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cudnn deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

**Pin dependencies (requirements.txt):**

```text
# requirements.txt - Use exact versions for reproducibility
numpy==1.24.3
pandas==2.0.2
scikit-learn==1.3.0
torch==2.0.1
transformers==4.30.2
mlflow==2.4.1
```

**Log everything:**

```python
# src/utils/experiment_logger.py
import json
import sys
from pathlib import Path
from datetime import datetime

class ExperimentLogger:
    """Log experiment details for reproducibility"""

    def __init__(self, log_dir='experiments'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def log_environment(self):
        """Log environment information"""
        import platform
        import torch

        env_info = {
            'timestamp': self.timestamp,
            'python_version': sys.version,
            'platform': platform.platform(),
            'cpu_count': os.cpu_count(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        }

        log_file = self.log_dir / f'environment_{self.timestamp}.json'
        with open(log_file, 'w') as f:
            json.dump(env_info, f, indent=2)

        return env_info

    def log_git_commit(self):
        """Log git commit hash for code versioning"""
        try:
            import subprocess
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']
            ).decode('utf-8').strip()

            git_info = {
                'commit_hash': commit_hash,
                'timestamp': self.timestamp
            }

            log_file = self.log_dir / f'git_info_{self.timestamp}.json'
            with open(log_file, 'w') as f:
                json.dump(git_info, f, indent=2)

            return git_info
        except Exception as e:
            print(f"Could not log git info: {e}")
            return None

# Usage
logger = ExperimentLogger()
logger.log_environment()
logger.log_git_commit()
```

---

## 6. Production Deployment

### 6.1 Model Serving with FastAPI

```python
# src/serving/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np
import joblib
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

# Load model at startup
model = joblib.load('models/model.pkl')

app = FastAPI(
    title="ML Prediction API",
    description="Production ML model serving",
    version="1.0.0"
)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'predictions_total',
    'Total number of predictions'
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds'
)

# Request/Response models
class PredictionRequest(BaseModel):
    """Prediction request schema"""
    features: List[float] = Field(..., min_items=10, max_items=10)
    user_id: Optional[str] = None

    @validator('features')
    def validate_features(cls, v):
        """Validate feature values"""
        if any(np.isnan(val) for val in v):
            raise ValueError("Features cannot contain NaN values")
        return v

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    prediction: float
    confidence: float
    latency_ms: float
    model_version: str = "1.0.0"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction"""
    start_time = time.time()

    try:
        # Reshape features
        features = np.array(request.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features).max()

        # Calculate latency
        latency = (time.time() - start_time) * 1000  # ms

        # Update metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(latency / 1000)

        return PredictionResponse(
            prediction=float(prediction),
            confidence=float(confidence),
            latency_ms=latency
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "metrics": "/metrics"
        }
    }
```

### 6.2 Containerization

**Dockerfile for serving:**

```dockerfile
# docker/Dockerfile.serving
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose for local testing:**

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.serving
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/model.pkl
      - LOG_LEVEL=INFO
    volumes:
      - ../models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ../monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ../monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ../monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
```

### 6.3 Kubernetes Deployment

```yaml
# deployment/kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  labels:
    app: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: your-registry/ml-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models/model.pkl"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  selector:
    app: ml-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 7. Monitoring & Maintenance

### 7.1 Model Performance Monitoring

```python
# src/monitoring/model_monitor.py
from prometheus_client import Gauge, Histogram, Counter
import numpy as np
from collections import deque
from typing import Deque, Optional

class ModelPerformanceMonitor:
    """Monitor model performance in production"""

    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size

        # Prediction buffer for batch evaluation
        self.predictions: Deque = deque(maxlen=buffer_size)
        self.ground_truth: Deque = deque(maxlen=buffer_size)

        # Prometheus metrics
        self.accuracy_gauge = Gauge('model_accuracy', 'Current model accuracy')
        self.latency_hist = Histogram(
            'prediction_latency_seconds',
            'Prediction latency'
        )
        self.error_counter = Counter('prediction_errors', 'Number of errors')

    def log_prediction(
        self,
        prediction: float,
        ground_truth: Optional[float] = None,
        latency: float = 0.0
    ):
        """Log individual prediction"""
        self.predictions.append(prediction)

        if ground_truth is not None:
            self.ground_truth.append(ground_truth)

        # Update latency
        if latency > 0:
            self.latency_hist.observe(latency)

        # Evaluate batch when buffer is full
        if len(self.predictions) >= self.buffer_size and len(self.ground_truth) >= self.buffer_size:
            self._evaluate_batch()

    def _evaluate_batch(self):
        """Evaluate model on buffered predictions"""
        if len(self.ground_truth) == 0:
            return

        # Calculate accuracy
        predictions_arr = np.array(list(self.predictions))
        ground_truth_arr = np.array(list(self.ground_truth))

        accuracy = np.mean(predictions_arr == ground_truth_arr)

        # Update Prometheus gauge
        self.accuracy_gauge.set(accuracy)

        print(f"Batch accuracy: {accuracy:.4f}")

    def log_error(self, error_type: str):
        """Log prediction error"""
        self.error_counter.inc()

# Usage in API
monitor = ModelPerformanceMonitor()

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()

    try:
        prediction = model.predict(...)
        latency = time.time() - start_time

        # Log prediction
        monitor.log_prediction(
            prediction=prediction,
            latency=latency
        )

        return {"prediction": prediction}

    except Exception as e:
        monitor.log_error(error_type=type(e).__name__)
        raise
```

### 7.2 Data Drift Detection

```python
# src/monitoring/drift_detection.py
from scipy import stats
import numpy as np
import pandas as pd
from typing import Dict, List

class DataDriftDetector:
    """Detect data drift in production"""

    def __init__(self, reference_data: pd.DataFrame):
        """
        Args:
            reference_data: Training data distribution
        """
        self.reference_data = reference_data
        self.reference_stats = self._compute_stats(reference_data)

    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """Compute statistical summary"""
        stats_dict = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q25': df[col].quantile(0.25),
                'q50': df[col].quantile(0.50),
                'q75': df[col].quantile(0.75)
            }

        return stats_dict

    def detect_drift_ks_test(
        self,
        current_data: pd.DataFrame,
        significance_level: float = 0.05
    ) -> Dict[str, Dict]:
        """
        Detect drift using Kolmogorov-Smirnov test

        Returns:
            Dictionary with drift detection results per feature
        """
        drift_results = {}

        numeric_cols = current_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in self.reference_data.columns:
                continue

            # Perform KS test
            statistic, p_value = stats.ks_2samp(
                self.reference_data[col].dropna(),
                current_data[col].dropna()
            )

            drift_results[col] = {
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': p_value < significance_level,
                'reference_mean': self.reference_stats[col]['mean'],
                'current_mean': current_data[col].mean(),
                'mean_change_pct': (
                    (current_data[col].mean() - self.reference_stats[col]['mean']) /
                    self.reference_stats[col]['mean'] * 100
                )
            }

        return drift_results

    def detect_drift_psi(
        self,
        current_data: pd.DataFrame,
        num_bins: int = 10,
        threshold: float = 0.2
    ) -> Dict[str, Dict]:
        """
        Detect drift using Population Stability Index (PSI)

        PSI Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change (drift)
        """
        drift_results = {}

        numeric_cols = current_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in self.reference_data.columns:
                continue

            # Create bins based on reference data
            bins = pd.qcut(
                self.reference_data[col],
                q=num_bins,
                duplicates='drop',
                retbins=True
            )[1]

            # Calculate distributions
            ref_dist = pd.cut(
                self.reference_data[col],
                bins=bins
            ).value_counts(normalize=True).sort_index()

            curr_dist = pd.cut(
                current_data[col],
                bins=bins
            ).value_counts(normalize=True).reindex(ref_dist.index, fill_value=0)

            # Calculate PSI
            psi = np.sum(
                (curr_dist - ref_dist) * np.log((curr_dist + 1e-10) / (ref_dist + 1e-10))
            )

            drift_results[col] = {
                'psi': psi,
                'drift_detected': psi >= threshold,
                'severity': 'high' if psi >= 0.2 else 'moderate' if psi >= 0.1 else 'low'
            }

        return drift_results

# Usage
# Train time: save reference data
reference_data = pd.read_csv('data/train.csv')
detector = DataDriftDetector(reference_data)

# Production: check for drift
current_batch = pd.read_csv('data/production_batch.csv')
ks_drift = detector.detect_drift_ks_test(current_batch)
psi_drift = detector.detect_drift_psi(current_batch)

# Alert if drift detected
for feature, result in psi_drift.items():
    if result['drift_detected']:
        print(f"ALERT: Drift detected in {feature}! PSI={result['psi']:.3f}")
```

---

## 8. Performance Optimization

### 8.1 Model Optimization

**Quantization for faster inference:**

```python
# src/optimization/quantization.py
import torch
import torch.ao.quantization as quantization

def quantize_model(model: torch.nn.Module, backend='fbgemm'):
    """
    Quantize PyTorch model for faster inference

    Args:
        model: PyTorch model
        backend: 'fbgemm' (CPU) or 'qnnpack' (mobile)

    Returns:
        Quantized model
    """
    # Set backend
    torch.backends.quantized.engine = backend

    # Prepare for quantization
    model.eval()
    model.qconfig = quantization.get_default_qconfig(backend)
    model_prepared = quantization.prepare(model)

    # Calibrate with sample data
    # (run representative data through model)
    # model_prepared(calibration_data)

    # Convert to quantized model
    model_quantized = quantization.convert(model_prepared)

    return model_quantized

# Dynamic quantization (simpler, good for RNNs/Transformers)
def dynamic_quantize_model(model: torch.nn.Module):
    """Dynamic quantization - quantize weights, activations stay float"""
    model_quantized = quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM},  # Layers to quantize
        dtype=torch.qint8
    )
    return model_quantized

# Compare performance
model_fp32 = MyModel()
model_int8 = dynamic_quantize_model(model_fp32)

# Size comparison
fp32_size = os.path.getsize('model_fp32.pth')
int8_size = os.path.getsize('model_int8.pth')
print(f"Size reduction: {fp32_size / int8_size:.2f}x")

# Speed comparison
import time

start = time.time()
for _ in range(1000):
    _ = model_fp32(sample_input)
fp32_time = time.time() - start

start = time.time()
for _ in range(1000):
    _ = model_int8(sample_input)
int8_time = time.time() - start

print(f"Speedup: {fp32_time / int8_time:.2f}x")
```

### 8.2 Batch Prediction Optimization

```python
# src/optimization/batch_predictor.py
import asyncio
from collections import deque
from typing import List, Any
import numpy as np

class BatchPredictor:
    """
    Batch predictions for better throughput
    Collects individual requests and processes in batches
    """

    def __init__(
        self,
        model,
        max_batch_size: int = 32,
        max_wait_time: float = 0.01  # 10ms
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time

        self.queue = deque()
        self.batch_task = None

    async def predict(self, features: np.ndarray) -> np.ndarray:
        """Add prediction to batch queue"""
        # Create future for this request
        future = asyncio.Future()

        # Add to queue
        self.queue.append((features, future))

        # Start batch processor if not running
        if self.batch_task is None or self.batch_task.done():
            self.batch_task = asyncio.create_task(self._process_batch())

        # Wait for result
        return await future

    async def _process_batch(self):
        """Process batch of predictions"""
        await asyncio.sleep(self.max_wait_time)

        if not self.queue:
            return

        # Collect batch
        batch = []
        futures = []

        while self.queue and len(batch) < self.max_batch_size:
            features, future = self.queue.popleft()
            batch.append(features)
            futures.append(future)

        # Process batch
        try:
            batch_features = np.vstack(batch)
            predictions = self.model.predict(batch_features)

            # Return results
            for future, prediction in zip(futures, predictions):
                future.set_result(prediction)

        except Exception as e:
            # Propagate error to all futures
            for future in futures:
                future.set_exception(e)

# Usage in FastAPI
batch_predictor = BatchPredictor(model)

@app.post("/predict")
async def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = await batch_predictor.predict(features)
    return {"prediction": float(prediction[0])}
```

### 8.3 Caching

```python
# src/optimization/caching.py
from functools import lru_cache
import hashlib
import json
import redis
from typing import Any, Optional

class PredictionCache:
    """Cache predictions for frequently requested inputs"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour

    def _hash_features(self, features: np.ndarray) -> str:
        """Create hash of features for cache key"""
        # Round to reduce sensitivity
        rounded = np.round(features, decimals=4)
        features_bytes = rounded.tobytes()
        return hashlib.sha256(features_bytes).hexdigest()

    def get(self, features: np.ndarray) -> Optional[Any]:
        """Get cached prediction"""
        key = self._hash_features(features)
        cached = self.redis_client.get(f"pred:{key}")

        if cached:
            return json.loads(cached)
        return None

    def set(self, features: np.ndarray, prediction: Any):
        """Cache prediction"""
        key = self._hash_features(features)
        self.redis_client.setex(
            f"pred:{key}",
            self.ttl,
            json.dumps(prediction)
        )

# Usage
cache = PredictionCache()

@app.post("/predict")
async def predict(request: PredictionRequest):
    features = np.array(request.features)

    # Check cache
    cached_pred = cache.get(features)
    if cached_pred is not None:
        return {"prediction": cached_pred, "from_cache": True}

    # Make prediction
    prediction = model.predict(features.reshape(1, -1))[0]

    # Cache result
    cache.set(features, float(prediction))

    return {"prediction": float(prediction), "from_cache": False}
```

---

## 9. Security & Compliance

### 9.1 API Security

```python
# src/serving/security.py
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
import secrets

# API Key authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Store API keys securely (use database in production)
VALID_API_KEYS = {
    "your-api-key-here": {"client": "client1", "tier": "premium"},
    "another-key": {"client": "client2", "tier": "basic"}
}

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Verify API key"""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return VALID_API_KEYS[api_key]

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")  # 100 requests per minute
async def predict(
    request: PredictionRequest,
    api_key_info: dict = Depends(verify_api_key)
):
    """Make prediction with rate limiting and auth"""
    # Prediction logic
    pass

# Input validation
from pydantic import BaseModel, validator

class SecurePredictionRequest(BaseModel):
    features: List[float]

    @validator('features')
    def validate_features(cls, v):
        # Check for attacks
        if len(v) > 1000:
            raise ValueError("Too many features (max 1000)")

        # Check for injection attempts
        if any(not isinstance(f, (int, float)) for f in v):
            raise ValueError("Features must be numeric")

        # Check ranges
        if any(abs(f) > 1e10 for f in v):
            raise ValueError("Feature values out of range")

        return v
```

### 9.2 Model Security

```python
# src/security/model_security.py
import hashlib
import hmac

class ModelSecurityManager:
    """Ensure model integrity"""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()

    def sign_model(self, model_path: str) -> str:
        """Create signature for model file"""
        with open(model_path, 'rb') as f:
            model_bytes = f.read()

        signature = hmac.new(
            self.secret_key,
            model_bytes,
            hashlib.sha256
        ).hexdigest()

        # Save signature
        with open(f"{model_path}.sig", 'w') as f:
            f.write(signature)

        return signature

    def verify_model(self, model_path: str) -> bool:
        """Verify model hasn't been tampered with"""
        # Load signature
        try:
            with open(f"{model_path}.sig", 'r') as f:
                expected_sig = f.read().strip()
        except FileNotFoundError:
            return False

        # Compute current signature
        with open(model_path, 'rb') as f:
            model_bytes = f.read()

        current_sig = hmac.new(
            self.secret_key,
            model_bytes,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(current_sig, expected_sig)

# Usage
security = ModelSecurityManager(secret_key=os.getenv('MODEL_SECRET_KEY'))

# Sign model before deployment
security.sign_model('models/model.pkl')

# Verify before loading
if not security.verify_model('models/model.pkl'):
    raise RuntimeError("Model verification failed! Possible tampering detected.")

model = joblib.load('models/model.pkl')
```

---

## 10. Collaboration & Documentation

### 10.1 Code Documentation

**Write clear docstrings:**

```python
# src/models/recommendation.py
from typing import List, Tuple, Optional
import numpy as np

class CollaborativeFilteringModel:
    """
    Collaborative Filtering recommendation model using matrix factorization.

    This model learns latent factors for users and items to predict ratings.
    Uses Alternating Least Squares (ALS) for optimization.

    Attributes:
        num_factors (int): Number of latent factors
        regularization (float): L2 regularization parameter
        iterations (int): Number of ALS iterations
        user_factors (np.ndarray): Learned user factor matrix (num_users x num_factors)
        item_factors (np.ndarray): Learned item factor matrix (num_items x num_factors)

    Example:
        >>> model = CollaborativeFilteringModel(num_factors=50)
        >>> model.fit(user_item_matrix)
        >>> recommendations = model.recommend(user_id=42, top_k=10)

    References:
        - Koren et al. "Matrix Factorization Techniques for Recommender Systems" (2009)
        - Hu et al. "Collaborative Filtering for Implicit Feedback Datasets" (2008)
    """

    def __init__(
        self,
        num_factors: int = 50,
        regularization: float = 0.01,
        iterations: int = 15
    ):
        """
        Initialize collaborative filtering model.

        Args:
            num_factors: Number of latent factors to learn. Higher values can
                capture more complex patterns but may overfit. Typical range: 20-200.
            regularization: L2 regularization strength. Prevents overfitting by
                penalizing large factor values. Typical range: 0.001-0.1.
            iterations: Number of ALS iterations. More iterations improve convergence
                but take longer. Typical range: 10-30.

        Raises:
            ValueError: If num_factors <= 0 or regularization < 0
        """
        if num_factors <= 0:
            raise ValueError("num_factors must be positive")
        if regularization < 0:
            raise ValueError("regularization must be non-negative")

        self.num_factors = num_factors
        self.regularization = regularization
        self.iterations = iterations

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(self, user_item_matrix: np.ndarray) -> 'CollaborativeFilteringModel':
        """
        Fit the model using Alternating Least Squares.

        The algorithm alternates between:
        1. Fixing item factors, solving for user factors
        2. Fixing user factors, solving for item factors

        Time Complexity: O(iterations * (num_users + num_items) * num_factors^2)
        Space Complexity: O((num_users + num_items) * num_factors)

        Args:
            user_item_matrix: Interaction matrix of shape (num_users, num_items).
                Non-zero values indicate interactions (e.g., ratings, clicks).

        Returns:
            self: Fitted model

        Raises:
            ValueError: If user_item_matrix is empty or has wrong shape

        Example:
            >>> ratings = np.array([[5, 3, 0], [4, 0, 2], [0, 1, 5]])
            >>> model.fit(ratings)
        """
        num_users, num_items = user_item_matrix.shape

        # Initialize factors randomly
        self.user_factors = np.random.normal(
            scale=0.01,
            size=(num_users, self.num_factors)
        )
        self.item_factors = np.random.normal(
            scale=0.01,
            size=(num_items, self.num_factors)
        )

        # ALS iterations
        for iteration in range(self.iterations):
            # Update user factors
            self._update_user_factors(user_item_matrix)

            # Update item factors
            self._update_item_factors(user_item_matrix)

        return self

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_known: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate top-K recommendations for a user.

        Args:
            user_id: ID of user to generate recommendations for
            top_k: Number of recommendations to return
            exclude_known: If True, exclude items user has already interacted with

        Returns:
            List of (item_id, score) tuples, sorted by score descending

        Raises:
            ValueError: If model hasn't been fitted or user_id is invalid

        Example:
            >>> recommendations = model.recommend(user_id=42, top_k=5)
            >>> for item_id, score in recommendations:
            >>>     print(f"Item {item_id}: score = {score:.3f}")
        """
        if self.user_factors is None:
            raise ValueError("Model must be fitted before making recommendations")

        if user_id < 0 or user_id >= len(self.user_factors):
            raise ValueError(f"Invalid user_id: {user_id}")

        # Compute scores for all items
        user_vector = self.user_factors[user_id]
        scores = self.item_factors @ user_vector

        # Get top-K items
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]
```

### 10.2 Model Cards

**Document model details for stakeholders:**

```markdown
# Model Card: Product Recommendation System

## Model Details
- **Model Name:** CollaborativeFilteringRecommender
- **Version:** 1.2.0
- **Date:** 2025-10-26
- **Model Type:** Matrix Factorization (Collaborative Filtering)
- **Owner:** ML Team (ml-team@company.com)

## Intended Use
- **Primary Use:** Recommend products to users on e-commerce platform
- **Intended Users:** Logged-in customers
- **Out-of-Scope:**
  - New users with no interaction history (use popularity-based cold-start model)
  - Explicit NSFW content filtering (handled by separate content moderation system)

## Training Data
- **Dataset:** User-Product interactions (Jan 2024 - Sep 2025)
- **Size:** 10M users, 500K products, 150M interactions
- **Geographic Distribution:**
  - North America: 60%
  - Europe: 25%
  - Asia: 10%
  - Other: 5%
- **Data Collection:** Click data, purchases, ratings
- **Preprocessing:**
  - Filtered users with < 5 interactions
  - Filtered products with < 10 interactions
  - Interaction values: Implicit feedback (clicks=1, cart_add=2, purchase=5)

## Evaluation Data
- **Test Set:** Held-out 20% from Oct 2025
- **Validation Strategy:** Temporal split (train on past, test on future)

## Metrics
| Metric | Value | Threshold |
|--------|-------|-----------|
| Precision@10 | 0.23 | >0.20 |
| Recall@10 | 0.15 | >0.12 |
| NDCG@10 | 0.31 | >0.28 |
| Coverage | 0.45 | >0.40 |
| Latency (p99) | 45ms | <100ms |

## Ethical Considerations
- **Bias Concerns:**
  - Popular items over-represented in recommendations
  - Mitigation: Diversity regularization in ranking
- **Privacy:**
  - No PII in model features
  - Recommendations based on aggregated behavior patterns
- **Fairness:**
  - Monitored for demographic biases
  - No protected attributes used in model

## Limitations
- **Cold Start:** Cannot recommend to users with <3 interactions
- **Popularity Bias:** Tends to recommend popular items
- **Temporal Drift:** Performance degrades without retraining (retrain weekly)
- **Data Quality:** Sensitive to bot traffic (filtered upstream)

## Monitoring
- **Performance:** Daily batch evaluation on recent data
- **Drift:** PSI monitoring on user/item distributions
- **Alerts:**
  - Precision@10 < 0.18
  - Latency p99 > 100ms
  - Error rate > 1%

## Deployment
- **Infrastructure:** Kubernetes cluster (3 replicas)
- **Serving:** FastAPI + Redis caching
- **Update Frequency:** Model retrained weekly, deployed Monday 2AM UTC
- **Rollback:** Automated rollback if error rate > 5% for 5 minutes

## Contact
- **Model Owner:** Jane Doe (jane@company.com)
- **On-Call:** ml-oncall@company.com
- **Documentation:** https://wiki.company.com/ml/recommendations
```

---

## 11. Common Pitfalls & Solutions

### Pitfall 1: Data Leakage

**Problem:** Information from test set leaks into training

```python
# ❌ BAD: Leakage through scaling
from sklearn.preprocessing import StandardScaler

# Fit scaler on entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # LEAKAGE!

# Then split
X_train, X_test = train_test_split(X_scaled, ...)

# ✅ GOOD: Fit only on training data
X_train, X_test = train_test_split(X, ...)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)  # Transform test using train stats
```

### Pitfall 2: Not Using Pipelines

```python
# ❌ BAD: Manual preprocessing
X_train_filled = fillna(X_train)
X_train_scaled = scale(X_train_filled)
model.fit(X_train_scaled, y_train)

X_test_filled = fillna(X_test)  # Easy to forget steps!
X_test_scaled = scale(X_test_filled)
pred = model.predict(X_test_scaled)

# ✅ GOOD: Pipeline ensures consistency
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)  # All steps applied automatically
```

### Pitfall 3: Ignoring Class Imbalance

```python
# ❌ BAD: Ignoring 1% positive class
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Model predicts all negatives, 99% accuracy but useless!

# ✅ GOOD: Handle imbalance
from imblearn.over_sampling import SMOTE

# Option 1: Resampling
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Option 2: Class weights
model = RandomForestClassifier(class_weight='balanced')

# Option 3: Different metric
from sklearn.metrics import f1_score
score = f1_score(y_test, y_pred, average='weighted')
```

### Pitfall 4: Not Monitoring Production

```python
# ❌ BAD: Deploy and forget
model = train_model()
deploy(model)
# No monitoring!

# ✅ GOOD: Comprehensive monitoring
@app.post("/predict")
async def predict(request: PredictionRequest):
    # Log input distribution
    monitor.log_features(request.features)

    # Make prediction
    prediction = model.predict(request.features)

    # Log prediction
    monitor.log_prediction(prediction)

    # Log latency
    monitor.log_latency(latency)

    # Check for drift (async)
    asyncio.create_task(monitor.check_drift())

    return {"prediction": prediction}
```

### Pitfall 5: Over-Optimizing Hyperparameters

```python
# ❌ BAD: 1000+ hyperparameter combinations
param_grid = {
    'n_estimators': range(50, 500, 10),  # 45 values
    'max_depth': range(3, 30),  # 27 values
    'min_samples_split': range(2, 20),  # 18 values
    # ... millions of combinations
}
# This will take days!

# ✅ GOOD: Hierarchical optimization
# Step 1: Coarse search
coarse_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
}
coarse_search = GridSearchCV(model, coarse_params)

# Step 2: Fine search around best
best_n_est = coarse_search.best_params_['n_estimators']
fine_params = {
    'n_estimators': [best_n_est-20, best_n_est, best_n_est+20],
    'max_depth': [best_depth-2, best_depth, best_depth+2],
}
```

---

## 12. Checklist for Production ML

### Pre-Development
- [ ] Define clear success metrics (what does "good" look like?)
- [ ] Establish baseline (random, heuristic, or simple model)
- [ ] Check data availability and quality
- [ ] Verify legal/privacy compliance
- [ ] Get stakeholder buy-in on timeline and scope

### Development
- [ ] Set up version control (Git)
- [ ] Create virtual environment and pin dependencies
- [ ] Set up experiment tracking (MLflow, Weights & Biases)
- [ ] Implement data validation
- [ ] Write unit tests for data processing
- [ ] Build reproducible pipelines
- [ ] Log all experiments with parameters and results
- [ ] Document model assumptions and limitations

### Pre-Deployment
- [ ] Test on production-like data
- [ ] Benchmark latency and throughput
- [ ] Create model card documenting model details
- [ ] Set up monitoring (performance, drift, errors)
- [ ] Implement logging (requests, predictions, errors)
- [ ] Add API authentication and rate limiting
- [ ] Create deployment runbook
- [ ] Plan rollback strategy
- [ ] Load test API endpoints
- [ ] Security review (input validation, model signing)

### Post-Deployment
- [ ] Monitor model performance daily
- [ ] Check for data drift weekly
- [ ] Review error logs regularly
- [ ] A/B test against previous model
- [ ] Collect ground truth for evaluation
- [ ] Retrain model periodically
- [ ] Update documentation as model evolves
- [ ] Conduct post-mortem for incidents

---

## 13. Interview Questions

### Q1: How do you prevent data leakage in ML pipelines?

**Answer:**
Data leakage occurs when information from the test set influences the training process. Prevention strategies:

1. **Split Before Preprocessing:**
   - Always split data FIRST, then preprocess train and test separately
   - Fit transformations (scaling, encoding) only on training data
   - Apply learned transformations to test data

2. **Use Pipelines:**
   - sklearn pipelines ensure consistent preprocessing
   - Cross-validation with pipelines prevents leakage

3. **Temporal Validation for Time Series:**
   - Use time-based splits, not random splits
   - Never train on future data to predict past

4. **Feature Engineering:**
   - Don't use features that wouldn't be available at prediction time
   - Example: Don't use "days until event" if predicting event occurrence

**Code Example:**
```python
# Correct approach
X_train, X_test = train_test_split(X, test_size=0.2)

scaler = StandardScaler().fit(X_train)  # Fit on train only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Even better: use pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
```

---

### Q2: How do you detect and handle model drift in production?

**Answer:**
Model drift occurs when the relationship between features and target changes over time.

**Detection Methods:**

1. **Statistical Tests:**
   - Kolmogorov-Smirnov test for feature distributions
   - Population Stability Index (PSI) for binned features
   - Chi-square test for categorical features

2. **Performance Monitoring:**
   - Track accuracy/F1/AUC over time
   - Alert when performance drops below threshold

3. **Prediction Distribution:**
   - Monitor distribution of predicted probabilities
   - Detect sudden shifts in prediction patterns

**Handling Strategies:**

1. **Automated Retraining:**
   - Schedule regular retraining (weekly/monthly)
   - Trigger retraining when drift detected

2. **Online Learning:**
   - Incrementally update model with new data
   - Use algorithms that support partial_fit()

3. **Ensemble with Time Decay:**
   - Weight recent models higher
   - Gradually phase out old models

**Code Example:**
```python
# PSI calculation
def calculate_psi(expected, actual, bins=10):
    expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)

    psi = np.sum(
        (actual_percents - expected_percents) *
        np.log((actual_percents + 1e-10) / (expected_percents + 1e-10))
    )

    # PSI > 0.2 indicates significant drift
    return psi
```

---

### Q3: What's your approach to deploying ML models to production?

**Answer:**
My deployment approach follows these principles:

**1. Containerization:**
- Package model with dependencies in Docker
- Ensures consistency across environments
- Easy to scale and deploy

**2. Serving Options:**
- **REST API** (FastAPI/Flask): Most flexible, easy integration
- **Batch Predictions**: For offline use cases
- **Streaming**: Kafka + model for real-time events

**3. Deployment Strategy:**
- **Blue-Green**: Deploy to new environment, switch traffic
- **Canary**: Gradually route traffic to new model (5% → 50% → 100%)
- **Shadow Mode**: Run new model alongside old, compare predictions

**4. Infrastructure:**
- Kubernetes for orchestration and auto-scaling
- Load balancer for distributing traffic
- Redis for caching frequent predictions

**5. Monitoring:**
- Performance metrics (latency, throughput)
- Model metrics (accuracy, drift)
- System metrics (CPU, memory, error rate)

**Example Deployment Flow:**
```
1. Train model → Save to model registry (MLflow)
2. Build Docker image with model
3. Deploy to staging → Integration tests
4. Deploy to production (canary) → Monitor 5% traffic
5. Full deployment if metrics healthy
6. Automated rollback if issues detected
```

---

### Q4: How do you handle class imbalance?

**Answer:**
Class imbalance (e.g., 1% fraud, 99% normal) requires special handling:

**Data-Level:**
1. **Oversampling minority class** (SMOTE, ADASYN)
2. **Undersampling majority class** (Random, Tomek links)
3. **Hybrid approaches** (SMOTE + ENN)

**Algorithm-Level:**
1. **Class weights**: Penalize misclassifying minority class more
2. **Threshold adjustment**: Move decision boundary
3. **Ensemble methods**: Balance datasets for each weak learner

**Metric Selection:**
- **Don't use accuracy!** (99% accuracy by predicting all majority)
- Use: F1-score, Precision, Recall, AUC-ROC, AUC-PR

**Code Example:**
```python
# Approach 1: SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

# Approach 2: Class weights
model = RandomForestClassifier(class_weight='balanced')

# Approach 3: Custom weights
class_weights = {0: 1, 1: 99}  # Penalize minority 99x more
model = RandomForestClassifier(class_weight=class_weights)

# Evaluate properly
from sklearn.metrics import f1_score, classification_report
print(classification_report(y_test, y_pred))
```

---

### Q5: Explain your process for debugging a model that isn't learning.

**Answer:**
Systematic debugging approach:

**1. Sanity Checks:**
- Can model overfit small batch? (Ensures model has capacity)
- Is loss decreasing? (Ensures optimization works)
- Are gradients flowing? (Check for vanishing/exploding gradients)

**2. Data Issues:**
- Check for label errors (mislabeled data)
- Verify data preprocessing (scaling, normalization)
- Look for data leakage (too good to be true performance)
- Check class balance

**3. Model Issues:**
- Too simple? (Increase capacity: more layers, units)
- Too complex? (Add regularization: dropout, L2)
- Wrong architecture for problem?
- Initialization issues? (Try different initialization schemes)

**4. Training Issues:**
- Learning rate too high/low? (Use learning rate finder)
- Batch size issues? (Try different batch sizes)
- Optimizer choice? (Try Adam vs SGD vs RMSprop)

**5. Loss Function:**
- Is loss function appropriate for problem?
- Are you minimizing the right objective?

**Debugging Code:**
```python
# 1. Overfit single batch (should get 100% accuracy)
single_batch = X_train[:32]
single_labels = y_train[:32]

for epoch in range(1000):
    loss = model.train_on_batch(single_batch, single_labels)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
# If this doesn't work, model can't learn at all!

# 2. Check gradients
for name, param in model.named_parameters():
    print(f"{name}: grad norm = {param.grad.norm()}")

# 3. Learning rate finder
from torch_lr_finder import LRFinder
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader)
lr_finder.plot()  # Find steep decline region
```

---

### Q6: How do you choose between different ML models for a problem?

**Answer:**

**Decision Framework:**

1. **Problem Type:**
   - Classification, Regression, Clustering, Ranking?
   - Binary or multi-class?
   - Structured or unstructured data?

2. **Data Characteristics:**
   - Dataset size (small <1K, medium 1K-100K, large >100K)
   - Feature types (numeric, categorical, text, images)
   - Linear or non-linear relationships?

3. **Business Constraints:**
   - Latency requirements (<10ms, <100ms, <1s, batch)
   - Interpretability needed? (YES → Linear/Trees, NO → Deep learning OK)
   - Training time budget
   - Deployment resources (edge device vs cloud)

**Selection Strategy:**

**Small Data (<10K samples):**
- Start: Logistic Regression / Linear Regression
- Next: Random Forest (handles non-linearity, features)
- Avoid: Deep learning (needs lots of data)

**Medium Data (10K-1M):**
- Start: XGBoost / LightGBM (best overall performance)
- If interpretability needed: Random Forest + SHAP
- If text/images: Consider deep learning

**Large Data (>1M):**
- Structured: XGBoost, Neural Networks
- Text: Transformers (BERT, RoBERTa)
- Images: CNNs (ResNet, EfficientNet)
- Both: Multimodal models

**My Typical Progression:**
1. Baseline (majority class, mean)
2. Linear model (Logistic/Linear Regression)
3. Tree-based (Random Forest, then XGBoost)
4. Neural network (only if 1-3 underperform)

**Code Example:**
```python
from sklearn.metrics import accuracy_score

# 1. Baseline
baseline_pred = [majority_class] * len(y_test)
baseline_acc = accuracy_score(y_test, baseline_pred)

# 2. Linear
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

# 3. Tree
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# 4. Boosting
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

# Compare and pick best
print(f"Baseline: {baseline_acc:.3f}")
print(f"Linear: {lr_acc:.3f} (+{lr_acc - baseline_acc:.3f})")
print(f"Random Forest: {rf_acc:.3f} (+{rf_acc - baseline_acc:.3f})")
print(f"XGBoost: {xgb_acc:.3f} (+{xgb_acc - baseline_acc:.3f})")
```

---

## Conclusion

Production ML engineering is about:
- **Starting simple** and adding complexity only when needed
- **Measuring everything** to enable data-driven decisions
- **Automating** repetitive tasks to reduce errors
- **Planning for failure** because systems will fail
- **Documenting thoroughly** for maintainability

**Key Takeaways:**
1. Simple models often outperform complex ones in production
2. Data quality matters more than model complexity
3. Monitoring and maintenance are ongoing, not one-time tasks
4. Reproducibility requires discipline and tooling
5. Security and compliance cannot be afterthoughts

**Next Steps:**
- Apply these practices to your current projects
- Build an end-to-end ML project using this guide
- Contribute best practices from your experience

---

## Additional Resources

**Books:**
- "Designing Machine Learning Systems" - Chip Huyen
- "Machine Learning Design Patterns" - Lakshmanan et al.
- "Reliable Machine Learning" - Hulten et al.

**Courses:**
- Made With ML (Production ML) - https://madewithml.com
- Full Stack Deep Learning - https://fullstackdeeplearning.com

**Blogs:**
- Google Cloud ML Blog
- Netflix Tech Blog
- Uber Engineering Blog
- Airbnb Engineering Blog

**Tools:**
- Experiment Tracking: MLflow, Weights & Biases
- Data Validation: Great Expectations
- Model Serving: FastAPI, TensorFlow Serving, TorchServe
- Monitoring: Prometheus, Grafana, ELK Stack

---

*Last Updated: October 26, 2025*
*Version: 1.0*
*Maintained by: ML Engineering Team*
