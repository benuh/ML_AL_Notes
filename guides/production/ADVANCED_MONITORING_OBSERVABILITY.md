# Advanced Monitoring & Observability for Production ML

**Build robust monitoring systems to detect issues before they impact users**

> 📊 **Metrics & Alerting** | 🔍 **Data Drift Detection** | 📈 **Performance Monitoring** | 🚨 **Incident Response** | 📉 **Model Degradation**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Monitoring Stack Architecture](#monitoring-stack-architecture)
3. [Model Performance Monitoring](#model-performance-monitoring)
4. [Data Drift Detection](#data-drift-detection)
5. [System Metrics & Infrastructure](#system-metrics-infrastructure)
6. [Alerting & Incident Response](#alerting-incident-response)
7. [Observability Best Practices](#observability-best-practices)
8. [Interview Questions](#interview-questions)

---

## 1. Introduction

### What is ML Observability?

ML observability is the ability to understand the internal state of your ML system by examining its outputs (logs, metrics, traces).

**Why It Matters:**
- Models **degrade over time** (data drift, concept drift)
- **Silent failures** are common (model runs but produces bad predictions)
- **Debugging** production issues is hard without visibility
- **Business impact** of poor predictions can be severe

### The Three Pillars of Observability

**1. Metrics** - Quantitative measurements over time
- Model accuracy, latency, throughput
- Data distribution statistics
- Infrastructure utilization

**2. Logs** - Discrete event records
- Prediction logs with inputs/outputs
- Error logs and exceptions
- Audit trails

**3. Traces** - Request flow through system
- End-to-end latency breakdown
- Bottleneck identification
- Dependency mapping

---

## 2. Monitoring Stack Architecture

### Production Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring Stack                          │
└─────────────────────────────────────────────────────────────┘

ML Application
    ├── Prediction API (FastAPI)
    │   ├── Logs → Fluentd → Elasticsearch
    │   ├── Metrics → Prometheus
    │   └── Traces → Jaeger
    │
    ├── Model Server
    │   ├── Performance Metrics → Prometheus
    │   ├── Prediction Logs → S3/BigQuery
    │   └── Drift Detection → Custom Service
    │
    └── Data Pipeline
        ├── Data Quality Metrics → Prometheus
        └── Pipeline Status → Airflow Monitoring

Visualization & Alerting
    ├── Grafana (Dashboards)
    ├── AlertManager (Alerts)
    └── PagerDuty/Slack (Notifications)
```

### Technology Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Metrics** | Prometheus | Time-series metrics storage |
| **Logs** | ELK Stack (Elasticsearch, Logstash, Kibana) | Log aggregation and search |
| **Traces** | Jaeger/Zipkin | Distributed tracing |
| **Dashboards** | Grafana | Visualization |
| **Alerting** | AlertManager | Alert routing and management |
| **ML Monitoring** | Evidently AI, WhyLabs | ML-specific monitoring |

---

## 3. Model Performance Monitoring

### Key Metrics to Track

#### 1. Model Quality Metrics

```python
# model_monitor.py
import prometheus_client as prom
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Prometheus metrics
accuracy_gauge = prom.Gauge('model_accuracy', 'Model accuracy score')
precision_gauge = prom.Gauge('model_precision', 'Model precision score')
recall_gauge = prom.Gauge('model_recall', 'Model recall score')
f1_gauge = prom.Gauge('model_f1_score', 'Model F1 score')

prediction_counter = prom.Counter('predictions_total', 'Total predictions made', ['model_version'])
error_counter = prom.Counter('prediction_errors_total', 'Total prediction errors')

class ModelMonitor:
    """Monitor model performance in production"""

    def __init__(self, model, model_version='v1'):
        self.model = model
        self.model_version = model_version
        self.prediction_buffer = []
        self.label_buffer = []
        self.buffer_size = 1000  # Evaluate every 1000 predictions

    def log_prediction(self, features, prediction, ground_truth=None):
        """Log a single prediction for monitoring"""

        # Increment prediction counter
        prediction_counter.labels(model_version=self.model_version).inc()

        # Store for batch evaluation
        self.prediction_buffer.append(prediction)

        if ground_truth is not None:
            self.label_buffer.append(ground_truth)

            # Evaluate when buffer is full
            if len(self.label_buffer) >= self.buffer_size:
                self._evaluate_batch()
                self._clear_buffer()

    def _evaluate_batch(self):
        """Evaluate accumulated predictions"""
        if len(self.label_buffer) == 0:
            return

        predictions = np.array(self.prediction_buffer[-len(self.label_buffer):])
        labels = np.array(self.label_buffer)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

        # Update Prometheus metrics
        accuracy_gauge.set(accuracy)
        precision_gauge.set(precision)
        recall_gauge.set(recall)
        f1_gauge.set(f1)

        print(f"Batch Metrics (n={len(labels)}): "
              f"Accuracy={accuracy:.3f}, Precision={precision:.3f}, "
              f"Recall={recall:.3f}, F1={f1:.3f}")

    def _clear_buffer(self):
        """Clear buffers after evaluation"""
        self.prediction_buffer = []
        self.label_buffer = []

# Usage in FastAPI
from fastapi import FastAPI
import uvicorn

app = FastAPI()
monitor = ModelMonitor(model, model_version='v1')

# Expose metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    return prom.generate_latest()

@app.post("/predict")
async def predict(features: list[float]):
    # Make prediction
    prediction = model.predict([features])[0]

    # Log for monitoring
    monitor.log_prediction(features, prediction)

    return {"prediction": int(prediction)}

@app.post("/feedback")
async def feedback(features: list[float], prediction: int, ground_truth: int):
    """Receive ground truth labels for monitoring"""
    monitor.log_prediction(features, prediction, ground_truth)
    return {"status": "logged"}
```

#### 2. Prediction Distribution Monitoring

```python
import numpy as np
from scipy import stats
import prometheus_client as prom

# Prometheus histogram for prediction distribution
prediction_histogram = prom.Histogram(
    'prediction_distribution',
    'Distribution of model predictions',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

class PredictionDistributionMonitor:
    """Monitor changes in prediction distribution"""

    def __init__(self, baseline_predictions, alert_threshold=0.05):
        self.baseline_predictions = baseline_predictions
        self.alert_threshold = alert_threshold
        self.current_predictions = []

    def log_prediction(self, prediction_proba):
        """Log prediction probability"""
        self.current_predictions.append(prediction_proba)
        prediction_histogram.observe(prediction_proba)

        # Check distribution drift every 1000 predictions
        if len(self.current_predictions) >= 1000:
            self._check_distribution_drift()
            self.current_predictions = []

    def _check_distribution_drift(self):
        """Detect if prediction distribution has drifted"""
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(
            self.baseline_predictions,
            self.current_predictions
        )

        if p_value < self.alert_threshold:
            print(f"⚠️  ALERT: Prediction distribution drift detected!")
            print(f"KS statistic: {statistic:.4f}, p-value: {p_value:.4f}")
            # Trigger alert (send to Slack/PagerDuty)
            self._send_alert(
                severity='warning',
                message=f"Prediction distribution drift: p-value={p_value:.4f}"
            )
        else:
            print(f"✓ Prediction distribution stable (p-value: {p_value:.4f})")

    def _send_alert(self, severity, message):
        """Send alert to monitoring system"""
        # Integration with AlertManager, Slack, PagerDuty, etc.
        pass
```

#### 3. Latency & Throughput Monitoring

```python
import time
import prometheus_client as prom

# Prometheus metrics
latency_histogram = prom.Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
)

throughput_counter = prom.Counter(
    'predictions_per_second',
    'Number of predictions per second'
)

class LatencyMonitor:
    """Monitor prediction latency"""

    def __init__(self, p95_threshold_ms=100):
        self.p95_threshold_ms = p95_threshold_ms
        self.latencies = []

    def measure(self, func):
        """Decorator to measure function latency"""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            latency = time.time() - start

            # Record latency
            latency_histogram.observe(latency)
            throughput_counter.inc()
            self.latencies.append(latency * 1000)  # ms

            # Check SLA every 1000 requests
            if len(self.latencies) >= 1000:
                self._check_sla()
                self.latencies = []

            return result
        return wrapper

    def _check_sla(self):
        """Check if latency SLA is met"""
        p95 = np.percentile(self.latencies, 95)
        p99 = np.percentile(self.latencies, 99)
        median = np.median(self.latencies)

        print(f"Latency - Median: {median:.1f}ms, P95: {p95:.1f}ms, P99: {p99:.1f}ms")

        if p95 > self.p95_threshold_ms:
            print(f"⚠️  ALERT: P95 latency ({p95:.1f}ms) exceeds threshold ({self.p95_threshold_ms}ms)")
            # Trigger alert
        else:
            print(f"✓ Latency SLA met")

# Usage
latency_monitor = LatencyMonitor(p95_threshold_ms=100)

@latency_monitor.measure
def predict_with_monitoring(features):
    return model.predict([features])[0]
```

---

## 4. Data Drift Detection

### Input Feature Distribution Monitoring

```python
import pandas as pd
import numpy as np
from scipy import stats
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns

class DataDriftMonitor:
    """Comprehensive data drift detection"""

    def __init__(self, reference_data, feature_names):
        """
        Args:
            reference_data: Training data or recent production data (DataFrame)
            feature_names: List of feature column names
        """
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.current_data_buffer = []
        self.buffer_size = 1000

    def log_features(self, features):
        """Log incoming features"""
        self.current_data_buffer.append(features)

        if len(self.current_data_buffer) >= self.buffer_size:
            self._check_drift()
            self.current_data_buffer = []

    def _check_drift(self):
        """Check for data drift using multiple methods"""
        current_df = pd.DataFrame(self.current_data_buffer, columns=self.feature_names)

        # Method 1: Statistical tests (K-S test)
        self._ks_test_drift(current_df)

        # Method 2: Population Stability Index (PSI)
        psi_scores = self._calculate_psi(current_df)

        # Method 3: Evidently AI drift detection
        self._evidently_drift_report(current_df)

    def _ks_test_drift(self, current_df, threshold=0.05):
        """Kolmogorov-Smirnov test for each feature"""
        drifted_features = []

        for feature in self.feature_names:
            statistic, p_value = stats.ks_2samp(
                self.reference_data[feature],
                current_df[feature]
            )

            if p_value < threshold:
                drifted_features.append(feature)
                print(f"⚠️  Drift detected in '{feature}': p-value={p_value:.4f}")

        if len(drifted_features) > 0:
            self._send_drift_alert(drifted_features, method='KS-test')
        else:
            print("✓ No drift detected (KS test)")

        return drifted_features

    def _calculate_psi(self, current_df, threshold=0.2):
        """Population Stability Index for each feature"""
        psi_scores = {}

        for feature in self.feature_names:
            # Create bins
            _, bins = np.histogram(self.reference_data[feature], bins=10)

            # Calculate distributions
            ref_dist, _ = np.histogram(self.reference_data[feature], bins=bins)
            curr_dist, _ = np.histogram(current_df[feature], bins=bins)

            # Normalize
            ref_dist = ref_dist / ref_dist.sum()
            curr_dist = curr_dist / curr_dist.sum()

            # Add small epsilon to avoid division by zero
            ref_dist = np.where(ref_dist == 0, 0.0001, ref_dist)
            curr_dist = np.where(curr_dist == 0, 0.0001, curr_dist)

            # Calculate PSI
            psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
            psi_scores[feature] = psi

            if psi > threshold:
                print(f"⚠️  High PSI for '{feature}': {psi:.4f} (threshold: {threshold})")

        return psi_scores

    def _evidently_drift_report(self, current_df):
        """Generate Evidently AI drift report"""
        # Create report
        data_drift_report = Report(metrics=[DataDriftPreset()])

        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_df
        )

        # Get results
        report_dict = data_drift_report.as_dict()
        dataset_drift = report_dict['metrics'][0]['result']['dataset_drift']

        if dataset_drift:
            print("⚠️  Dataset drift detected by Evidently AI")
            # Save HTML report
            data_drift_report.save_html("drift_report.html")
        else:
            print("✓ No dataset drift detected")

    def _send_drift_alert(self, drifted_features, method):
        """Send alert about drifted features"""
        message = f"Data drift detected ({method}): {', '.join(drifted_features)}"
        print(f"🚨 ALERT: {message}")
        # Send to Slack/PagerDuty
```

### Using Evidently AI for Comprehensive Monitoring

```python
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset, DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
    TestColumnsType,
    TestNumberOfMissingValues
)

class EvidentlyMonitor:
    """Production monitoring with Evidently AI"""

    def __init__(self, reference_data):
        self.reference_data = reference_data

    def generate_full_report(self, current_data):
        """Generate comprehensive monitoring report"""

        # Data Quality Report
        quality_report = Report(metrics=[DataQualityPreset()])
        quality_report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        # Data Drift Report
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        # Save reports
        quality_report.save_html("data_quality_report.html")
        drift_report.save_html("data_drift_report.html")

        return quality_report, drift_report

    def run_test_suite(self, current_data):
        """Run automated tests on production data"""

        tests = TestSuite(tests=[
            TestNumberOfDriftedColumns(gte=5),  # Alert if >=5 columns drifted
            TestShareOfDriftedColumns(lt=0.3),   # Alert if >30% columns drifted
            TestColumnsType(),                   # Check data types
            TestNumberOfMissingValues(),         # Check for missing values
        ])

        tests.run(reference_data=self.reference_data, current_data=current_data)

        # Get results
        test_results = tests.as_dict()

        # Check if tests passed
        if not all(test['status'] == 'SUCCESS' for test in test_results['tests']):
            print("⚠️  Some tests failed!")
            tests.save_html("test_results.html")
            return False
        else:
            print("✓ All tests passed")
            return True
```

---

## 5. System Metrics & Infrastructure

### Infrastructure Monitoring with Prometheus

```python
# infrastructure_monitor.py
import prometheus_client as prom
import psutil
import torch

# System metrics
cpu_usage = prom.Gauge('cpu_usage_percent', 'CPU usage percentage')
memory_usage = prom.Gauge('memory_usage_percent', 'Memory usage percentage')
disk_usage = prom.Gauge('disk_usage_percent', 'Disk usage percentage')

# GPU metrics (if available)
gpu_memory = prom.Gauge('gpu_memory_mb', 'GPU memory usage in MB', ['gpu_id'])
gpu_utilization = prom.Gauge('gpu_utilization_percent', 'GPU utilization %', ['gpu_id'])

# Model-specific metrics
model_memory_size = prom.Gauge('model_memory_mb', 'Model memory size in MB')
active_requests = prom.Gauge('active_requests', 'Number of active requests')

class InfrastructureMonitor:
    """Monitor system and infrastructure metrics"""

    def __init__(self, update_interval=10):
        self.update_interval = update_interval

    def update_metrics(self):
        """Update all infrastructure metrics"""

        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_usage.set(cpu_percent)

        # Memory
        memory = psutil.virtual_memory()
        memory_usage.set(memory.percent)

        # Disk
        disk = psutil.disk_usage('/')
        disk_usage.set(disk.percent)

        # GPU (if available)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**2)  # MB
                mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**2)
                utilization = (mem_allocated / mem_total) * 100

                gpu_memory.labels(gpu_id=str(i)).set(mem_allocated)
                gpu_utilization.labels(gpu_id=str(i)).set(utilization)

    def start_monitoring(self):
        """Start continuous monitoring"""
        import threading

        def monitor_loop():
            while True:
                self.update_metrics()
                time.sleep(self.update_interval)

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()

# Start monitoring on application startup
infra_monitor = InfrastructureMonitor(update_interval=10)
infra_monitor.start_monitoring()
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ml_api'
    static_configs:
      - targets: ['localhost:8000']  # Your FastAPI app
    metrics_path: '/metrics'

  - job_name: 'node_exporter'
    static_configs:
      - targets: ['localhost:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']

rule_files:
  - 'alert_rules.yml'
```

### Alert Rules

```yaml
# alert_rules.yml
groups:
  - name: ml_model_alerts
    interval: 1m
    rules:
      # Model accuracy alerts
      - alert: ModelAccuracyDegraded
        expr: model_accuracy < 0.85
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy below threshold"
          description: "Model accuracy is {{ $value }}, below 0.85 threshold"

      # Latency alerts
      - alert: HighP95Latency
        expr: histogram_quantile(0.95, prediction_latency_seconds) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High P95 latency"
          description: "P95 latency is {{ $value }}s, above 100ms threshold"

      # Throughput alerts
      - alert: LowThroughput
        expr: rate(predictions_total[5m]) < 10
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low prediction throughput"
          description: "Throughput is {{ $value }} predictions/sec"

      # Infrastructure alerts
      - alert: HighCPUUsage
        expr: cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value }}%"

      - alert: HighMemoryUsage
        expr: memory_usage_percent > 85
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"
```

---

## 6. Alerting & Incident Response

### Slack Integration

```python
import requests
import json

class SlackAlerter:
    """Send alerts to Slack"""

    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def send_alert(self, severity, title, message, metrics=None):
        """Send formatted alert to Slack"""

        # Color coding
        colors = {
            'critical': '#FF0000',
            'warning': '#FFA500',
            'info': '#00FF00'
        }

        # Build message
        payload = {
            "attachments": [{
                "color": colors.get(severity, '#808080'),
                "title": f"🚨 {severity.upper()}: {title}",
                "text": message,
                "fields": [],
                "footer": "ML Monitoring System",
                "ts": int(time.time())
            }]
        }

        # Add metrics if provided
        if metrics:
            for key, value in metrics.items():
                payload["attachments"][0]["fields"].append({
                    "title": key,
                    "value": str(value),
                    "short": True
                })

        # Send to Slack
        response = requests.post(
            self.webhook_url,
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'}
        )

        return response.status_code == 200

# Usage
slack = SlackAlerter(webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL")

# Send alert
slack.send_alert(
    severity='critical',
    title='Model Accuracy Degraded',
    message='Model accuracy dropped below 85%',
    metrics={
        'Current Accuracy': '0.82',
        'Threshold': '0.85',
        'Model Version': 'v1.2.3'
    }
)
```

### Incident Response Playbook

```python
class IncidentManager:
    """Manage ML incidents"""

    def __init__(self, alerter):
        self.alerter = alerter
        self.incidents = {}

    def handle_drift_alert(self, drifted_features):
        """Respond to data drift"""

        # Step 1: Log incident
        incident_id = self._create_incident('data_drift', drifted_features)

        # Step 2: Send alerts
        self.alerter.send_alert(
            severity='warning',
            title='Data Drift Detected',
            message=f'Features showing drift: {", ".join(drifted_features)}',
            metrics={'Drifted Features': len(drifted_features)}
        )

        # Step 3: Automated response
        self._automated_response_drift(drifted_features)

        return incident_id

    def handle_accuracy_degradation(self, current_accuracy, threshold):
        """Respond to accuracy drop"""

        # Step 1: Log incident
        incident_id = self._create_incident('accuracy_degradation', {
            'current': current_accuracy,
            'threshold': threshold
        })

        # Step 2: Critical alert
        self.alerter.send_alert(
            severity='critical',
            title='Model Accuracy Degraded',
            message='Immediate action required',
            metrics={
                'Current Accuracy': f'{current_accuracy:.3f}',
                'Threshold': f'{threshold:.3f}',
                'Degradation': f'{(threshold - current_accuracy)*100:.1f}%'
            }
        )

        # Step 3: Automated response
        self._automated_response_accuracy()

        return incident_id

    def _automated_response_drift(self, drifted_features):
        """Automated actions for drift"""
        print("🔧 Automated Response:")
        print("1. Triggering model retraining with recent data")
        print("2. Increasing prediction logging for analysis")
        print("3. Rolling back to previous model version (if configured)")
        # Trigger retraining pipeline, etc.

    def _automated_response_accuracy(self):
        """Automated actions for accuracy degradation"""
        print("🔧 Automated Response:")
        print("1. Switching to backup model")
        print("2. Triggering emergency retraining")
        print("3. Escalating to on-call engineer")
        # Implement failover logic

    def _create_incident(self, incident_type, details):
        """Create incident record"""
        incident_id = f"{incident_type}_{int(time.time())}"
        self.incidents[incident_id] = {
            'type': incident_type,
            'details': details,
            'timestamp': time.time(),
            'status': 'open'
        }
        return incident_id
```

---

## 7. Observability Best Practices

### 1. Comprehensive Logging

```python
import logging
import json
from datetime import datetime

# Structured logging for ML predictions
class MLLogger:
    """Structured logging for ML systems"""

    def __init__(self, log_file='predictions.jsonl'):
        self.log_file = log_file
        self.logger = logging.getLogger('ml_logger')
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log_prediction(self, request_id, features, prediction, latency_ms,
                      model_version, ground_truth=None):
        """Log prediction with full context"""

        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request_id,
            'model_version': model_version,
            'features': features,
            'prediction': prediction,
            'latency_ms': latency_ms,
            'ground_truth': ground_truth
        }

        self.logger.info(json.dumps(log_entry))

    def log_error(self, request_id, error_type, error_message):
        """Log errors for debugging"""

        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request_id,
            'error_type': error_type,
            'error_message': error_message,
            'level': 'ERROR'
        }

        self.logger.error(json.dumps(log_entry))

# Usage
ml_logger = MLLogger('predictions.jsonl')

@app.post("/predict")
async def predict(features: list[float], request_id: str):
    try:
        start = time.time()
        prediction = model.predict([features])[0]
        latency = (time.time() - start) * 1000

        ml_logger.log_prediction(
            request_id=request_id,
            features=features,
            prediction=int(prediction),
            latency_ms=latency,
            model_version='v1.0.0'
        )

        return {"prediction": int(prediction)}

    except Exception as e:
        ml_logger.log_error(request_id, type(e).__name__, str(e))
        raise
```

### 2. Prediction Sampling Strategy

```python
import random

class PredictionSampler:
    """Sample predictions for detailed analysis"""

    def __init__(self, sample_rate=0.01, always_log_errors=True):
        """
        Args:
            sample_rate: Fraction of predictions to log (0.01 = 1%)
            always_log_errors: Always log incorrect predictions
        """
        self.sample_rate = sample_rate
        self.always_log_errors = always_log_errors

    def should_log(self, prediction, ground_truth=None):
        """Determine if prediction should be logged"""

        # Always log errors (if ground truth available)
        if self.always_log_errors and ground_truth is not None:
            if prediction != ground_truth:
                return True

        # Sample randomly
        return random.random() < self.sample_rate

    def log_if_sampled(self, logger, *args, **kwargs):
        """Log only if sampled"""
        if self.should_log(kwargs.get('prediction'), kwargs.get('ground_truth')):
            logger.log_prediction(*args, **kwargs)
```

### 3. Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "ML Model Performance",
    "panels": [
      {
        "title": "Model Accuracy",
        "targets": [
          {
            "expr": "model_accuracy",
            "legendFormat": "Accuracy"
          }
        ],
        "type": "graph",
        "thresholds": [
          {"value": 0.85, "color": "red"}
        ]
      },
      {
        "title": "Prediction Latency (P95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, prediction_latency_seconds)",
            "legendFormat": "P95"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Throughput (predictions/sec)",
        "targets": [
          {
            "expr": "rate(predictions_total[5m])",
            "legendFormat": "QPS"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

---

## 8. Interview Questions

### Q1: How would you detect and respond to data drift in a production ML system?

**Answer:**

**Detection Methods:**

1. **Statistical Tests:**
   - K-S test for continuous features
   - Chi-square test for categorical features
   - Population Stability Index (PSI)

2. **Distribution Monitoring:**
   - Compare current vs reference distributions
   - Alert if p-value < threshold (e.g., 0.05)

3. **Tools:**
   - Evidently AI for automated drift detection
   - Custom metrics in Prometheus

**Response Strategy:**

```
Drift Detected
    ├── Severity Assessment
    │   ├── Minor drift (1-2 features)
    │   │   └── Monitor closely, log for analysis
    │   └── Major drift (>30% features)
    │       └── Immediate action required
    │
    ├── Automated Response
    │   ├── Increase logging/sampling
    │   ├── Trigger retraining pipeline
    │   └── Send alerts to on-call
    │
    └── Manual Response
        ├── Investigate root cause
        ├── Update feature engineering
        └── Consider model rollback if severe
```

**Example:**
```python
# Weekly drift check
if psi_score > 0.2:  # Significant drift
    trigger_retraining()
    send_alert("Data drift detected")
elif psi_score > 0.1:  # Minor drift
    increase_logging_rate()
```

**Interview Follow-up:**
- "What causes data drift?" → Distribution changes, seasonality, external events
- "When to retrain?" → Drift + performance degradation
- "False positive prevention?" → Multiple detection methods, confidence intervals

---

### Q2: Design a monitoring system for a recommendation model serving 10K QPS. What metrics matter most?

**Answer:**

**Tiered Monitoring Approach:**

**Tier 1: Critical Metrics (Real-time alerts)**
1. **Availability:** Uptime > 99.9%
2. **Latency:** P95 < 100ms, P99 < 200ms
3. **Error Rate:** < 0.1%

**Tier 2: Model Performance (Hourly checks)**
1. **CTR (Click-Through Rate):** Track vs baseline
2. **Diversity:** Avoid filter bubbles
3. **Coverage:** % items recommended

**Tier 3: Business Metrics (Daily tracking)**
1. **Revenue per user**
2. **Engagement time**
3. **Conversion rate**

**Architecture:**

```
10K QPS System
    ├── Sampling Strategy
    │   ├── Sample 1% predictions (100/sec)
    │   ├── Always log errors
    │   └── Always log low-confidence predictions
    │
    ├── Metrics Collection
    │   ├── Prometheus (latency, throughput)
    │   ├── BigQuery (prediction logs)
    │   └── Redis (real-time CTR)
    │
    └── Dashboards
        ├── Grafana (real-time ops)
        └── Looker (business metrics)
```

**Trade-offs:**
- **Full logging at 10K QPS** = 10K * 86400 = 864M records/day (expensive!)
- **1% sampling** = 8.64M records/day (manageable)
- **Aggregate metrics** = Real-time visibility without storage burden

**Interview Follow-up:**
- "How to reduce latency?" → Caching, batching, model optimization
- "Handling traffic spikes?" → Auto-scaling, circuit breakers
- "When to page on-call?" → Only critical alerts (accuracy <80%, p95 >500ms)

---

### Q3: Your model accuracy dropped from 92% to 85% in production. Walk through your debugging process.

**Answer:**

**Systematic Debugging Process:**

**Step 1: Verify the Issue (5 minutes)**
```python
# Check if issue is real
- Look at evaluation window (last hour? day? week?)
- Check sample size (enough data points?)
- Compare to historical baseline
```

**Step 2: Check Data Quality (15 minutes)**
```python
# Most common cause: bad input data
- Missing values increased?
- Feature distributions shifted?
- Data schema changed?
- Upstream pipeline failures?
```

**Step 3: Check Infrastructure (10 minutes)**
```python
# Rule out serving issues
- Model version correct?
- Feature serving working?
- Latency/timeout issues?
- Hardware problems (GPU, memory)?
```

**Step 4: Analyze Errors (20 minutes)**
```python
# Which predictions are failing?
- Specific user segments?
- New vs existing users?
- Certain products/categories?
- Time-of-day patterns?
```

**Step 5: Check for Drift (15 minutes)**
```python
# Data distribution changes
- Run KS test on all features
- Check PSI scores
- Compare feature importance
```

**Step 6: Hypothesis & Action (10 minutes)**
```
Root Cause Found → Action
├── Data drift → Retrain with recent data
├── Bad data → Fix pipeline, rollback to last good model
├── Concept drift → Update labels, retrain
├── Infrastructure → Fix hardware/config issue
└── Unknown → Rollback model, investigate offline
```

**Real Example:**
```
Scenario: E-commerce recommendations
Problem: Accuracy 92% → 85%

Investigation:
1. ✓ Real issue (10K predictions over 24 hours)
2. ✓ Data looks ok (no nulls, distributions similar)
3. ✓ Infrastructure fine (v1.2 model, no errors)
4. ❌ Analysis: New users (cold start) failing
5. Hypothesis: Recent influx of new users (marketing campaign)

Solution:
- Add content-based fallback for cold start
- Improved new user onboarding flow
- Accuracy recovered to 90% within 2 days
```

**Interview Follow-up:**
- "How long to investigate?" → 1 hour for initial triage, deeper analysis if needed
- "When to rollback?" → Immediately if critical business impact
- "How to prevent?" → Automated testing, gradual rollouts, monitoring

---

## Summary

**Advanced Monitoring & Observability enables:**
- ✅ Early detection of model degradation
- ✅ Root cause analysis for production issues
- ✅ Data quality and drift monitoring
- ✅ Performance optimization insights
- ✅ Incident response automation

**Key Technologies:**
- **Metrics:** Prometheus, Grafana
- **Drift Detection:** Evidently AI, custom statistical tests
- **Logging:** Structured JSON logs, ELK stack
- **Alerting:** AlertManager, Slack, PagerDuty

**Best Practices:**
- Monitor at multiple levels (model, data, infrastructure)
- Use sampling to reduce costs at scale
- Automate incident response where possible
- Balance alerting (avoid fatigue)
- Track business metrics, not just technical

**Next Steps:**
1. Set up Prometheus + Grafana
2. Implement drift detection with Evidently
3. Create alert rules for critical metrics
4. Build incident response playbook
5. Practice debugging production issues

---

**Generated with Claude Code**

*Last Updated: 2025-10-25*
