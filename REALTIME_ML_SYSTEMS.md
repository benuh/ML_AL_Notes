# Real-Time ML Systems: Complete Guide

**Build low-latency, high-throughput ML systems for real-time predictions**

> ⚡ **Real-Time Serving** | 📨 **Event-Driven ML** | 🔄 **Online Learning** | 🚀 **Low-Latency Optimization** | 📊 **Streaming Inference**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Real-Time ML Architecture](#real-time-ml-architecture)
3. [Low-Latency Model Serving](#low-latency-model-serving)
4. [Event-Driven ML with Kafka](#event-driven-ml-with-kafka)
5. [Online Learning](#online-learning)
6. [Optimization Strategies](#optimization-strategies)
7. [Production Patterns](#production-patterns)
8. [Interview Questions](#interview-questions)

---

## 1. Introduction

### What is Real-Time ML?

Real-time ML systems make predictions with **low latency** (typically <100ms) and handle **high throughput** (thousands of requests per second).

**Use Cases:**
- 🔍 **Search ranking** - Personalize results in <50ms
- 💳 **Fraud detection** - Block transactions in real-time
- 📱 **Recommendation systems** - Serve personalized content instantly
- 🚗 **Autonomous vehicles** - React to environment in milliseconds
- 💬 **Chatbots** - Respond to user messages immediately
- 📊 **Dynamic pricing** - Adjust prices based on demand

### Real-Time vs Batch ML

| Aspect | Batch ML | Real-Time ML |
|--------|----------|--------------|
| **Latency** | Minutes to hours | <100ms |
| **Throughput** | Process all data | Handle high TPS |
| **Predictions** | Pre-computed | On-demand |
| **Data Freshness** | Hours old | Milliseconds old |
| **Complexity** | Lower | Higher |
| **Cost** | Lower | Higher (infrastructure) |
| **Use Case** | Email campaigns | Fraud detection |

### Performance Requirements

**Latency Targets:**
- **Sub-10ms:** Programmatic advertising bidding
- **10-50ms:** Search ranking, autocomplete
- **50-100ms:** Recommendations, fraud detection
- **100-500ms:** Personalization, content moderation
- **500ms-2s:** Complex inference (LLMs, image generation)

**Throughput Targets:**
- **Low:** <100 QPS (queries per second)
- **Medium:** 100-1K QPS
- **High:** 1K-10K QPS
- **Very High:** >10K QPS

---

## 2. Real-Time ML Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Real-Time ML System                         │
└──────────────────────────────────────────────────────────────┘

1. Request Path (Synchronous)
   ┌─────────┐    ┌──────────┐    ┌─────────────┐    ┌────────┐
   │ Client  │───▶│   API    │───▶│  Feature    │───▶│ Model  │
   │         │    │ Gateway  │    │  Serving    │    │ Server │
   └─────────┘    └──────────┘    └─────────────┘    └────────┘
        ▲                                                   │
        │                                                   │
        └───────────────── Prediction ─────────────────────┘

2. Event Stream (Asynchronous)
   ┌──────────┐    ┌─────────┐    ┌──────────────┐    ┌──────────┐
   │  Events  │───▶│  Kafka  │───▶│    Flink     │───▶│  Feature │
   │ Producer │    │ Stream  │    │  Processor   │    │  Store   │
   └──────────┘    └─────────┘    └──────────────┘    └──────────┘

3. Feedback Loop (Continuous Learning)
   ┌──────────┐    ┌─────────┐    ┌──────────────┐    ┌──────────┐
   │Prediction│───▶│ Labels  │───▶│   Online     │───▶│  Update  │
   │   Logs   │    │ Stream  │    │   Training   │    │  Model   │
   └──────────┘    └─────────┘    └──────────────┘    └──────────┘
```

### Components

1. **API Gateway** - Load balancing, rate limiting, authentication
2. **Feature Serving** - Low-latency feature retrieval (Redis, DynamoDB)
3. **Model Server** - Optimized inference (TensorFlow Serving, TorchServe, ONNX Runtime)
4. **Caching Layer** - Reduce redundant computations
5. **Streaming Pipeline** - Real-time feature updates (Kafka, Flink)
6. **Monitoring** - Latency, throughput, errors

---

## 3. Low-Latency Model Serving

### Option 1: TensorFlow Serving

**Best for:** TensorFlow models, high throughput

#### Installation

```bash
# Docker
docker pull tensorflow/serving

# Start server
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  tensorflow/serving
```

#### Export Model

```python
import tensorflow as tf

# Train model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)

# Export for serving
export_path = "/models/my_model/1"
model.save(export_path, save_format='tf')
```

#### Client Request

```python
import requests
import numpy as np

# Prepare data
data = {
    "instances": X_test[:5].tolist()
}

# REST API
response = requests.post(
    'http://localhost:8501/v1/models/my_model:predict',
    json=data
)

predictions = response.json()['predictions']
print(predictions)
```

#### gRPC (Lower Latency)

```python
import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import tensorflow as tf

# Create channel
channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Create request
request = predict_pb2.PredictRequest()
request.model_spec.name = 'my_model'
request.model_spec.signature_name = 'serving_default'

# Add input
request.inputs['input'].CopyFrom(
    tf.make_tensor_proto(X_test[:5], shape=[5, 100])
)

# Send request
result = stub.Predict(request, 10.0)  # 10 second timeout

# Parse response
predictions = tf.make_ndarray(result.outputs['output'])
print(predictions)
```

### Option 2: TorchServe

**Best for:** PyTorch models

#### Installation

```bash
pip install torchserve torch-model-archiver
```

#### Export Model

```python
import torch
import torch.nn as nn

# Define model
class RecommendationModel(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Train model
model = RecommendationModel()
# ... training code ...

# Save model
torch.save(model.state_dict(), "model.pth")

# Create handler (handler.py)
# Custom preprocessing/postprocessing
```

```python
# handler.py
from ts.torch_handler.base_handler import BaseHandler
import torch
import json

class RecommendationHandler(BaseHandler):
    def preprocess(self, data):
        """Preprocess input data"""
        input_data = data[0].get("data") or data[0].get("body")
        if isinstance(input_data, (bytes, bytearray)):
            input_data = input_data.decode('utf-8')

        features = json.loads(input_data)
        tensor = torch.tensor(features['input'], dtype=torch.float32)
        return tensor

    def postprocess(self, inference_output):
        """Postprocess predictions"""
        predictions = inference_output.detach().cpu().numpy().tolist()
        return [{"predictions": predictions}]
```

#### Archive Model

```bash
# Create .mar file
torch-model-archiver \
  --model-name recommendation_model \
  --version 1.0 \
  --model-file model.py \
  --serialized-file model.pth \
  --handler handler.py \
  --export-path model-store
```

#### Start Server

```bash
torchserve --start \
  --model-store model-store \
  --models recommendation_model=recommendation_model.mar \
  --ncs \
  --ts-config config.properties
```

#### Client Request

```python
import requests

data = {
    "input": [0.1, 0.2, 0.3, ...]  # 100 features
}

response = requests.post(
    'http://localhost:8080/predictions/recommendation_model',
    json=data
)

predictions = response.json()['predictions']
```

### Option 3: ONNX Runtime (Cross-Framework)

**Best for:** Framework-agnostic, optimized inference

```python
import onnx
import onnxruntime as ort
import torch

# Convert PyTorch to ONNX
model = RecommendationModel()
dummy_input = torch.randn(1, 100)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Load and run with ONNX Runtime (faster!)
ort_session = ort.InferenceSession("model.onnx")

inputs = {'input': X_test[:5].astype(np.float32)}
outputs = ort_session.run(None, inputs)
predictions = outputs[0]
```

### Option 4: FastAPI + Python (Simplest)

**Best for:** Quick prototypes, custom logic

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
import logging

app = FastAPI(title="Recommendation API")

# Load model at startup
model = None

class PredictionRequest(BaseModel):
    user_id: int
    product_ids: list[int]
    features: list[float]

class PredictionResponse(BaseModel):
    user_id: int
    recommendations: list[dict]
    latency_ms: float

@app.on_event("startup")
def load_model():
    global model
    model = RecommendationModel()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    logging.info("Model loaded successfully")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    import time
    start = time.time()

    try:
        # Convert to tensor
        features_tensor = torch.tensor([request.features], dtype=torch.float32)

        # Inference
        with torch.no_grad():
            logits = model(features_tensor)
            scores = torch.softmax(logits, dim=1)

        # Top-k recommendations
        top_k = 10
        top_scores, top_indices = torch.topk(scores, top_k)

        recommendations = [
            {
                "product_id": request.product_ids[idx],
                "score": float(top_scores[0][i])
            }
            for i, idx in enumerate(top_indices[0])
        ]

        latency_ms = (time.time() - start) * 1000

        return PredictionResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            latency_ms=latency_ms
        )

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

---

## 4. Event-Driven ML with Kafka

### Real-Time Feature Computation

#### Kafka Producer (Collect Events)

```python
from kafka import KafkaProducer
from datetime import datetime
import json

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    compression_type='gzip',
    batch_size=16384,
    linger_ms=10
)

def send_user_event(user_id, event_type, metadata):
    """Send user event to Kafka"""
    event = {
        'user_id': user_id,
        'event_type': event_type,
        'metadata': metadata,
        'timestamp': datetime.utcnow().isoformat()
    }

    producer.send('user-events', value=event, key=str(user_id).encode('utf-8'))
    producer.flush()

# Example events
send_user_event(1001, 'view', {'product_id': 5001, 'duration': 30})
send_user_event(1001, 'click', {'product_id': 5001, 'position': 3})
send_user_event(1001, 'purchase', {'product_id': 5001, 'amount': 99.99})
```

#### Kafka Streams (Real-Time Aggregation)

```python
from kafka import KafkaConsumer
from collections import defaultdict
import redis
import json

# Redis for online features
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Kafka consumer
consumer = KafkaConsumer(
    'user-events',
    bootstrap_servers=['localhost:9092'],
    group_id='feature-processor',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest'
)

# In-memory state (should use Redis/RocksDB for production)
user_state = defaultdict(lambda: {
    'total_views': 0,
    'total_clicks': 0,
    'total_purchases': 0,
    'total_spent': 0.0
})

for message in consumer:
    event = message.value
    user_id = event['user_id']

    # Update state
    if event['event_type'] == 'view':
        user_state[user_id]['total_views'] += 1
    elif event['event_type'] == 'click':
        user_state[user_id]['total_clicks'] += 1
    elif event['event_type'] == 'purchase':
        user_state[user_id]['total_purchases'] += 1
        user_state[user_id]['total_spent'] += event['metadata']['amount']

    # Compute derived features
    features = {
        'total_views': user_state[user_id]['total_views'],
        'total_clicks': user_state[user_id]['total_clicks'],
        'total_purchases': user_state[user_id]['total_purchases'],
        'total_spent': user_state[user_id]['total_spent'],
        'ctr': user_state[user_id]['total_clicks'] / max(user_state[user_id]['total_views'], 1),
        'avg_purchase_value': user_state[user_id]['total_spent'] / max(user_state[user_id]['total_purchases'], 1),
        'last_updated': event['timestamp']
    }

    # Update Redis (online feature store)
    redis_client.hmset(f"user_features:{user_id}", features)
    redis_client.expire(f"user_features:{user_id}", 7 * 24 * 60 * 60)  # 7 days TTL

    print(f"Updated features for user {user_id}: {features}")
```

### Real-Time Model Inference Pipeline

```python
from kafka import KafkaConsumer, KafkaProducer
import json
import torch

# Consumer: prediction requests
consumer = KafkaConsumer(
    'prediction-requests',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# Producer: prediction results
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Load model
model = RecommendationModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

for message in consumer:
    request = message.value

    # Get features from Redis
    user_features = redis_client.hgetall(f"user_features:{request['user_id']}")

    # Convert to tensor
    features_list = [
        float(user_features[b'total_views']),
        float(user_features[b'total_clicks']),
        float(user_features[b'ctr']),
        # ... more features
    ]
    features_tensor = torch.tensor([features_list], dtype=torch.float32)

    # Inference
    with torch.no_grad():
        logits = model(features_tensor)
        predictions = torch.softmax(logits, dim=1).numpy().tolist()[0]

    # Send result to output topic
    result = {
        'request_id': request['request_id'],
        'user_id': request['user_id'],
        'predictions': predictions,
        'timestamp': datetime.utcnow().isoformat()
    }

    producer.send('prediction-results', value=result)
```

---

## 5. Online Learning

### Incremental Learning (Update Model with New Data)

```python
import torch
import torch.nn as nn
from kafka import KafkaConsumer
import json

class OnlineRecommendationModel:
    """Model that learns incrementally from streaming data"""

    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def update(self, features, labels):
        """Incremental update with single batch"""
        self.model.train()

        # Forward pass
        outputs = self.model(features)
        loss = self.criterion(outputs, labels)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, features):
        """Make prediction"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features)
            predictions = torch.softmax(outputs, dim=1)
        return predictions

# Initialize
model = RecommendationModel()
online_learner = OnlineRecommendationModel(model, learning_rate=0.0001)

# Consume feedback stream
consumer = KafkaConsumer(
    'user-feedback',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

update_buffer = []
buffer_size = 32

for message in consumer:
    feedback = message.value

    # Buffer for mini-batch update
    update_buffer.append({
        'features': feedback['features'],
        'label': feedback['label']
    })

    # Update model when buffer is full
    if len(update_buffer) >= buffer_size:
        features = torch.tensor([d['features'] for d in update_buffer], dtype=torch.float32)
        labels = torch.tensor([d['label'] for d in update_buffer], dtype=torch.long)

        loss = online_learner.update(features, labels)

        print(f"Model updated with batch of {buffer_size}, loss: {loss:.4f}")

        # Save checkpoint
        if message.offset % 1000 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_offset_{message.offset}.pth")

        update_buffer = []
```

### Bandits for Online Learning

```python
import numpy as np
from collections import defaultdict

class ContextualBandit:
    """Thompson Sampling for recommendations"""

    def __init__(self, n_actions, feature_dim):
        self.n_actions = n_actions
        self.feature_dim = feature_dim

        # Prior for each action: Normal-Inverse-Gamma
        self.mu = [np.zeros(feature_dim) for _ in range(n_actions)]
        self.cov = [np.eye(feature_dim) for _ in range(n_actions)]
        self.alpha = np.ones(n_actions)  # Success count
        self.beta = np.ones(n_actions)   # Failure count

    def select_action(self, context):
        """Thompson Sampling: sample from posterior"""
        expected_rewards = []

        for i in range(self.n_actions):
            # Sample reward estimate from posterior
            theta = np.random.multivariate_normal(self.mu[i], self.cov[i])
            expected_reward = np.dot(context, theta)
            expected_rewards.append(expected_reward)

        # Select action with highest sampled reward
        action = np.argmax(expected_rewards)
        return action

    def update(self, action, context, reward):
        """Update posterior with observed reward"""
        # Bayesian linear regression update
        cov_inv = np.linalg.inv(self.cov[action])
        self.cov[action] = np.linalg.inv(cov_inv + np.outer(context, context))
        self.mu[action] = self.cov[action] @ (cov_inv @ self.mu[action] + reward * context)

        # Update success/failure counts
        if reward > 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1

# Usage
bandit = ContextualBandit(n_actions=10, feature_dim=50)

# Real-time serving
def get_recommendation(user_features):
    # Context = user features
    context = np.array(user_features)

    # Select product
    product_id = bandit.select_action(context)

    return product_id

# Feedback loop
def process_feedback(user_features, product_id, clicked):
    context = np.array(user_features)
    reward = 1.0 if clicked else 0.0

    bandit.update(product_id, context, reward)
```

---

## 6. Optimization Strategies

### 1. Model Optimization

#### Quantization (Reduce Model Size)

```python
import torch

# Post-Training Quantization (PTQ)
model_fp32 = RecommendationModel()
model_fp32.load_state_dict(torch.load("model.pth"))
model_fp32.eval()

# Dynamic Quantization (weights only)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},  # Quantize linear layers
    dtype=torch.qint8
)

# Compare sizes
torch.save(model_fp32.state_dict(), "model_fp32.pth")
torch.save(model_int8.state_dict(), "model_int8.pth")

# Inference speedup
import time

x = torch.randn(100, 100)

start = time.time()
_ = model_fp32(x)
fp32_time = time.time() - start

start = time.time()
_ = model_int8(x)
int8_time = time.time() - start

print(f"FP32: {fp32_time*1000:.2f}ms, INT8: {int8_time*1000:.2f}ms")
print(f"Speedup: {fp32_time/int8_time:.2f}x")
```

#### Model Pruning (Remove Redundant Weights)

```python
import torch.nn.utils.prune as prune

# Prune 30% of weights
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# Make pruning permanent
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.remove(module, 'weight')

# Check sparsity
total_params = sum(p.numel() for p in model.parameters())
zero_params = sum((p == 0).sum().item() for p in model.parameters())
sparsity = zero_params / total_params
print(f"Sparsity: {sparsity:.2%}")
```

### 2. Batching Requests

```python
import asyncio
from fastapi import FastAPI
import torch

app = FastAPI()

class BatchPredictor:
    """Batch multiple requests for efficient inference"""

    def __init__(self, model, max_batch_size=32, max_wait_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []
        self.futures = []

    async def predict(self, features):
        """Add to queue and wait for batched prediction"""
        future = asyncio.Future()

        self.queue.append(features)
        self.futures.append(future)

        # Trigger batch processing if full
        if len(self.queue) >= self.max_batch_size:
            await self._process_batch()

        # Wait for result
        return await asyncio.wait_for(future, timeout=1.0)

    async def _process_batch(self):
        """Process accumulated batch"""
        if not self.queue:
            return

        # Collect batch
        batch_features = torch.tensor(self.queue, dtype=torch.float32)
        batch_futures = self.futures

        # Clear queue
        self.queue = []
        self.futures = []

        # Batch inference
        with torch.no_grad():
            predictions = self.model(batch_features)

        # Resolve futures
        for i, future in enumerate(batch_futures):
            future.set_result(predictions[i].numpy().tolist())

# Global predictor
predictor = BatchPredictor(model)

@app.post("/predict")
async def predict(features: list[float]):
    prediction = await predictor.predict(features)
    return {"prediction": prediction}

# Background task to process partial batches
async def batch_processor():
    while True:
        await asyncio.sleep(predictor.max_wait_ms / 1000)
        await predictor._process_batch()

@app.on_event("startup")
async def start_batch_processor():
    asyncio.create_task(batch_processor())
```

### 3. Caching Predictions

```python
from functools import lru_cache
import hashlib
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_prediction_cached(user_id, product_ids, ttl=300):
    """Cache predictions in Redis"""

    # Create cache key
    cache_key = f"prediction:{user_id}:{hashlib.md5(str(product_ids).encode()).hexdigest()}"

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Compute prediction
    features = get_features(user_id, product_ids)
    prediction = model(torch.tensor(features, dtype=torch.float32))

    # Cache result
    result = prediction.numpy().tolist()
    redis_client.setex(cache_key, ttl, json.dumps(result))

    return result
```

### 4. Multi-Model Serving (A/B Testing)

```python
from fastapi import FastAPI, Header
import random

app = FastAPI()

# Load multiple model versions
model_v1 = load_model("model_v1.pth")
model_v2 = load_model("model_v2.pth")

@app.post("/predict")
async def predict(features: list[float], x_experiment: str = Header(None)):
    """Serve different models based on experiment assignment"""

    # Deterministic assignment based on user_id
    if x_experiment:
        user_id = int(x_experiment)
        variant = 'v2' if user_id % 10 < 3 else 'v1'  # 30% v2, 70% v1
    else:
        variant = 'v1'

    # Select model
    model = model_v2 if variant == 'v2' else model_v1

    # Inference
    features_tensor = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(features_tensor)

    return {
        "prediction": prediction.numpy().tolist(),
        "model_version": variant
    }
```

---

## 7. Production Patterns

### Pattern 1: Circuit Breaker

```python
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = 1  # Normal operation
    OPEN = 2    # Failing, reject requests
    HALF_OPEN = 3  # Testing recovery

class CircuitBreaker:
    """Prevent cascading failures"""

    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker"""

        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Reset on success"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Increment failure count"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)

def get_prediction(features):
    try:
        return circuit_breaker.call(model.predict, features)
    except Exception:
        # Fallback to default prediction
        return get_default_prediction()
```

### Pattern 2: Request Hedging

```python
import asyncio

async def predict_with_hedging(features, timeout=50):
    """Send duplicate requests to reduce tail latency"""

    # Primary request
    primary = asyncio.create_task(model_server_1.predict(features))

    # Wait for primary
    try:
        result = await asyncio.wait_for(primary, timeout=timeout/1000)
        return result
    except asyncio.TimeoutError:
        # Hedge: send backup request
        backup = asyncio.create_task(model_server_2.predict(features))

        # Return whichever completes first
        done, pending = await asyncio.wait(
            [primary, backup],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel remaining
        for task in pending:
            task.cancel()

        return done.pop().result()
```

---

## 8. Interview Questions

### Q1: How would you design a real-time fraud detection system with <50ms latency?

**Answer:**

**Requirements:**
- Latency: <50ms p99
- Throughput: 10K TPS
- Accuracy: Minimize false positives

**Architecture:**

```
Transaction → API Gateway → Feature Serving (Redis) → Model Server → Block/Allow
                ↓
              Kafka → Fraud Analyst Queue (for review)
```

**Components:**

1. **Feature Serving (Redis):**
   - Pre-computed user features (e.g., avg_transaction_amount_30d)
   - Real-time features (e.g., transactions_last_hour)
   - Latency: <5ms

2. **Model Server (ONNX Runtime):**
   - Quantized model (INT8)
   - Batch size: 1 (real-time)
   - Latency: <20ms

3. **Optimization:**
   - **Caching:** Cache low-risk users (reduce predictions by 70%)
   - **Rule-based pre-filter:** Block obvious fraud immediately
   - **Model ensemble:** Simple model first (fast), complex model for edge cases

4. **Fallback:**
   - If latency >50ms → allow transaction, async review
   - If model unavailable → rule-based scoring

**Latency Breakdown:**
- Feature retrieval: 5ms
- Model inference: 20ms
- API overhead: 10ms
- Network: 10ms
- **Total: 45ms** ✅

**Interview Follow-up:**
- "How to handle model updates?" → Blue-green deployment, gradual rollout
- "What if Redis is down?" → Fallback to DB or default features
- "How to reduce false positives?" → Ensemble models, active learning from feedback

---

### Q2: Explain online learning vs batch retraining. When would you use each?

**Answer:**

| Aspect | Batch Retraining | Online Learning |
|--------|-----------------|-----------------|
| **Frequency** | Daily/weekly | Continuous |
| **Data** | Full dataset | Streaming |
| **Model Update** | Replace model | Incremental update |
| **Computational Cost** | High (full retrain) | Low (incremental) |
| **Data Drift** | Delayed response | Fast adaptation |
| **Stability** | More stable | Can be unstable |

**When to use Batch Retraining:**
- Stable data distribution
- Complex models (deep learning)
- Sufficient computational resources
- Need for rigorous validation

**When to use Online Learning:**
- Fast-changing distributions (ad CTR)
- Concept drift (user preferences)
- Limited computational resources
- Need real-time adaptation

**Hybrid Approach (Best Practice):**
1. **Base model:** Retrained weekly with full data
2. **Online updates:** Incremental adjustments daily
3. **Reset:** Periodically reset to prevent drift

```python
# Hybrid approach
def hybrid_learning_pipeline():
    # Weekly: Full retrain
    if is_sunday():
        base_model = full_retrain(all_data)
        save_model(base_model, "base_model.pth")

    # Daily: Online updates
    online_model = load_model("base_model.pth")
    for batch in get_daily_data():
        online_model.update(batch)

    save_model(online_model, "current_model.pth")
```

**Interview Follow-up:**
- "How to prevent catastrophic forgetting?" → Replay buffer, regularization
- "When does online learning fail?" → Non-stationary labels, adversarial data
- "How to validate online model?" → Holdout set, compare to base model

---

### Q3: How do you optimize model serving latency from 200ms to 50ms?

**Answer:**

**Optimization Strategies:**

1. **Model Optimization (50ms → 20ms):**
   ```python
   # Quantization: FP32 → INT8
   model_int8 = torch.quantization.quantize_dynamic(model_fp32)
   # Speedup: 2-4x

   # Pruning: Remove 30% weights
   prune.l1_unstructured(model, 'weight', amount=0.3)
   # Speedup: 1.5x

   # Knowledge Distillation: Large model → Small model
   student_model = train_distillation(teacher_model, student_model, data)
   # Speedup: 5-10x
   ```

2. **Feature Serving Optimization (30ms → 5ms):**
   ```python
   # Pre-compute features in Redis
   redis_client.hmset(f"user_features:{user_id}", features)

   # Use pipelining for multiple features
   pipeline = redis_client.pipeline()
   pipeline.hgetall(f"user_features:{user_id}")
   pipeline.hgetall(f"product_features:{product_id}")
   results = pipeline.execute()
   ```

3. **Request Batching (20ms → 10ms):**
   ```python
   # Batch 32 requests together
   # Single request: 20ms
   # Batched 32: 40ms / 32 = 1.25ms per request
   async def batch_predict(requests):
       batch = torch.stack([r.features for r in requests])
       predictions = model(batch)
       return predictions
   ```

4. **Caching (reduce 70% of requests):**
   ```python
   # Cache predictions for 5 minutes
   @cache(ttl=300)
   def get_prediction(user_id, product_id):
       return model.predict(...)
   ```

5. **Hardware Acceleration:**
   - GPU inference (TensorRT): 5-10x speedup
   - AWS Inferentia: Cost-effective for production

**Latency Breakdown (Before → After):**
- Feature serving: 30ms → 5ms
- Model inference: 50ms → 10ms (batching + quantization)
- API overhead: 20ms → 10ms (async)
- Caching: 0ms (70% cache hit)
- **Total: 200ms → 25ms** ✅

**Interview Follow-up:**
- "Trade-offs of quantization?" → Slight accuracy loss (0.5-1%)
- "When does batching not help?" → Strict latency SLA per request
- "GPU vs CPU for serving?" → GPU for high throughput, CPU for low latency

---

## Summary

**Real-Time ML Systems require:**
- ✅ Low-latency model serving (<100ms)
- ✅ Event-driven architecture (Kafka)
- ✅ Online feature stores (Redis)
- ✅ Optimization (quantization, batching, caching)
- ✅ Resilience (circuit breakers, fallbacks)

**Key Technologies:**
- **Serving:** TensorFlow Serving, TorchServe, ONNX Runtime
- **Streaming:** Kafka, Flink, Spark Streaming
- **Caching:** Redis, DynamoDB
- **Monitoring:** Prometheus, Grafana

**Next Steps:**
1. Build FastAPI model serving endpoint
2. Set up Kafka for event streaming
3. Implement online learning with feedback loop
4. Optimize model with quantization
5. Add caching and batching

---

**Generated with Claude Code**

*Last Updated: 2025-10-25*
