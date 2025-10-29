# ML System Design: Complete Interview Guide

## Table of Contents
1. [Introduction](#introduction)
2. [ML System Design Framework](#ml-system-design-framework)
3. [Key Components of ML Systems](#key-components-of-ml-systems)
4. [Common Design Patterns](#common-design-patterns)
5. [Real ML System Design Problems](#real-ml-system-design-problems)
6. [Scaling Considerations](#scaling-considerations)
7. [Trade-offs and Decision Making](#trade-offs-and-decision-making)
8. [Interview Questions](#interview-questions)

---

## Introduction

### What is ML System Design?

ML system design interviews assess your ability to:
- Design end-to-end ML systems that solve real business problems
- Make architecture decisions considering scalability, latency, and cost
- Identify appropriate ML approaches and techniques
- Navigate trade-offs between accuracy, latency, cost, and complexity
- Design data pipelines, training infrastructure, and serving systems
- Consider monitoring, debugging, and maintenance

### Why ML System Design Matters

Unlike algorithm interviews, system design interviews evaluate:
- **Practical ML engineering skills** - Building production systems, not just models
- **Architecture thinking** - Designing scalable, maintainable systems
- **Business sense** - Understanding requirements and constraints
- **Communication** - Explaining complex technical decisions
- **Trade-off analysis** - Balancing competing objectives

### Interview Format (45-60 minutes)

1. **Requirements gathering** (5-10 min) - Clarify problem, constraints, scale
2. **High-level design** (10-15 min) - Architecture diagram, major components
3. **Deep dive** (15-20 min) - Detailed discussion of key components
4. **Scalability & trade-offs** (10-15 min) - Scaling, monitoring, edge cases
5. **Q&A** (5 min) - Address interviewer questions

---

## ML System Design Framework

### Step 1: Clarify Requirements (5-10 minutes)

#### Functional Requirements
- **What is the business objective?** - Revenue, engagement, efficiency?
- **What are we predicting?** - Classification, regression, ranking, recommendation?
- **Who are the users?** - End users, internal teams, B2B customers?
- **What's the input/output?** - Data format, API contract
- **Online or offline predictions?** - Real-time, batch, or both?

#### Non-Functional Requirements
- **Scale** - QPS, data volume, number of users
- **Latency** - P50, P95, P99 requirements
- **Throughput** - Predictions per second
- **Availability** - SLA (99%, 99.9%, 99.99%?)
- **Cost constraints** - Compute budget, serving costs
- **Accuracy requirements** - Minimum acceptable performance

#### Example Questions to Ask

```
Interviewer: Design a spam detection system for email.

You should ask:
- "What's the email volume we need to handle?" (e.g., 100M emails/day)
- "What's the acceptable latency?" (e.g., <100ms per email)
- "What's the cost of false positives vs false negatives?" (false positive worse)
- "Do we need real-time detection or can we do batch processing?" (real-time)
- "What data do we have access to?" (email content, metadata, user history)
- "What's the acceptable accuracy?" (e.g., >99% precision on spam)
```

### Step 2: High-Level Design (10-15 minutes)

#### Core ML System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      ML SYSTEM ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────┘

Data Collection → Data Processing → Feature Engineering → Training
      ↓                 ↓                   ↓                ↓
  Logging          Cleaning           Feature Store    Model Training
  Storage          Validation         (Feast/Tecton)   (Distributed)
                                                             ↓
                                                        Model Registry
                                                             ↓
Online Prediction ← Model Serving ← Feature Serving ← Deployment
      ↓                 ↓                                    ↓
   API/SDK          Load Balancer                      Monitoring
   Response         (K8s/ECS)                          (Metrics/Alerts)
                         ↓
                    Feedback Loop
                         ↓
                   Retraining Pipeline
```

#### Key Design Decisions

1. **Model choice** - Traditional ML vs deep learning vs ensemble
2. **Training approach** - Batch, online, active learning
3. **Serving architecture** - Model server, embedded, edge
4. **Feature store** - Real-time, batch, or hybrid
5. **Monitoring strategy** - Model performance, data drift, system health

### Step 3: Detailed Component Design (15-20 minutes)

#### Data Pipeline

```python
"""
Design pattern: Lambda architecture for real-time + batch features
"""

# Batch layer: Historical features
class BatchFeaturePipeline:
    def __init__(self, spark_session):
        self.spark = spark_session

    def compute_user_features(self, start_date, end_date):
        """
        Compute aggregated user features from historical data
        Examples: 30-day avg spend, lifetime purchases, etc.
        """
        df = self.spark.read.parquet(f"s3://data/events/{start_date}/")

        user_features = df.groupBy("user_id").agg(
            F.avg("purchase_amount").alias("avg_purchase_30d"),
            F.count("purchase_id").alias("num_purchases_30d"),
            F.max("purchase_amount").alias("max_purchase_30d"),
            F.stddev("purchase_amount").alias("std_purchase_30d")
        )

        # Write to feature store
        user_features.write.mode("overwrite").parquet(
            "s3://features/user_features/batch/"
        )

        return user_features

# Speed layer: Real-time features
class StreamFeaturePipeline:
    def __init__(self, kafka_brokers):
        self.consumer = KafkaConsumer(
            'user-events',
            bootstrap_servers=kafka_brokers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

    def process_realtime_features(self):
        """
        Compute features from streaming events
        Examples: last action time, session features, etc.
        """
        for message in self.consumer:
            event = message.value

            # Compute real-time features
            features = {
                'user_id': event['user_id'],
                'last_activity_timestamp': event['timestamp'],
                'session_page_views': self.get_session_count(event['user_id']),
                'time_since_last_purchase': self.time_since_last_action(
                    event['user_id'], 'purchase'
                )
            }

            # Update feature store (e.g., Redis for low-latency lookup)
            self.feature_store.update_realtime_features(features)
```

#### Model Training Pipeline

```python
"""
Design pattern: Distributed training with experiment tracking
"""

class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.mlflow = mlflow

    def train_model(self, train_data, val_data):
        """
        Distributed training with hyperparameter tuning
        """
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(self.config)

            # Initialize distributed training
            if self.config['distributed']:
                model = self.train_distributed(train_data, val_data)
            else:
                model = self.train_single(train_data, val_data)

            # Evaluate model
            metrics = self.evaluate(model, val_data)
            mlflow.log_metrics(metrics)

            # Save model
            mlflow.pytorch.log_model(model, "model")

            # Register model if metrics pass threshold
            if metrics['auc'] > self.config['min_auc']:
                self.register_model(model, metrics)

            return model

    def train_distributed(self, train_data, val_data):
        """
        Distributed training using PyTorch DDP
        """
        # Initialize distributed process group
        dist.init_process_group(backend='nccl')

        # Wrap model in DDP
        model = MyModel(self.config)
        model = nn.parallel.DistributedDataParallel(model)

        # Distributed data sampler
        train_sampler = DistributedSampler(train_data)
        train_loader = DataLoader(
            train_data,
            batch_size=self.config['batch_size'],
            sampler=train_sampler
        )

        # Training loop
        optimizer = torch.optim.Adam(model.parameters())
        for epoch in range(self.config['epochs']):
            train_sampler.set_epoch(epoch)

            for batch in train_loader:
                loss = self.train_step(model, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model

# Continuous training pipeline
class ContinuousTrainingPipeline:
    def __init__(self, airflow_dag):
        self.dag = airflow_dag

    def create_training_dag(self):
        """
        Airflow DAG for automated retraining
        """
        with DAG(
            'ml_model_training',
            schedule_interval='@daily',
            start_date=datetime(2025, 1, 1)
        ) as dag:

            # Task 1: Extract training data
            extract_data = PythonOperator(
                task_id='extract_data',
                python_callable=self.extract_training_data
            )

            # Task 2: Feature engineering
            engineer_features = PythonOperator(
                task_id='engineer_features',
                python_callable=self.engineer_features
            )

            # Task 3: Train model
            train_model = PythonOperator(
                task_id='train_model',
                python_callable=self.train_model
            )

            # Task 4: Evaluate model
            evaluate_model = PythonOperator(
                task_id='evaluate_model',
                python_callable=self.evaluate_model
            )

            # Task 5: Deploy if better
            deploy_model = BranchPythonOperator(
                task_id='deploy_model',
                python_callable=self.deploy_if_better
            )

            extract_data >> engineer_features >> train_model >> evaluate_model >> deploy_model

        return dag
```

#### Model Serving

```python
"""
Design pattern: Multi-model serving with A/B testing
"""

from fastapi import FastAPI
import torch
import redis

app = FastAPI()

class ModelServer:
    def __init__(self):
        self.models = {}
        self.feature_store = redis.Redis(host='localhost', port=6379)
        self.load_models()

    def load_models(self):
        """
        Load multiple model versions for A/B testing
        """
        self.models['model_v1'] = torch.jit.load('models/model_v1.pt')
        self.models['model_v2'] = torch.jit.load('models/model_v2.pt')
        self.models['model_v1'].eval()
        self.models['model_v2'].eval()

    def get_features(self, user_id, item_id):
        """
        Fetch features from feature store
        Combines batch (S3) and real-time (Redis) features
        """
        # Real-time features from Redis
        realtime_features = self.feature_store.hgetall(f"user:{user_id}")

        # Batch features (cached in Redis, fallback to S3)
        batch_features = self.feature_store.hgetall(f"user_batch:{user_id}")
        if not batch_features:
            batch_features = self.load_from_s3(user_id)

        # Combine features
        features = {**realtime_features, **batch_features}
        return self.preprocess_features(features)

    def select_model(self, user_id):
        """
        A/B test: Route users to different model versions
        """
        # Simple hash-based assignment (consistent for same user)
        assignment = hash(user_id) % 100
        if assignment < 90:  # 90% traffic to v1
            return 'model_v1'
        else:  # 10% traffic to v2
            return 'model_v2'

    def predict(self, user_id, item_id):
        """
        Make prediction with selected model
        """
        # Get features
        features = self.get_features(user_id, item_id)

        # Select model
        model_name = self.select_model(user_id)
        model = self.models[model_name]

        # Predict
        with torch.no_grad():
            prediction = model(features)

        # Log for monitoring
        self.log_prediction(user_id, item_id, model_name, prediction)

        return {
            'user_id': user_id,
            'item_id': item_id,
            'score': prediction.item(),
            'model_version': model_name
        }

@app.post("/predict")
async def predict_endpoint(user_id: str, item_id: str):
    """
    Prediction API endpoint
    """
    server = ModelServer()
    result = server.predict(user_id, item_id)
    return result

# Deployment configuration (Kubernetes)
"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-server
spec:
  replicas: 5
  selector:
    matchLabels:
      app: ml-model-server
  template:
    metadata:
      labels:
        app: ml-model-server
    spec:
      containers:
      - name: model-server
        image: my-ml-server:v1
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: MODEL_PATH
          value: "/models"
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
"""
```

#### Monitoring & Feedback Loop

```python
"""
Design pattern: Comprehensive monitoring with automated alerts
"""

from prometheus_client import Counter, Histogram, Gauge
import time

class ModelMonitoring:
    def __init__(self):
        # Prometheus metrics
        self.prediction_counter = Counter(
            'model_predictions_total',
            'Total predictions made',
            ['model_version', 'status']
        )

        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds',
            'Prediction latency',
            ['model_version']
        )

        self.model_score_distribution = Histogram(
            'model_score_distribution',
            'Distribution of prediction scores',
            ['model_version'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        self.feature_drift_score = Gauge(
            'feature_drift_score',
            'Feature drift detection score',
            ['feature_name']
        )

    def log_prediction(self, model_version, score, latency, status='success'):
        """
        Log prediction metrics
        """
        self.prediction_counter.labels(
            model_version=model_version,
            status=status
        ).inc()

        self.prediction_latency.labels(
            model_version=model_version
        ).observe(latency)

        self.model_score_distribution.labels(
            model_version=model_version
        ).observe(score)

    def detect_drift(self, reference_data, current_data):
        """
        Detect feature drift using KS test
        """
        from scipy import stats

        drift_scores = {}
        for feature in reference_data.columns:
            ks_statistic, p_value = stats.ks_2samp(
                reference_data[feature],
                current_data[feature]
            )

            drift_scores[feature] = ks_statistic

            # Update Prometheus gauge
            self.feature_drift_score.labels(
                feature_name=feature
            ).set(ks_statistic)

            # Alert if drift detected
            if ks_statistic > 0.1:  # Threshold
                self.send_alert(
                    f"Drift detected in feature {feature}: KS={ks_statistic:.3f}"
                )

        return drift_scores

    def monitor_model_performance(self, predictions, actuals):
        """
        Monitor online model performance
        """
        from sklearn.metrics import roc_auc_score, log_loss

        # Calculate metrics on recent predictions with ground truth
        auc = roc_auc_score(actuals, predictions)
        logloss = log_loss(actuals, predictions)

        # Check if performance degraded
        if auc < 0.70:  # Threshold
            self.send_alert(
                f"Model performance degraded: AUC={auc:.3f}"
            )

        return {'auc': auc, 'log_loss': logloss}

# Feedback loop
class FeedbackLoop:
    def __init__(self):
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def collect_feedback(self, user_id, item_id, prediction, action):
        """
        Collect user feedback for model improvement
        """
        feedback = {
            'user_id': user_id,
            'item_id': item_id,
            'prediction': prediction,
            'action': action,  # e.g., click, purchase, skip
            'timestamp': time.time()
        }

        # Send to Kafka for retraining pipeline
        self.kafka_producer.send('model-feedback', feedback)

    def create_training_data_from_feedback(self):
        """
        Create training examples from user feedback
        """
        # Consume feedback events
        consumer = KafkaConsumer(
            'model-feedback',
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='earliest'
        )

        training_examples = []
        for message in consumer:
            feedback = json.loads(message.value)

            # Label based on user action
            label = 1 if feedback['action'] in ['click', 'purchase'] else 0

            training_examples.append({
                'user_id': feedback['user_id'],
                'item_id': feedback['item_id'],
                'label': label,
                'timestamp': feedback['timestamp']
            })

        return training_examples
```

### Step 4: Scaling & Trade-offs (10-15 minutes)

#### Scaling Strategies

```python
"""
Design pattern: Multi-stage caching for high QPS
"""

class ScalableInferenceSystem:
    def __init__(self):
        self.l1_cache = {}  # In-memory cache (LRU)
        self.l2_cache = redis.Redis()  # Redis cache
        self.model_server = ModelServer()

    def get_prediction_cached(self, user_id, item_id):
        """
        Three-tier caching strategy:
        L1: In-memory (microseconds)
        L2: Redis (milliseconds)
        L3: Model inference (tens of milliseconds)
        """
        cache_key = f"{user_id}:{item_id}"

        # L1: Check in-memory cache
        if cache_key in self.l1_cache:
            return self.l1_cache[cache_key]

        # L2: Check Redis
        cached_result = self.l2_cache.get(cache_key)
        if cached_result:
            result = json.loads(cached_result)
            self.l1_cache[cache_key] = result  # Populate L1
            return result

        # L3: Model inference
        result = self.model_server.predict(user_id, item_id)

        # Cache results
        self.l2_cache.setex(cache_key, 3600, json.dumps(result))  # 1 hour TTL
        self.l1_cache[cache_key] = result

        return result

# Horizontal scaling with load balancing
"""
Architecture for 10K+ QPS:

              Load Balancer (AWS ALB / NGINX)
                         |
        ┌────────────────┼────────────────┐
        │                │                │
    Server 1         Server 2        Server 3
    (4 cores)        (4 cores)       (4 cores)
        │                │                │
        └────────────────┼────────────────┘
                         |
                  Redis Cluster
                  (Feature Cache)
                         |
                  Model Storage (S3)
"""

# Batch prediction for high throughput
class BatchInferenceSystem:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.model = torch.jit.load('model.pt')
        self.model.eval()

    def predict_batch(self, requests):
        """
        Batch requests for higher throughput
        Trades latency for throughput
        """
        # Group requests into batches
        batches = [requests[i:i+self.batch_size]
                  for i in range(0, len(requests), self.batch_size)]

        results = []
        for batch in batches:
            # Prepare batch input
            features = torch.stack([req['features'] for req in batch])

            # Batch inference
            with torch.no_grad():
                predictions = self.model(features)

            # Map predictions back to requests
            for req, pred in zip(batch, predictions):
                results.append({
                    'request_id': req['id'],
                    'prediction': pred.item()
                })

        return results
```

---

## Common Design Patterns

### Pattern 1: Two-Tower Architecture (Retrieval + Ranking)

Used in: Recommendation systems, search engines

```
┌─────────────────────────────────────────────────┐
│              RECOMMENDATION SYSTEM               │
└─────────────────────────────────────────────────┘

User Query → Retrieval Model → Top 1000 Candidates
                                       ↓
                                Ranking Model → Top 10 Results
```

**When to use:**
- Large candidate set (millions of items)
- Need sub-second latency
- Two-stage approach: fast retrieval + precise ranking

**Implementation:**

```python
class TwoTowerRecommendation:
    def __init__(self):
        self.retrieval_model = RetrievalModel()  # Simple, fast
        self.ranking_model = RankingModel()      # Complex, accurate
        self.ann_index = FaissIndex()            # Approximate nearest neighbor

    def recommend(self, user_id, k=10):
        """
        Two-stage recommendation
        """
        # Stage 1: Retrieval (fast, broad)
        user_embedding = self.retrieval_model.encode_user(user_id)
        candidate_ids = self.ann_index.search(user_embedding, k=1000)

        # Stage 2: Ranking (slow, precise)
        candidates = self.get_candidate_features(user_id, candidate_ids)
        scores = self.ranking_model.predict(candidates)

        # Return top-k
        top_k_indices = np.argsort(scores)[-k:]
        return [candidate_ids[i] for i in top_k_indices]

class RetrievalModel(nn.Module):
    """
    Fast retrieval using dot product similarity
    """
    def __init__(self, num_users, num_items, embedding_dim=128):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def encode_user(self, user_id):
        return self.user_embedding(user_id)

    def encode_item(self, item_id):
        return self.item_embedding(item_id)

class RankingModel(nn.Module):
    """
    Complex ranking with many features
    """
    def __init__(self, input_dim=200):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.network(features)
```

### Pattern 2: Online Learning with Bandit Algorithms

Used in: Content recommendation, ad placement, dynamic pricing

```python
class ContextualBandit:
    """
    Explore-exploit trade-off with Thompson Sampling
    """
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Beta distribution parameters
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self, context):
        """
        Select arm using Thompson Sampling
        """
        # Sample from posterior
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, reward):
        """
        Update posterior with observed reward
        """
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

# Online learning pipeline
class OnlineLearningSystem:
    def __init__(self):
        self.model = OnlineModel()
        self.bandit = ContextualBandit(n_arms=10)

    def serve_and_learn(self, user_id):
        """
        Serve prediction and learn from feedback in real-time
        """
        # Get user context
        context = self.get_user_context(user_id)

        # Select arm (action)
        arm = self.bandit.select_arm(context)

        # Serve prediction
        prediction = self.model.predict(context, arm)

        # Wait for user feedback (async)
        feedback = self.collect_feedback(user_id, arm)

        # Update model online
        self.model.partial_fit(context, arm, feedback)
        self.bandit.update(arm, feedback)

        return prediction
```

### Pattern 3: Multi-Armed Multi-Stage (MAMS) Pipeline

Used in: Content moderation, fraud detection, spam filtering

```
┌─────────────────────────────────────────────┐
│          MULTI-STAGE ML PIPELINE             │
└─────────────────────────────────────────────┘

Input → Rule-Based Filter → Fast ML Model → Complex ML Model → Human Review
         (90% filtered)      (99% coverage)    (0.99% coverage)  (0.01%)

Each stage has different:
- Latency requirements
- Accuracy requirements
- Cost per prediction
```

**Implementation:**

```python
class MultiStageFilteringPipeline:
    def __init__(self):
        self.rule_filter = RuleBasedFilter()
        self.fast_model = FastClassifier()
        self.complex_model = ComplexClassifier()

    def filter(self, content):
        """
        Multi-stage filtering with early exits
        """
        # Stage 1: Rule-based (microseconds, 90% recall)
        if self.rule_filter.is_obviously_spam(content):
            return {'label': 'spam', 'confidence': 1.0, 'stage': 'rules'}

        if self.rule_filter.is_obviously_legit(content):
            return {'label': 'legit', 'confidence': 1.0, 'stage': 'rules'}

        # Stage 2: Fast ML model (milliseconds, 99% recall)
        fast_score = self.fast_model.predict_proba(content)
        if fast_score > 0.95:
            return {'label': 'spam', 'confidence': fast_score, 'stage': 'fast_ml'}
        if fast_score < 0.05:
            return {'label': 'legit', 'confidence': 1-fast_score, 'stage': 'fast_ml'}

        # Stage 3: Complex model (hundreds of milliseconds, 99.9% recall)
        complex_score = self.complex_model.predict_proba(content)
        if complex_score > 0.9 or complex_score < 0.1:
            label = 'spam' if complex_score > 0.5 else 'legit'
            return {'label': label, 'confidence': complex_score, 'stage': 'complex_ml'}

        # Stage 4: Human review (uncertain cases)
        return {'label': 'review', 'confidence': complex_score, 'stage': 'human'}

# Cost analysis
"""
Assumptions:
- 1M items per day
- Rule-based: $0 per prediction
- Fast model: $0.0001 per prediction
- Complex model: $0.001 per prediction
- Human review: $0.1 per item

Without multi-stage: 1M * $0.001 = $1,000/day
With multi-stage:
  - Rules: 900K items * $0 = $0
  - Fast ML: 90K items * $0.0001 = $9
  - Complex ML: 9K items * $0.001 = $9
  - Human: 100 items * $0.1 = $10
  Total: $28/day (97% cost reduction!)
"""
```

---

## Real ML System Design Problems

### Problem 1: Design YouTube Recommendation System

**Requirements:**
- 500M daily active users
- 2B videos in catalog
- Real-time recommendations (<100ms)
- Personalized based on watch history

**Solution:**

```
┌─────────────────────────────────────────────────────────────┐
│              YOUTUBE RECOMMENDATION ARCHITECTURE             │
└─────────────────────────────────────────────────────────────┘

User Watch History → Feature Engineering → User Embedding
                                                  ↓
Video Catalog → Video Features → Video Embedding → ANN Index
                                                  ↓
                                            Retrieval (1000 videos)
                                                  ↓
                                  Ranking Model (Top 20 videos)
                                                  ↓
                              Diversity & Business Rules (Top 10)
```

**Detailed Design:**

```python
class YouTubeRecommendationSystem:
    """
    Production-grade recommendation system
    """
    def __init__(self):
        # Models
        self.candidate_generation = CandidateGenerationModel()
        self.ranking_model = RankingModel()

        # Infrastructure
        self.feature_store = Feast()
        self.ann_index = ScaNN()  # Google's ANN library
        self.cache = Redis()

    def recommend(self, user_id, num_recommendations=10):
        """
        Generate personalized recommendations
        """
        # 1. Candidate Generation (1000 videos from 2B catalog)
        candidates = self.generate_candidates(user_id, k=1000)

        # 2. Ranking (score all candidates)
        ranked_candidates = self.rank_candidates(user_id, candidates)

        # 3. Post-processing (diversity, business rules)
        final_recommendations = self.post_process(
            ranked_candidates,
            num_recommendations
        )

        return final_recommendations

    def generate_candidates(self, user_id, k=1000):
        """
        Multi-source candidate generation
        """
        candidates = []

        # Source 1: Collaborative filtering (50% of candidates)
        user_embedding = self.get_user_embedding(user_id)
        cf_candidates = self.ann_index.search(user_embedding, k=500)
        candidates.extend(cf_candidates)

        # Source 2: Follow graph (20% of candidates)
        followed_channels = self.get_followed_channels(user_id)
        recent_uploads = self.get_recent_uploads(followed_channels, k=200)
        candidates.extend(recent_uploads)

        # Source 3: Trending videos (15% of candidates)
        trending = self.get_trending_videos(k=150)
        candidates.extend(trending)

        # Source 4: Similar to recently watched (15% of candidates)
        recent_watched = self.get_recent_watches(user_id, k=10)
        similar = self.get_similar_videos(recent_watched, k=150)
        candidates.extend(similar)

        return list(set(candidates))  # Remove duplicates

    def rank_candidates(self, user_id, candidates):
        """
        Ranking with multiple objectives
        """
        # Get user and video features
        user_features = self.feature_store.get_user_features(user_id)
        video_features = self.feature_store.get_video_features(candidates)

        # Combine features
        combined_features = self.combine_features(
            user_features,
            video_features
        )

        # Predict multiple objectives
        predictions = self.ranking_model.predict(combined_features)
        # predictions = {
        #   'watch_time': [...],
        #   'click_probability': [...],
        #   'like_probability': [...]
        # }

        # Weighted combination of objectives
        scores = (
            0.6 * predictions['watch_time'] +
            0.3 * predictions['click_probability'] +
            0.1 * predictions['like_probability']
        )

        # Sort by score
        ranked_indices = np.argsort(scores)[::-1]
        return [candidates[i] for i in ranked_indices]

    def post_process(self, ranked_videos, k):
        """
        Apply diversity and business rules
        """
        final_list = []
        seen_channels = set()

        for video in ranked_videos:
            if len(final_list) >= k:
                break

            # Diversity constraint: max 2 videos per channel
            channel_id = video['channel_id']
            channel_count = sum(1 for v in final_list if v['channel_id'] == channel_id)
            if channel_count >= 2:
                continue

            # Business rule: No sensitive content
            if video['content_rating'] == 'sensitive':
                continue

            final_list.append(video)

        return final_list

# Feature engineering
class YouTubeFeatures:
    def compute_user_features(self, user_id):
        """
        User features from watch history
        """
        watch_history = self.db.get_watch_history(user_id, days=30)

        return {
            'avg_watch_time': np.mean([w['watch_time'] for w in watch_history]),
            'favorite_categories': self.top_categories(watch_history),
            'watch_frequency': len(watch_history) / 30,
            'time_of_day_preference': self.time_distribution(watch_history),
            'device_type': self.most_common_device(watch_history)
        }

    def compute_video_features(self, video_id):
        """
        Video features
        """
        video = self.db.get_video(video_id)
        stats = self.db.get_video_stats(video_id)

        return {
            'duration': video['duration'],
            'upload_date': video['upload_date'],
            'view_count': stats['views'],
            'like_rate': stats['likes'] / stats['views'],
            'avg_watch_time': stats['avg_watch_time'],
            'category': video['category'],
            'tags': video['tags'],
            'video_quality': video['resolution']
        }

# Ranking model architecture
class RankingModel(nn.Module):
    """
    Multi-objective ranking model
    """
    def __init__(self, user_feature_dim=100, video_feature_dim=150):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(user_feature_dim + video_feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Task-specific heads
        self.watch_time_head = nn.Linear(256, 1)
        self.click_head = nn.Linear(256, 1)
        self.like_head = nn.Linear(256, 1)

    def forward(self, user_features, video_features):
        combined = torch.cat([user_features, video_features], dim=1)
        shared_repr = self.shared(combined)

        return {
            'watch_time': self.watch_time_head(shared_repr),
            'click_probability': torch.sigmoid(self.click_head(shared_repr)),
            'like_probability': torch.sigmoid(self.like_head(shared_repr))
        }

# Scaling considerations
"""
INFRASTRUCTURE:

1. Feature Store (Feast + Redis):
   - Real-time features: Redis (user sessions, recent watches)
   - Batch features: BigQuery + Redis cache (historical aggregations)

2. Model Serving:
   - Candidate generation: TensorFlow Serving (GPU instances)
   - Ranking: TorchServe (GPU instances)
   - Load balancer: AWS ALB
   - Replicas: 100+ instances

3. ANN Index:
   - Library: ScaNN (Google) or Faiss (Facebook)
   - Index size: 2B vectors × 128 dimensions
   - Sharding: 100 shards, each handling 20M vectors
   - Update frequency: Daily batch rebuild

4. Caching:
   - L1: Application cache (most frequent users)
   - L2: Redis cluster (100+ nodes)
   - L3: Model inference

5. Data Pipeline:
   - Stream processing: Kafka + Flink (real-time features)
   - Batch processing: Spark (daily aggregations)
   - Training data: Daily snapshot to BigQuery

LATENCY BREAKDOWN (P95):
- Feature lookup: 10ms
- Candidate generation: 30ms
- Ranking: 40ms
- Post-processing: 10ms
- Total: 90ms

COST ESTIMATE:
- Compute: ~$50K/month (GPU instances)
- Storage: ~$10K/month (features + models)
- Data transfer: ~$5K/month
- Total: ~$65K/month for 500M DAU
- Cost per user: ~$0.0043/month
"""
```

### Problem 2: Design Real-Time Fraud Detection System

**Requirements:**
- Detect fraudulent transactions in real-time (<50ms)
- 10,000 transactions per second
- False positive rate <0.1% (legitimate transactions blocked)
- Fraud detection rate >95%

**Solution:**

```
┌─────────────────────────────────────────────────────────────┐
│           REAL-TIME FRAUD DETECTION ARCHITECTURE             │
└─────────────────────────────────────────────────────────────┘

Transaction → Feature Engineering → Rule Engine → ML Model → Decision
     ↓               ↓                    ↓            ↓          ↓
  Logging      Real-time +          Hard Rules    Gradient   Accept/
             Historical              (instant)    Boosting   Reject/
               Features                           Ensemble    Review
```

**Detailed Design:**

```python
class FraudDetectionSystem:
    """
    Real-time fraud detection with <50ms latency
    """
    def __init__(self):
        self.rule_engine = RuleEngine()
        self.feature_store = FeatureStore()
        self.model = FraudModel()
        self.risk_scorer = RiskScorer()

    def evaluate_transaction(self, transaction):
        """
        Evaluate transaction for fraud
        """
        start_time = time.time()

        # Stage 1: Rule-based filtering (instant, ~5ms)
        rule_result = self.rule_engine.evaluate(transaction)
        if rule_result['action'] != 'continue':
            return rule_result

        # Stage 2: Feature engineering (~10ms)
        features = self.feature_store.get_features(transaction)

        # Stage 3: ML model prediction (~20ms)
        fraud_probability = self.model.predict(features)

        # Stage 4: Risk-based decision (~5ms)
        decision = self.risk_scorer.make_decision(
            transaction,
            fraud_probability
        )

        latency = (time.time() - start_time) * 1000
        self.log_transaction(transaction, decision, latency)

        return decision

class RuleEngine:
    """
    Hard rules for obvious fraud patterns
    """
    def evaluate(self, txn):
        """
        Check hard rules
        """
        # Rule 1: Impossible velocity
        if self.is_impossible_velocity(txn):
            return {'action': 'reject', 'reason': 'impossible_velocity'}

        # Rule 2: Blacklisted card
        if self.is_blacklisted(txn['card_number']):
            return {'action': 'reject', 'reason': 'blacklisted_card'}

        # Rule 3: Suspicious country
        if self.is_high_risk_country(txn['country']):
            return {'action': 'review', 'reason': 'high_risk_country'}

        # Rule 4: Unusual amount
        if self.is_unusual_amount(txn):
            return {'action': 'review', 'reason': 'unusual_amount'}

        return {'action': 'continue'}

    def is_impossible_velocity(self, txn):
        """
        Check if two transactions from same card are too close in time
        but too far in distance (impossible travel)
        """
        user_id = txn['user_id']
        last_txn = self.cache.get(f"last_txn:{user_id}")

        if last_txn:
            time_diff = txn['timestamp'] - last_txn['timestamp']
            distance = self.haversine_distance(
                txn['location'],
                last_txn['location']
            )

            # If distance > 1000km and time < 1 hour
            if distance > 1000 and time_diff < 3600:
                return True

        return False

class FeatureStore:
    """
    Real-time + historical features
    """
    def __init__(self):
        self.redis = redis.Redis()
        self.bigquery = bigquery.Client()

    def get_features(self, txn):
        """
        Combine real-time and historical features
        """
        user_id = txn['user_id']

        # Real-time features (from Redis, ~5ms)
        realtime = self.get_realtime_features(user_id)

        # Historical features (from BigQuery, cached in Redis, ~5ms)
        historical = self.get_historical_features(user_id)

        # Transaction features
        transaction = self.extract_transaction_features(txn)

        return {**realtime, **historical, **transaction}

    def get_realtime_features(self, user_id):
        """
        Features from recent activity (last hour)
        """
        recent_txns = self.redis.lrange(f"txns:{user_id}", 0, 100)

        if not recent_txns:
            return {
                'txn_count_1h': 0,
                'total_amount_1h': 0,
                'unique_merchants_1h': 0
            }

        txns = [json.loads(t) for t in recent_txns]
        now = time.time()
        recent_1h = [t for t in txns if now - t['timestamp'] < 3600]

        return {
            'txn_count_1h': len(recent_1h),
            'total_amount_1h': sum(t['amount'] for t in recent_1h),
            'unique_merchants_1h': len(set(t['merchant_id'] for t in recent_1h)),
            'avg_amount_1h': np.mean([t['amount'] for t in recent_1h]) if recent_1h else 0
        }

    def get_historical_features(self, user_id):
        """
        Features from long-term history (cached)
        """
        # Check cache first
        cached = self.redis.hgetall(f"hist_features:{user_id}")
        if cached:
            return {k: float(v) for k, v in cached.items()}

        # Query BigQuery (fallback)
        query = f"""
        SELECT
            COUNT(*) as lifetime_txn_count,
            AVG(amount) as avg_txn_amount,
            STDDEV(amount) as std_txn_amount,
            MAX(amount) as max_txn_amount,
            COUNT(DISTINCT merchant_id) as num_merchants,
            COUNT(DISTINCT country) as num_countries
        FROM transactions
        WHERE user_id = '{user_id}'
          AND timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
        """

        result = self.bigquery.query(query).to_dataframe().iloc[0].to_dict()

        # Cache for 1 hour
        self.redis.hmset(f"hist_features:{user_id}", result)
        self.redis.expire(f"hist_features:{user_id}", 3600)

        return result

    def extract_transaction_features(self, txn):
        """
        Features from current transaction
        """
        return {
            'amount': txn['amount'],
            'merchant_category': self.encode_category(txn['merchant_category']),
            'hour_of_day': datetime.fromtimestamp(txn['timestamp']).hour,
            'day_of_week': datetime.fromtimestamp(txn['timestamp']).weekday(),
            'is_international': int(txn['country'] != txn['user_country']),
            'is_online': int(txn['transaction_type'] == 'online'),
            'device_type': self.encode_device(txn['device_type'])
        }

class FraudModel:
    """
    Ensemble model for fraud detection
    """
    def __init__(self):
        # Load pre-trained models
        self.xgboost_model = xgb.Booster()
        self.xgboost_model.load_model('models/xgboost_fraud.json')

        self.lightgbm_model = lgb.Booster(model_file='models/lightgbm_fraud.txt')

        # Model weights (tuned on validation set)
        self.weights = {'xgboost': 0.6, 'lightgbm': 0.4}

    def predict(self, features):
        """
        Ensemble prediction
        """
        # Convert features to model format
        feature_vector = self.prepare_features(features)

        # Get predictions from both models
        xgb_pred = self.xgboost_model.predict(
            xgb.DMatrix(feature_vector)
        )[0]

        lgb_pred = self.lightgbm_model.predict(feature_vector)[0]

        # Weighted average
        fraud_prob = (
            self.weights['xgboost'] * xgb_pred +
            self.weights['lightgbm'] * lgb_pred
        )

        return fraud_prob

class RiskScorer:
    """
    Convert fraud probability to business decision
    """
    def make_decision(self, transaction, fraud_probability):
        """
        Risk-based decision making
        """
        amount = transaction['amount']

        # Dynamic thresholds based on transaction amount
        if amount < 100:
            # Low-value transactions: more lenient
            reject_threshold = 0.95
            review_threshold = 0.85
        elif amount < 1000:
            # Medium-value transactions
            reject_threshold = 0.90
            review_threshold = 0.75
        else:
            # High-value transactions: more strict
            reject_threshold = 0.80
            review_threshold = 0.60

        if fraud_probability >= reject_threshold:
            return {
                'action': 'reject',
                'reason': 'high_fraud_probability',
                'fraud_score': fraud_probability
            }
        elif fraud_probability >= review_threshold:
            return {
                'action': 'review',
                'reason': 'moderate_fraud_probability',
                'fraud_score': fraud_probability
            }
        else:
            return {
                'action': 'approve',
                'fraud_score': fraud_probability
            }

# Training pipeline with feedback loop
class FraudModelTraining:
    def __init__(self):
        self.mlflow = mlflow

    def train_model(self, start_date, end_date):
        """
        Train fraud detection model
        """
        # Extract training data
        query = f"""
        SELECT
            t.*,
            IFNULL(f.is_fraud, 0) as label
        FROM transactions t
        LEFT JOIN fraud_labels f USING (transaction_id)
        WHERE t.timestamp BETWEEN '{start_date}' AND '{end_date}'
        """

        df = self.bigquery.query(query).to_dataframe()

        # Handle class imbalance (fraud is rare ~0.1%)
        fraud_df = df[df['label'] == 1]
        non_fraud_df = df[df['label'] == 0].sample(n=len(fraud_df) * 10)
        balanced_df = pd.concat([fraud_df, non_fraud_df])

        # Feature engineering
        X = self.engineer_features(balanced_df)
        y = balanced_df['label']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )

        # Train XGBoost
        xgb_model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            scale_pos_weight=10  # Address class imbalance
        )
        xgb_model.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        # Calculate precision at various recall levels
        precision_at_95_recall = self.precision_at_recall(
            y_test, y_pred_proba, recall_level=0.95
        )

        print(f"AUC: {auc:.4f}")
        print(f"Precision at 95% recall: {precision_at_95_recall:.4f}")

        # Save model
        xgb_model.save_model('models/xgboost_fraud.json')

        return xgb_model

# Infrastructure scaling
"""
ARCHITECTURE FOR 10K TPS:

1. API Layer:
   - Load balancer: AWS ALB
   - Compute: 50 EC2 instances (c5.2xlarge)
   - Each instance: 200 TPS capacity

2. Feature Store:
   - Redis Cluster: 20 nodes (cache.r5.2xlarge)
   - BigQuery: For historical features (cached)

3. Model Serving:
   - In-memory model loading (fast inference)
   - XGBoost: ~5ms inference on CPU
   - No GPU needed (tree-based models are CPU-efficient)

4. Data Pipeline:
   - Kafka: Ingest all transactions
   - Flink: Real-time feature computation
   - Airflow: Daily model retraining

5. Monitoring:
   - Prometheus + Grafana: Latency, TPS, fraud rate
   - ELK Stack: Transaction logs
   - PagerDuty: Alerting

COST BREAKDOWN (monthly):
- Compute (EC2): ~$10K
- Redis Cluster: ~$5K
- BigQuery: ~$2K
- Kafka + Flink: ~$3K
- Total: ~$20K/month for 10K TPS
- Cost per transaction: ~$0.000076

LATENCY SLA:
- P50: 25ms
- P95: 45ms
- P99: 65ms
- P99.9: 100ms
"""
```

### Problem 3: Design Search Ranking System

**Requirements:**
- Search engine for e-commerce platform
- 10M products
- 1000 queries per second
- Personalized search results
- <200ms end-to-end latency

**Solution:**

```
┌─────────────────────────────────────────────────────────────┐
│                SEARCH RANKING ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────┘

Query → Query Understanding → Candidate Retrieval → Ranking → Results
  ↓            ↓                      ↓                ↓          ↓
Spell    Intent/Entity           Elasticsearch      L2R       Top 20
Check    Classification          + semantic         Model     Products
                                 search (1000)
```

**Detailed Design:**

```python
class SearchRankingSystem:
    """
    E-commerce search with learning-to-rank
    """
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.retrieval = CandidateRetrieval()
        self.ranker = LearningToRank()

    def search(self, query, user_id, num_results=20):
        """
        End-to-end search pipeline
        """
        # Stage 1: Query understanding (~10ms)
        processed_query = self.query_processor.process(query)

        # Stage 2: Candidate retrieval (~50ms)
        candidates = self.retrieval.retrieve(processed_query, k=1000)

        # Stage 3: Ranking (~100ms)
        ranked_results = self.ranker.rank(
            processed_query,
            candidates,
            user_id
        )

        # Stage 4: Post-processing (~10ms)
        final_results = self.post_process(ranked_results, num_results)

        return final_results

class QueryProcessor:
    """
    Query understanding and preprocessing
    """
    def __init__(self):
        self.spell_checker = SpellChecker()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()

    def process(self, query):
        """
        Process and understand query
        """
        # Spell correction
        corrected_query = self.spell_checker.correct(query)

        # Intent classification (navigational, informational, transactional)
        intent = self.intent_classifier.predict(corrected_query)

        # Entity extraction (brand, category, attributes)
        entities = self.entity_extractor.extract(corrected_query)

        # Query expansion (synonyms, related terms)
        expanded_terms = self.expand_query(corrected_query)

        return {
            'original': query,
            'corrected': corrected_query,
            'intent': intent,
            'entities': entities,
            'expanded_terms': expanded_terms
        }

class CandidateRetrieval:
    """
    Multi-strategy retrieval
    """
    def __init__(self):
        self.elasticsearch = Elasticsearch(['localhost:9200'])
        self.semantic_search = SemanticSearch()

    def retrieve(self, processed_query, k=1000):
        """
        Combine lexical and semantic search
        """
        # Strategy 1: Elasticsearch (BM25, ~30ms)
        es_results = self.elasticsearch_search(processed_query, k=700)

        # Strategy 2: Semantic search (embeddings, ~20ms)
        semantic_results = self.semantic_search.search(
            processed_query['corrected'], k=300
        )

        # Merge and deduplicate
        all_candidates = self.merge_results(es_results, semantic_results)

        return all_candidates[:k]

    def elasticsearch_search(self, processed_query, k):
        """
        Elasticsearch query with boosting
        """
        query = {
            "query": {
                "bool": {
                    "should": [
                        # Exact match in title (highest boost)
                        {
                            "match": {
                                "title": {
                                    "query": processed_query['corrected'],
                                    "boost": 3.0
                                }
                            }
                        },
                        # Fuzzy match in title
                        {
                            "match": {
                                "title": {
                                    "query": processed_query['corrected'],
                                    "fuzziness": "AUTO",
                                    "boost": 2.0
                                }
                            }
                        },
                        # Match in description
                        {
                            "match": {
                                "description": {
                                    "query": processed_query['corrected'],
                                    "boost": 1.0
                                }
                            }
                        },
                        # Match expanded terms
                        {
                            "terms": {
                                "tags": processed_query['expanded_terms'],
                                "boost": 1.5
                            }
                        }
                    ],
                    # Filter by entities (brand, category)
                    "filter": self.build_filters(processed_query['entities'])
                }
            },
            "size": k
        }

        results = self.elasticsearch.search(index="products", body=query)
        return [hit['_source'] for hit in results['hits']['hits']]

class SemanticSearch:
    """
    Dense retrieval using embeddings
    """
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index('product_embeddings.index')
        self.product_ids = np.load('product_ids.npy')

    def search(self, query, k=300):
        """
        Semantic search using FAISS
        """
        # Encode query
        query_embedding = self.encoder.encode([query])

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)

        # Get product IDs
        product_ids = self.product_ids[indices[0]]

        # Fetch product data
        products = self.fetch_products(product_ids)

        return products

class LearningToRank:
    """
    Learning-to-rank model
    """
    def __init__(self):
        self.model = lgb.Booster(model_file='models/ltr_model.txt')
        self.feature_store = FeatureStore()

    def rank(self, query, candidates, user_id):
        """
        Rank candidates using L2R model
        """
        # Extract features for all query-document pairs
        features = self.extract_features(query, candidates, user_id)

        # Predict relevance scores
        scores = self.model.predict(features)

        # Sort by score
        ranked_indices = np.argsort(scores)[::-1]
        ranked_candidates = [candidates[i] for i in ranked_indices]

        return ranked_candidates

    def extract_features(self, query, candidates, user_id):
        """
        Extract ranking features
        """
        features = []

        for product in candidates:
            feature_dict = {}

            # Query-document features
            feature_dict['bm25_score'] = self.compute_bm25(query, product)
            feature_dict['title_match'] = self.title_match_score(query, product)
            feature_dict['semantic_similarity'] = self.semantic_sim(query, product)

            # Document features
            feature_dict['price'] = product['price']
            feature_dict['avg_rating'] = product['avg_rating']
            feature_dict['num_reviews'] = product['num_reviews']
            feature_dict['sales_rank'] = product['sales_rank']
            feature_dict['in_stock'] = int(product['in_stock'])

            # User-document features
            feature_dict['user_category_affinity'] = self.category_affinity(
                user_id, product['category']
            )
            feature_dict['user_brand_affinity'] = self.brand_affinity(
                user_id, product['brand']
            )
            feature_dict['user_price_affinity'] = self.price_affinity(
                user_id, product['price']
            )

            features.append(list(feature_dict.values()))

        return np.array(features)

# Training learning-to-rank model
class LTRTraining:
    """
    Train L2R model from click data
    """
    def __init__(self):
        self.bigquery = bigquery.Client()

    def prepare_training_data(self):
        """
        Create training data from search logs
        """
        query = """
        SELECT
            s.query,
            s.user_id,
            s.product_id,
            s.position,
            CASE
                WHEN c.clicked = 1 THEN 2  -- Clicked
                WHEN a.added_to_cart = 1 THEN 3  -- Added to cart
                WHEN p.purchased = 1 THEN 4  -- Purchased
                ELSE 0  -- Impression only
            END as relevance_label
        FROM search_impressions s
        LEFT JOIN clicks c USING (query_id, product_id)
        LEFT JOIN add_to_cart a USING (query_id, product_id)
        LEFT JOIN purchases p USING (query_id, product_id)
        WHERE s.date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
        """

        df = self.bigquery.query(query).to_dataframe()

        return df

    def train_ltr_model(self, df):
        """
        Train LightGBM L2R model
        """
        # Extract features
        X = self.extract_features_batch(df)
        y = df['relevance_label']

        # Create query groups (for ranking metric)
        query_groups = df.groupby('query').size().values

        # Train LightGBM ranker
        train_data = lgb.Dataset(X, label=y, group=query_groups)

        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5, 10],
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            verbose_eval=10
        )

        # Save model
        model.save_model('models/ltr_model.txt')

        return model
```

---

## Scaling Considerations

### 1. Horizontal vs Vertical Scaling

**When to use horizontal scaling:**
- Stateless services (model serving, API endpoints)
- Can distribute load across multiple instances
- Need high availability

**When to use vertical scaling:**
- Stateful services (databases, caching)
- Single large model that doesn't fit on smaller instances
- Cost-effective for predictable load

**Example:**

```python
# Horizontal scaling configuration (Kubernetes)
"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-server
  minReplicas: 5
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
"""
```

### 2. Caching Strategies

```python
class MultiLevelCache:
    """
    L1 (in-memory) → L2 (Redis) → L3 (Database/Model)
    """
    def __init__(self):
        self.l1_cache = LRUCache(maxsize=10000)  # In-memory
        self.l2_cache = redis.Redis()  # Redis cluster

    def get(self, key):
        # Try L1
        if key in self.l1_cache:
            return self.l1_cache[key]

        # Try L2
        value = self.l2_cache.get(key)
        if value:
            parsed = json.loads(value)
            self.l1_cache[key] = parsed  # Populate L1
            return parsed

        # Fallback to source (database/model inference)
        value = self.fetch_from_source(key)

        # Populate both caches
        self.l2_cache.setex(key, 3600, json.dumps(value))
        self.l1_cache[key] = value

        return value
```

### 3. Database Sharding

```python
class ShardedDatabase:
    """
    Horizontal database sharding by user_id
    """
    def __init__(self, num_shards=10):
        self.num_shards = num_shards
        self.connections = [
            self.connect_to_shard(i) for i in range(num_shards)
        ]

    def get_shard(self, user_id):
        """
        Consistent hashing to assign user to shard
        """
        shard_id = hash(user_id) % self.num_shards
        return self.connections[shard_id]

    def get_user_data(self, user_id):
        """
        Query appropriate shard
        """
        conn = self.get_shard(user_id)
        return conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
```

### 4. Asynchronous Processing

```python
from celery import Celery

app = Celery('ml_tasks', broker='redis://localhost:6379')

@app.task
def retrain_model():
    """
    Asynchronous model retraining
    """
    # Long-running training job
    model = train_new_model()
    model.save('models/latest.pkl')
    return "Model training complete"

# Trigger asynchronous task
retrain_model.delay()
```

---

## Trade-offs and Decision Making

### Model Complexity vs Latency

| Model Type | Latency (P95) | Accuracy | Use Case |
|------------|---------------|----------|----------|
| Linear/Logistic | <1ms | Low | High QPS, simple patterns |
| Gradient Boosting | 5-10ms | Medium | Structured data, interpretability |
| Neural Network | 20-50ms | High | Complex patterns, sufficient compute |
| Large Transformers | 100ms+ | Very High | NLP, when latency is acceptable |

**Decision Framework:**

```python
def select_model_architecture(requirements):
    """
    Select model based on requirements
    """
    if requirements['latency_p95'] < 10:
        if requirements['data_type'] == 'tabular':
            return 'gradient_boosting'
        else:
            return 'linear_model'

    elif requirements['latency_p95'] < 100:
        if requirements['accuracy'] == 'high':
            return 'neural_network'
        else:
            return 'gradient_boosting'

    else:
        return 'large_transformer'
```

### Batch vs Online Learning

**Batch Learning:**
- Pros: Stable, reproducible, easier to debug
- Cons: Can become stale, requires retraining infrastructure

**Online Learning:**
- Pros: Always fresh, adapts to new patterns
- Cons: Can drift, harder to debug, risk of catastrophic forgetting

**Hybrid Approach (Best):**

```python
class HybridLearning:
    """
    Batch model + online fine-tuning
    """
    def __init__(self):
        self.base_model = load_batch_model()  # Weekly retraining
        self.online_layer = OnlineLayer()      # Continuous updates

    def predict(self, features):
        # Base predictions
        base_pred = self.base_model(features)

        # Online adjustment
        adjusted_pred = self.online_layer.adjust(base_pred, features)

        return adjusted_pred

    def update(self, features, label):
        # Update online layer only
        self.online_layer.partial_fit(features, label)
```

### Feature Engineering vs Deep Learning

**When to use feature engineering (traditional ML):**
- Structured/tabular data
- Limited training data (<100K examples)
- Need interpretability
- Latency-critical (<10ms)

**When to use deep learning:**
- Unstructured data (text, images, audio)
- Large training data (>1M examples)
- Complex patterns
- Sufficient compute budget

---

## Interview Questions

### Question 1: Design a news feed ranking system for a social media platform

**Key Points to Cover:**

1. **Requirements Clarification:**
   - Scale: 500M daily active users
   - Feed update frequency: Real-time
   - Latency: <1 second for feed generation
   - Personalization: Yes
   - Content types: Posts, videos, ads

2. **High-Level Architecture:**

```
User opens app → Candidate Generation (1000 posts) → Ranking Model → Feed
                        ↓                                    ↓
                   Follow graph                        Engagement
                   Trending posts                      prediction
                   User interests
```

3. **Candidate Generation:**
   - Follow graph: Posts from connections
   - Interests: Posts matching user interests
   - Trending: Popular posts
   - Exploration: Random posts for diversity

4. **Ranking Objectives:**
   - Engagement: P(like) + P(comment) + P(share)
   - Time spent: Predicted watch time for videos
   - Recency: Decay factor for old posts
   - Diversity: Avoid echo chambers

5. **Key Challenges:**
   - **Cold start:** New users have no history
     - Solution: Onboarding questions, collaborative filtering
   - **Real-time updates:** New posts should appear quickly
     - Solution: Streaming pipeline (Kafka + Flink)
   - **Diversity:** Avoid filter bubbles
     - Solution: Exploration bonus, diversity re-ranking

6. **Monitoring:**
   - Engagement metrics: Click-through rate, time spent
   - User retention: Daily active users
   - Model performance: Offline AUC, online A/B tests

### Question 2: How would you detect and handle concept drift in production?

**Key Points:**

1. **Types of Drift:**
   - **Covariate drift:** Input distribution changes (P(X) changes)
   - **Prior probability drift:** Label distribution changes (P(Y) changes)
   - **Concept drift:** Relationship changes (P(Y|X) changes)

2. **Detection Methods:**

```python
class DriftDetector:
    def detect_covariate_drift(self, reference_data, current_data):
        """
        Kolmogorov-Smirnov test for distribution shift
        """
        from scipy.stats import ks_2samp

        drift_detected = {}
        for feature in reference_data.columns:
            statistic, p_value = ks_2samp(
                reference_data[feature],
                current_data[feature]
            )
            drift_detected[feature] = p_value < 0.05

        return drift_detected

    def detect_concept_drift(self, model, labeled_data):
        """
        Monitor model performance on recent labeled data
        """
        recent_accuracy = model.score(
            labeled_data['features'],
            labeled_data['labels']
        )

        if recent_accuracy < self.baseline_accuracy - 0.05:
            return True
        return False
```

3. **Mitigation Strategies:**
   - **Continuous retraining:** Retrain on recent data
   - **Ensemble with recency weighting:** Weight recent models higher
   - **Adaptive learning rate:** Increase learning rate when drift detected
   - **Feature monitoring:** Track feature distributions

### Question 3: Explain your approach to A/B testing ML models

**Key Points:**

1. **Experimental Design:**
   - **Randomization:** Random assignment to treatment/control
   - **Sample size calculation:** Based on MDE and power
   - **Duration:** Balance statistical power vs time to decision

2. **Metrics:**
   - **Primary metric:** Business objective (revenue, engagement)
   - **Secondary metrics:** Model performance (AUC, latency)
   - **Guardrail metrics:** User experience (latency, error rate)

3. **Analysis:**

```python
def analyze_ab_test(control, treatment, alpha=0.05):
    """
    Analyze A/B test results
    """
    # T-test
    t_stat, p_value = stats.ttest_ind(control, treatment)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(control) + np.var(treatment)) / 2)
    cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std

    # Confidence interval
    diff = np.mean(treatment) - np.mean(control)
    se = pooled_std * np.sqrt(1/len(control) + 1/len(treatment))
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se

    return {
        'statistically_significant': p_value < alpha,
        'p_value': p_value,
        'effect_size': cohens_d,
        'relative_lift': (diff / np.mean(control)) * 100,
        'confidence_interval_95': (ci_lower, ci_upper)
    }
```

4. **Common Pitfalls:**
   - **Peeking:** Stopping test early based on interim results
     - Solution: Pre-commit to sample size, use sequential testing
   - **Multiple testing:** Testing many variants increases false positives
     - Solution: Bonferroni correction or control FDR
   - **Network effects:** Users in control/treatment interact
     - Solution: Cluster randomization (by geography)

---

## Summary

### Key Takeaways for ML System Design Interviews:

1. **Clarify requirements first** - Don't jump to solutions
2. **Think end-to-end** - Data → Training → Serving → Monitoring
3. **Consider trade-offs** - Accuracy vs latency, complexity vs maintainability
4. **Scale matters** - 1K QPS vs 100K QPS requires different architectures
5. **Monitoring is critical** - Model performance degrades over time
6. **Iterate** - Start simple, add complexity as needed

### Resources:

- **Books:**
  - "Designing Machine Learning Systems" by Chip Huyen
  - "Machine Learning Design Patterns" by Lakshmanan et al.

- **Practice:**
  - Daily LeetCode ML system design problems
  - Mock interviews with peers
  - Read engineering blogs (Netflix, Uber, Airbnb)

- **Tools to Know:**
  - Feature stores: Feast, Tecton
  - Model serving: TensorFlow Serving, TorchServe
  - Monitoring: Prometheus, Grafana, Evidently
  - Orchestration: Airflow, Kubeflow

---

**Version:** 1.0
**Last Updated:** October 2025
**Estimated Reading Time:** 120 minutes