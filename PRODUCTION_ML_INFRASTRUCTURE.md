# Production ML Infrastructure: Complete Guide

**Build production-grade ML infrastructure for real-world systems at scale**

> 🏗️ **Feature Stores** | 📊 **Data Pipelines** | ⚡ **Streaming ML** | 🗄️ **Data Infrastructure** | 🚀 **Production Patterns**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Feature Stores](#feature-stores)
3. [Data Infrastructure](#data-infrastructure)
4. [Data Pipelines](#data-pipelines)
5. [Streaming ML](#streaming-ml)
6. [Production Patterns](#production-patterns)
7. [Best Practices](#best-practices)
8. [Interview Questions](#interview-questions)

---

## 1. Introduction

### What is Production ML Infrastructure?

Production ML infrastructure refers to the systems, tools, and practices that enable ML models to run reliably and efficiently in production environments.

**Key Components:**
1. **Feature Stores** - Centralized feature management
2. **Data Infrastructure** - Scalable data storage and processing
3. **Data Pipelines** - Automated data workflows
4. **Streaming Systems** - Real-time data processing
5. **Serving Infrastructure** - Low-latency model inference
6. **Monitoring Systems** - Track performance and data quality

### Why It Matters

**Without proper infrastructure:**
- ❌ Training-serving skew (different features in training vs production)
- ❌ Manual feature engineering for each model
- ❌ Slow iteration cycles
- ❌ Unreliable predictions
- ❌ Expensive re-computation

**With proper infrastructure:**
- ✅ Consistent features across training and serving
- ✅ Feature reuse across teams
- ✅ Fast experimentation
- ✅ Reliable production systems
- ✅ Cost-efficient operations

---

## 2. Feature Stores

### What is a Feature Store?

A **feature store** is a centralized repository for storing, managing, and serving ML features for both training and inference.

**Key Benefits:**
1. **Consistency** - Same features in training and serving
2. **Reusability** - Share features across teams
3. **Governance** - Track feature lineage and quality
4. **Performance** - Fast feature retrieval
5. **Monitoring** - Detect feature drift

### Feature Store Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Feature Store                           │
│                                                              │
│  ┌────────────────┐        ┌────────────────┐              │
│  │  Offline Store │        │  Online Store  │              │
│  │  (Historical   │◄───────┤  (Low-latency) │              │
│  │   Features)    │        │   Features)    │              │
│  │                │        │                │              │
│  │  - S3/BigQuery │        │  - Redis       │              │
│  │  - Parquet     │        │  - DynamoDB    │              │
│  │  - Delta Lake  │        │  - Cassandra   │              │
│  └────────────────┘        └────────────────┘              │
│         ▲                          ▲                        │
│         │                          │                        │
│  ┌──────┴──────────┐      ┌───────┴────────┐              │
│  │ Batch Features  │      │ Online Features │              │
│  │ (Training)      │      │ (Inference)     │              │
│  └─────────────────┘      └─────────────────┘              │
│                                                              │
│  ┌────────────────────────────────────────┐                │
│  │      Feature Registry (Metadata)        │                │
│  │  - Feature definitions                  │                │
│  │  - Data lineage                         │                │
│  │  - Statistics & monitoring              │                │
│  └────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Feast (Feature Store)

**Feast** is an open-source feature store developed by Tecton.

#### Installation

```bash
pip install feast
```

#### Setup Feast Project

```bash
# Create new Feast project
feast init my_feature_repo
cd my_feature_repo

# Project structure:
# my_feature_repo/
#   feature_store.yaml  # Configuration
#   features.py         # Feature definitions
#   data/               # Sample data
```

#### Define Features

```python
# features.py
from datetime import timedelta
from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource, BigQuerySource
import pandas as pd

# Define entity (primary key)
user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="User identifier"
)

# Define data source
user_stats_source = FileSource(
    path="data/user_stats.parquet",
    event_timestamp_column="event_timestamp",
)

# Alternative: BigQuery source
# user_stats_source = BigQuerySource(
#     table_ref="project.dataset.user_stats",
#     event_timestamp_column="event_timestamp",
# )

# Define feature view
user_stats_fv = FeatureView(
    name="user_statistics",
    entities=["user_id"],
    ttl=timedelta(days=7),
    features=[
        Feature(name="total_purchases", dtype=ValueType.INT64),
        Feature(name="avg_purchase_value", dtype=ValueType.DOUBLE),
        Feature(name="days_since_last_purchase", dtype=ValueType.INT32),
        Feature(name="favorite_category", dtype=ValueType.STRING),
        Feature(name="total_sessions", dtype=ValueType.INT64),
        Feature(name="avg_session_duration", dtype=ValueType.DOUBLE),
    ],
    online=True,  # Enable online serving
    source=user_stats_source,
    tags={"team": "ml", "project": "recommendations"}
)

# Product features
product = Entity(
    name="product_id",
    value_type=ValueType.INT64,
    description="Product identifier"
)

product_features_source = FileSource(
    path="data/product_features.parquet",
    event_timestamp_column="event_timestamp",
)

product_features_fv = FeatureView(
    name="product_features",
    entities=["product_id"],
    ttl=timedelta(days=30),
    features=[
        Feature(name="price", dtype=ValueType.DOUBLE),
        Feature(name="category", dtype=ValueType.STRING),
        Feature(name="brand", dtype=ValueType.STRING),
        Feature(name="rating", dtype=ValueType.DOUBLE),
        Feature(name="num_reviews", dtype=ValueType.INT64),
        Feature(name="in_stock", dtype=ValueType.BOOL),
    ],
    online=True,
    source=product_features_source,
    tags={"team": "ml", "project": "recommendations"}
)
```

#### Apply Feature Definitions

```bash
# Apply to Feast registry
feast apply

# Expected output:
# Created entity user_id
# Created feature view user_statistics
# Created entity product_id
# Created feature view product_features
```

#### Materialize Features (Offline → Online)

```python
from datetime import datetime, timedelta
from feast import FeatureStore

# Initialize feature store
store = FeatureStore(repo_path=".")

# Materialize features from offline to online store
store.materialize(
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now()
)

# This loads historical features into the online store (Redis/DynamoDB)
# for low-latency serving
```

#### Retrieve Features for Training

```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path=".")

# Training data with entity keys and timestamps
training_df = pd.DataFrame({
    'user_id': [1001, 1002, 1003, 1004, 1005],
    'product_id': [5001, 5002, 5003, 5001, 5004],
    'event_timestamp': pd.to_datetime([
        '2025-10-19 10:00:00',
        '2025-10-19 11:00:00',
        '2025-10-19 12:00:00',
        '2025-10-19 13:00:00',
        '2025-10-19 14:00:00',
    ]),
    'label': [1, 0, 1, 0, 1]  # Click/purchase label
})

# Retrieve historical features
training_data = store.get_historical_features(
    entity_df=training_df,
    features=[
        "user_statistics:total_purchases",
        "user_statistics:avg_purchase_value",
        "user_statistics:days_since_last_purchase",
        "product_features:price",
        "product_features:category",
        "product_features:rating",
    ],
).to_df()

print(training_data.head())
# Output:
#    user_id  product_id       event_timestamp  label  total_purchases  avg_purchase_value  ...
# 0     1001        5001  2025-10-19 10:00:00      1               12               45.50  ...
# 1     1002        5002  2025-10-19 11:00:00      0                5               30.20  ...
```

#### Retrieve Features for Inference (Online)

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Online feature retrieval (low-latency)
features = store.get_online_features(
    features=[
        "user_statistics:total_purchases",
        "user_statistics:avg_purchase_value",
        "product_features:price",
        "product_features:rating",
    ],
    entity_rows=[
        {"user_id": 1001, "product_id": 5001},
        {"user_id": 1002, "product_id": 5002},
    ]
).to_dict()

print(features)
# Output (from Redis/DynamoDB, <10ms latency):
# {
#   'user_id': [1001, 1002],
#   'product_id': [5001, 5002],
#   'total_purchases': [12, 5],
#   'avg_purchase_value': [45.50, 30.20],
#   'price': [99.99, 149.99],
#   'rating': [4.5, 4.2]
# }
```

#### Feature Service for Model Serving

```python
# features.py
from feast import FeatureService

# Define feature service for recommendation model
recommendation_features = FeatureService(
    name="recommendation_v1",
    features=[
        user_stats_fv[["total_purchases", "avg_purchase_value", "days_since_last_purchase"]],
        product_features_fv[["price", "category", "rating", "num_reviews"]],
    ],
    tags={"model": "recommendations", "version": "v1"}
)
```

```bash
# Apply feature service
feast apply
```

```python
# Use in serving
features = store.get_online_features(
    feature_service_name="recommendation_v1",
    entity_rows=[{"user_id": 1001, "product_id": 5001}]
).to_dict()
```

### Tecton (Enterprise Feature Store)

**Tecton** is an enterprise feature platform with advanced capabilities.

#### Key Features
- **Real-time feature computation** from streaming data
- **Feature pipelines** with transformations
- **Monitoring and alerting** built-in
- **Multi-cloud support** (AWS, GCP, Azure)
- **Data quality** validation

#### Example Tecton Feature Pipeline

```python
from tecton import Entity, batch_feature_view, Aggregation, FeatureView
from tecton.types import Field, String, Int64, Float64
from datetime import timedelta

# Define entity
user = Entity(
    name='user',
    join_keys=[Field('user_id', Int64)],
    description='User entity'
)

# Batch feature view with aggregations
@batch_feature_view(
    sources=[user_events],
    entities=[user],
    mode='spark_sql',
    aggregation_interval=timedelta(hours=1),
    aggregations=[
        Aggregation(column='purchase_amount', function='sum', time_window=timedelta(days=7)),
        Aggregation(column='purchase_amount', function='mean', time_window=timedelta(days=30)),
        Aggregation(column='event', function='count', time_window=timedelta(days=1)),
    ],
    online=True,
    offline=True,
    feature_start_time=datetime(2024, 1, 1),
    tags={'team': 'ml'}
)
def user_purchase_aggregates(user_events):
    return f'''
        SELECT
            user_id,
            event_timestamp,
            purchase_amount,
            event
        FROM {user_events}
        WHERE event_type = 'purchase'
    '''
```

### AWS Feature Store (SageMaker)

```python
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

# Create feature group
user_feature_group = FeatureGroup(
    name="user-features-v1",
    sagemaker_session=sagemaker.Session()
)

# Define features
user_feature_group.load_feature_definitions(
    data_frame=user_features_df
)

# Create feature group
user_feature_group.create(
    s3_uri=f"s3://my-bucket/feature-store",
    record_identifier_name="user_id",
    event_time_feature_name="event_timestamp",
    role_arn=role,
    enable_online_store=True
)

# Ingest features
user_feature_group.ingest(
    data_frame=user_features_df,
    max_workers=3,
    wait=True
)

# Online retrieval
from sagemaker.feature_store.feature_store import FeatureStore

feature_store = FeatureStore(sagemaker_session=sagemaker.Session())

record = feature_store.get_record(
    record_identifier_value_as_string="1001",
    feature_group_name="user-features-v1"
)
```

### Feature Store Best Practices

**1. Feature Naming Convention**
```python
# Good: descriptive, versioned
user_total_purchases_7d_v1
user_avg_session_duration_30d_v2
product_category_embedding_v1

# Bad: ambiguous
total_purchases
avg_duration
embedding
```

**2. Feature Documentation**
```python
Feature(
    name="user_total_purchases_7d",
    dtype=ValueType.INT64,
    description="Total number of purchases by user in last 7 days",
    owner="ml-team@company.com",
    tags={
        "category": "user_behavior",
        "update_frequency": "hourly",
        "data_source": "purchase_events"
    }
)
```

**3. Feature Monitoring**
```python
# Monitor feature statistics
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Get feature statistics
stats = store.get_feature_statistics(
    features=["user_statistics:total_purchases"],
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now()
)

print(f"Mean: {stats['mean']:.2f}")
print(f"Std: {stats['std']:.2f}")
print(f"Missing: {stats['missing_pct']:.2%}")
```

**4. Feature Versioning**
```python
# Version features when logic changes
user_stats_v1 = FeatureView(
    name="user_statistics_v1",
    # ... old logic
)

user_stats_v2 = FeatureView(
    name="user_statistics_v2",
    # ... new logic with improved calculation
)

# Gradual rollout: serve v2 to 10% of traffic
```

---

## 3. Data Infrastructure

### Data Warehouses vs Data Lakes vs Lakehouses

#### Data Warehouse (e.g., Snowflake, BigQuery, Redshift)

**Purpose:** Structured, queryable storage for analytics

**Characteristics:**
- Schema-on-write (structured data)
- SQL queries
- High performance for aggregations
- Optimized for BI and reporting

**Use Cases:**
- Feature engineering with SQL
- Model training on tabular data
- Analytics and dashboards

```sql
-- BigQuery: Feature engineering for training
SELECT
  user_id,
  COUNT(*) as total_purchases,
  AVG(purchase_amount) as avg_purchase_value,
  DATE_DIFF(CURRENT_DATE(), MAX(purchase_date), DAY) as days_since_last_purchase,
  APPROX_TOP_COUNT(category, 1)[OFFSET(0)].value as favorite_category
FROM purchases
WHERE purchase_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
GROUP BY user_id
```

#### Data Lake (e.g., S3, GCS, Azure Blob)

**Purpose:** Store raw, unstructured data at scale

**Characteristics:**
- Schema-on-read (flexible format)
- Supports all data types (logs, images, videos)
- Cost-effective storage
- Requires processing for queries

**Use Cases:**
- Store raw event logs
- Image/video datasets
- Model artifacts and experiments

```python
# S3 data lake structure
s3://my-ml-datalake/
  ├── raw/
  │   ├── user_events/dt=2025-10-19/hour=10/
  │   ├── images/
  │   └── logs/
  ├── processed/
  │   ├── features/
  │   └── labels/
  └── models/
      └── recommendation_v1/
```

#### Lakehouse (e.g., Delta Lake, Apache Iceberg, Hudi)

**Purpose:** Combine warehouse + lake benefits

**Characteristics:**
- ACID transactions on data lake
- Schema evolution
- Time travel (versioning)
- Unified batch + streaming

**Use Cases:**
- ML feature tables
- Experiment tracking
- Production data versioning

```python
# Delta Lake: ACID transactions on S3
from delta import *

# Write features with versioning
df.write.format("delta").mode("overwrite").save("s3://bucket/features/user_stats")

# Time travel: query historical version
df_v1 = spark.read.format("delta").option("versionAsOf", 1).load("s3://bucket/features/user_stats")
df_v2 = spark.read.format("delta").option("timestampAsOf", "2025-10-19").load("s3://bucket/features/user_stats")

# Schema evolution
df_new.write.format("delta").mode("append").option("mergeSchema", "true").save("s3://bucket/features/user_stats")
```

### Comparison Table

| Feature | Data Warehouse | Data Lake | Lakehouse |
|---------|---------------|-----------|-----------|
| **Data Type** | Structured | All types | All types |
| **Schema** | Schema-on-write | Schema-on-read | Flexible |
| **Query Performance** | Fast (SQL) | Slow (needs processing) | Fast (indexed) |
| **Cost** | High | Low | Medium |
| **ACID Transactions** | ✅ Yes | ❌ No | ✅ Yes |
| **ML Use Case** | Feature engineering | Raw data storage | Features + raw data |
| **Best For** | Tabular features | Unstructured data | ML end-to-end |

### Production Data Infrastructure Example

```
┌─────────────────────────────────────────────────────────────┐
│                  Production Data Flow                        │
└─────────────────────────────────────────────────────────────┘

1. Data Collection
   ├── App/Web → Kafka → S3 (Data Lake - Raw Events)
   └── Databases → CDC → S3 (Data Lake - DB Snapshots)

2. Data Processing
   ├── Batch: Spark → Delta Lake (Features)
   └── Streaming: Flink → Delta Lake (Real-time Features)

3. Feature Store
   ├── Offline: Delta Lake (Training)
   └── Online: Redis (Inference)

4. Data Warehouse
   └── BigQuery ← Delta Lake (Analytics & BI)

5. ML Training
   └── Training Jobs → Read from Feature Store Offline

6. ML Serving
   └── API → Feature Store Online → Model → Prediction
```

---

## 4. Data Pipelines

### Orchestration with Apache Airflow

**Apache Airflow** orchestrates complex data workflows as DAGs (Directed Acyclic Graphs).

#### Installation

```bash
pip install apache-airflow
airflow db init
airflow webserver --port 8080
airflow scheduler
```

#### ML Feature Pipeline DAG

```python
# dags/user_features_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.emr import EmrAddStepsOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['ml-alerts@company.com']
}

dag = DAG(
    'user_features_daily',
    default_args=default_args,
    description='Daily user feature computation',
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=['ml', 'features', 'production']
)

def extract_raw_events(**context):
    """Extract user events from data lake"""
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("ExtractEvents").getOrCreate()

    # Read yesterday's events
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')

    events = spark.read.parquet(f"s3://datalake/raw/user_events/dt={date_str}/")

    # Basic filtering
    events_clean = events.filter(
        (events.user_id.isNotNull()) &
        (events.event_type.isin(['view', 'click', 'purchase']))
    )

    # Save to staging
    events_clean.write.mode('overwrite').parquet(f"s3://datalake/staging/events/{date_str}/")

    return f"Processed {events_clean.count()} events"

def compute_user_features(**context):
    """Compute aggregated user features"""
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder.appName("ComputeFeatures").getOrCreate()

    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')

    # Read last 30 days of events
    events = spark.read.parquet("s3://datalake/staging/events/")

    # Aggregate features
    user_features = events.groupBy('user_id').agg(
        F.count('*').alias('total_events'),
        F.countDistinct(F.when(F.col('event_type') == 'purchase', F.col('event_id'))).alias('total_purchases'),
        F.avg(F.when(F.col('event_type') == 'purchase', F.col('amount'))).alias('avg_purchase_value'),
        F.datediff(F.lit(date_str), F.max(F.when(F.col('event_type') == 'purchase', F.col('event_date')))).alias('days_since_last_purchase'),
        F.collect_list(F.when(F.col('event_type') == 'view', F.col('category'))).alias('viewed_categories')
    )

    # Add timestamp
    user_features = user_features.withColumn('event_timestamp', F.lit(execution_date))

    # Write to Delta Lake
    user_features.write.format('delta').mode('overwrite').save('s3://datalake/features/user_stats/')

    return f"Computed features for {user_features.count()} users"

def materialize_to_online_store(**context):
    """Push features to online store (Redis)"""
    from feast import FeatureStore
    from datetime import datetime

    store = FeatureStore(repo_path="/opt/feast")

    # Materialize latest features to online store
    store.materialize(
        start_date=context['execution_date'] - timedelta(days=1),
        end_date=context['execution_date']
    )

    return "Features materialized to online store"

def validate_features(**context):
    """Validate feature quality"""
    from pyspark.sql import SparkSession
    import great_expectations as ge

    spark = SparkSession.builder.appName("ValidateFeatures").getOrCreate()

    features = spark.read.format('delta').load('s3://datalake/features/user_stats/')

    # Convert to pandas for Great Expectations
    features_pd = features.toPandas()

    # Validate with Great Expectations
    df_ge = ge.from_pandas(features_pd)

    # Expectations
    assert df_ge.expect_column_values_to_not_be_null('user_id')['success']
    assert df_ge.expect_column_values_to_be_between('total_purchases', 0, 10000)['success']
    assert df_ge.expect_column_mean_to_be_between('avg_purchase_value', 1, 1000)['success']

    return "Feature validation passed"

# Define task dependencies
extract_task = PythonOperator(
    task_id='extract_raw_events',
    python_callable=extract_raw_events,
    dag=dag
)

compute_task = PythonOperator(
    task_id='compute_user_features',
    python_callable=compute_user_features,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_features',
    python_callable=validate_features,
    dag=dag
)

materialize_task = PythonOperator(
    task_id='materialize_to_online_store',
    python_callable=materialize_to_online_store,
    dag=dag
)

# Set dependencies
extract_task >> compute_task >> validate_task >> materialize_task
```

### Modern Alternative: Prefect

```python
# Prefect: Modern orchestration
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from datetime import timedelta

@task(retries=3, retry_delay_seconds=60)
def extract_events(date: str):
    """Extract raw events"""
    # ... extraction logic
    return events_df

@task
def compute_features(events_df):
    """Compute features"""
    # ... feature engineering
    return features_df

@task
def validate_features(features_df):
    """Validate quality"""
    # ... validation logic
    return True

@task
def materialize_features(features_df):
    """Push to online store"""
    # ... materialization logic
    return "Success"

@flow(name="user-features-pipeline", task_runner=ConcurrentTaskRunner())
def user_features_flow(date: str):
    """Main feature pipeline"""
    events = extract_events(date)
    features = compute_features(events)
    valid = validate_features(features)

    if valid:
        result = materialize_features(features)
        return result
    else:
        raise ValueError("Feature validation failed")

# Deploy
if __name__ == "__main__":
    user_features_flow.serve(
        name="user-features-deployment",
        cron="0 2 * * *"  # 2 AM daily
    )
```

---

## 5. Streaming ML

### Real-Time Feature Computation with Kafka + Flink

#### Architecture

```
User Events → Kafka → Flink → Feature Store (Online) → Model Serving
```

#### Kafka Producer (Event Collection)

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send user event
event = {
    'user_id': 1001,
    'event_type': 'purchase',
    'product_id': 5001,
    'amount': 99.99,
    'timestamp': '2025-10-19T10:30:00Z'
}

producer.send('user-events', value=event)
producer.flush()
```

#### Flink Streaming Job (Feature Computation)

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream.functions import MapFunction, KeyedProcessFunction
from pyflink.datastream.state import ValueStateDescriptor
import json

env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(4)

# Kafka consumer
kafka_consumer = FlinkKafkaConsumer(
    topics='user-events',
    deserialization_schema=SimpleStringSchema(),
    properties={'bootstrap.servers': 'localhost:9092', 'group.id': 'feature-processor'}
)

# Read stream
event_stream = env.add_source(kafka_consumer)

class UserFeatureAggregator(KeyedProcessFunction):
    """Aggregate user features in real-time"""

    def __init__(self):
        self.total_purchases = None
        self.total_amount = None

    def open(self, runtime_context):
        # Initialize state
        self.total_purchases = runtime_context.get_state(
            ValueStateDescriptor("total_purchases", Types.LONG())
        )
        self.total_amount = runtime_context.get_state(
            ValueStateDescriptor("total_amount", Types.DOUBLE())
        )

    def process_element(self, event, ctx):
        # Parse event
        event_data = json.loads(event)

        if event_data['event_type'] == 'purchase':
            # Update state
            current_purchases = self.total_purchases.value() or 0
            current_amount = self.total_amount.value() or 0.0

            new_purchases = current_purchases + 1
            new_amount = current_amount + event_data['amount']

            self.total_purchases.update(new_purchases)
            self.total_amount.update(new_amount)

            # Emit updated features
            features = {
                'user_id': event_data['user_id'],
                'total_purchases': new_purchases,
                'avg_purchase_value': new_amount / new_purchases,
                'timestamp': event_data['timestamp']
            }

            yield json.dumps(features)

# Apply aggregation
features_stream = event_stream \
    .key_by(lambda x: json.loads(x)['user_id']) \
    .process(UserFeatureAggregator())

# Sink to Kafka (for online store ingestion)
kafka_producer = FlinkKafkaProducer(
    topic='user-features',
    serialization_schema=SimpleStringSchema(),
    producer_config={'bootstrap.servers': 'localhost:9092'}
)

features_stream.add_sink(kafka_producer)

# Execute
env.execute("Real-time User Features")
```

#### Consume Features in Online Store

```python
from kafka import KafkaConsumer
import redis
import json

# Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Kafka consumer
consumer = KafkaConsumer(
    'user-features',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# Consume and update Redis
for message in consumer:
    features = message.value

    # Store in Redis for low-latency lookup
    key = f"user_features:{features['user_id']}"
    redis_client.hmset(key, {
        'total_purchases': features['total_purchases'],
        'avg_purchase_value': features['avg_purchase_value'],
        'last_updated': features['timestamp']
    })

    # Set TTL (7 days)
    redis_client.expire(key, 7 * 24 * 60 * 60)
```

---

## 6. Production Patterns

### Pattern 1: Hybrid Batch + Streaming

```python
"""
Combine batch (historical) and streaming (real-time) features
"""

def get_features_for_inference(user_id, product_id):
    """Retrieve features from multiple sources"""

    # 1. Batch features from Feature Store offline
    batch_features = feature_store.get_online_features(
        features=[
            "user_statistics:total_purchases_30d",  # Computed daily
            "product_features:category",             # Computed weekly
        ],
        entity_rows=[{"user_id": user_id, "product_id": product_id}]
    ).to_dict()

    # 2. Real-time features from Redis
    realtime_features = redis_client.hgetall(f"user_realtime:{user_id}")

    # 3. Context features (request-time)
    context_features = {
        'hour_of_day': datetime.now().hour,
        'day_of_week': datetime.now().weekday(),
        'device_type': request.headers.get('User-Agent')
    }

    # Combine all features
    all_features = {
        **batch_features,
        **realtime_features,
        **context_features
    }

    return all_features
```

### Pattern 2: Feature Serving with Caching

```python
from functools import lru_cache
import hashlib

class FeatureCache:
    """Cache features with TTL"""

    def __init__(self, redis_client, ttl=300):
        self.redis = redis_client
        self.ttl = ttl

    def get_or_compute(self, entity_key, feature_func):
        """Get from cache or compute"""
        cache_key = f"features:{hashlib.md5(str(entity_key).encode()).hexdigest()}"

        # Try cache first
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # Compute if not cached
        features = feature_func(entity_key)

        # Cache for future requests
        self.redis.setex(cache_key, self.ttl, json.dumps(features))

        return features

# Usage
feature_cache = FeatureCache(redis_client, ttl=300)

features = feature_cache.get_or_compute(
    entity_key={'user_id': 1001, 'product_id': 5001},
    feature_func=lambda k: get_features_for_inference(k['user_id'], k['product_id'])
)
```

### Pattern 3: Graceful Degradation

```python
def get_features_with_fallback(user_id, product_id, timeout=100):
    """Get features with fallback if feature store is slow/down"""
    try:
        # Primary: Feature store
        features = feature_store.get_online_features(
            features=["user_statistics:*", "product_features:*"],
            entity_rows=[{"user_id": user_id, "product_id": product_id}],
            timeout=timeout
        ).to_dict()

    except TimeoutError:
        # Fallback 1: Redis cache
        logger.warning("Feature store timeout, using Redis fallback")
        features = redis_client.hgetall(f"features:{user_id}:{product_id}")

    except Exception as e:
        # Fallback 2: Default features
        logger.error(f"Feature retrieval failed: {e}, using defaults")
        features = get_default_features(user_id, product_id)

    return features

def get_default_features(user_id, product_id):
    """Return safe default features for cold start"""
    return {
        'total_purchases': 0,
        'avg_purchase_value': 0.0,
        'days_since_last_purchase': 999,
        'product_price': get_product_price(product_id),  # From product DB
        'product_category': get_product_category(product_id)
    }
```

---

## 7. Best Practices

### 1. Monitoring Feature Quality

```python
from great_expectations import DataContext
from evidently.test_suite import TestSuite
from evidently.tests import *

class FeatureQualityMonitor:
    """Monitor feature drift and quality"""

    def __init__(self, reference_data):
        self.reference_data = reference_data

    def check_data_drift(self, current_data):
        """Detect distribution drift"""
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        drift_detected = report.as_dict()['metrics'][0]['result']['dataset_drift']

        if drift_detected:
            logger.warning("Feature drift detected!")
            # Trigger alert or retraining

        return drift_detected

    def validate_statistics(self, features_df):
        """Validate feature statistics"""
        checks = {
            'total_purchases': {
                'min': 0,
                'max': 10000,
                'mean_range': (5, 50)
            },
            'avg_purchase_value': {
                'min': 0,
                'max': 5000,
                'mean_range': (10, 200)
            }
        }

        for col, rules in checks.items():
            if col in features_df.columns:
                assert features_df[col].min() >= rules['min']
                assert features_df[col].max() <= rules['max']
                mean = features_df[col].mean()
                assert rules['mean_range'][0] <= mean <= rules['mean_range'][1]
```

### 2. Testing Feature Pipelines

```python
import pytest
from datetime import datetime

def test_user_feature_computation():
    """Test feature computation logic"""

    # Sample input
    events = [
        {'user_id': 1, 'event_type': 'purchase', 'amount': 50, 'date': '2025-10-19'},
        {'user_id': 1, 'event_type': 'purchase', 'amount': 100, 'date': '2025-10-20'},
        {'user_id': 1, 'event_type': 'view', 'amount': 0, 'date': '2025-10-21'},
    ]

    # Compute features
    features = compute_user_features(events)

    # Assertions
    assert features['user_id'] == 1
    assert features['total_purchases'] == 2
    assert features['avg_purchase_value'] == 75.0
    assert features['days_since_last_purchase'] >= 0

def test_feature_consistency():
    """Ensure training and serving use same features"""

    # Generate training features
    training_features = get_training_features(user_id=1, product_id=5)

    # Generate serving features
    serving_features = get_serving_features(user_id=1, product_id=5)

    # Ensure same feature names
    assert set(training_features.keys()) == set(serving_features.keys())

    # Ensure same data types
    for key in training_features.keys():
        assert type(training_features[key]) == type(serving_features[key])
```

### 3. Cost Optimization

```python
"""
Optimize feature store costs
"""

# 1. TTL-based expiration
feature_view = FeatureView(
    name="user_features",
    ttl=timedelta(days=7),  # Expire old features
    # ...
)

# 2. Selective materialization (only active users)
active_users = get_active_users(days=30)
for user_batch in chunk(active_users, batch_size=1000):
    feature_store.materialize(
        features=["user_statistics"],
        entity_rows=[{"user_id": uid} for uid in user_batch]
    )

# 3. Compression for offline storage
df.write.format("delta") \
    .option("compression", "zstd") \
    .save("s3://bucket/features/")
```

---

## 8. Interview Questions

### Q1: Explain the architecture of a feature store. Why do we need both online and offline stores?

**Answer:**

A feature store has two main components:

**Offline Store (Historical Features):**
- Purpose: Training and batch analytics
- Storage: Data warehouse (BigQuery, Snowflake) or data lake (S3 + Parquet/Delta)
- Access pattern: Large batch reads
- Latency: Seconds to minutes OK
- Example: Read 1M users × 50 features for model training

**Online Store (Low-Latency Features):**
- Purpose: Real-time inference
- Storage: Key-value store (Redis, DynamoDB, Cassandra)
- Access pattern: Single-row lookups by key
- Latency: <10ms p99
- Example: Get features for user_id=1001 during API call

**Why both?**
1. **Different access patterns** - Training needs bulk reads, inference needs point lookups
2. **Different latency requirements** - Training can wait minutes, inference needs milliseconds
3. **Cost optimization** - Expensive online storage only for active users
4. **Consistency** - Materialize from offline → online ensures same features in training/serving

**Interview Follow-up:**
- "How do you handle feature freshness?" → Materialization schedule, streaming ingestion
- "What if online store is down?" → Fallback to cache or default features
- "How to prevent training-serving skew?" → Same feature computation code, validation tests

---

### Q2: How would you build a real-time recommendation system that needs features from both batch and streaming sources?

**Answer:**

**Architecture:**

```
Batch Pipeline (Daily):
  User Events (S3) → Spark → Feature Store Offline → Redis (Materialization)

Streaming Pipeline (Real-time):
  User Events (Kafka) → Flink → Redis

Serving:
  API Request → Get batch features (Redis) + streaming features (Redis) → Model → Recommendation
```

**Implementation:**

1. **Batch Features** (computed daily):
   - Total purchases last 30 days
   - Favorite categories
   - Average purchase value

2. **Streaming Features** (updated in real-time):
   - Products viewed in current session
   - Click-through rate in last hour
   - Time since last interaction

3. **Serving Logic:**

```python
def get_recommendation_features(user_id, session_id):
    # Batch features (updated daily)
    batch_features = redis.hgetall(f"user_batch:{user_id}")

    # Streaming features (updated real-time)
    streaming_features = redis.hgetall(f"user_streaming:{user_id}:{session_id}")

    # Context features (request-time)
    context = {
        'hour': datetime.now().hour,
        'device': request.device_type
    }

    return {**batch_features, **streaming_features, **context}
```

**Trade-offs:**
- Batch: Computationally expensive but complete historical view
- Streaming: Real-time but limited window (e.g., last hour)
- Hybrid: Best of both, but increased complexity

**Interview Follow-up:**
- "What if streaming pipeline has delay?" → Use batch features as fallback, add staleness check
- "How to handle cold start?" → Default features, content-based filtering
- "Cost optimization?" → Materialize batch features only for active users

---

### Q3: Describe how you would monitor and detect feature drift in production.

**Answer:**

**Monitoring Strategy:**

1. **Statistical Drift Detection:**

```python
from scipy.stats import ks_2samp

def detect_drift(reference_features, current_features, threshold=0.05):
    """Detect drift using Kolmogorov-Smirnov test"""
    drift_detected = {}

    for column in reference_features.columns:
        stat, p_value = ks_2samp(
            reference_features[column],
            current_features[column]
        )

        if p_value < threshold:
            drift_detected[column] = {
                'p_value': p_value,
                'statistic': stat,
                'status': 'DRIFT'
            }

    return drift_detected
```

2. **Monitoring Metrics:**
   - **Distribution metrics:** Mean, std, min, max, quantiles
   - **Missing values:** % null per feature
   - **Cardinality:** # unique values (for categorical)
   - **Correlation changes:** Feature relationships

3. **Alerting:**

```python
# Daily feature quality check
def daily_feature_check(date):
    current = load_features(date)
    reference = load_features(date - timedelta(days=7))

    drift = detect_drift(reference, current)

    if drift:
        send_alert(
            severity='HIGH',
            message=f"Feature drift detected: {drift.keys()}",
            action='Review and potentially retrain model'
        )
```

4. **Dashboard Metrics:**
   - Feature value distributions over time
   - Drift score trends
   - Model performance correlation with drift

**Best Practices:**
- Set drift thresholds based on business impact
- Separate critical vs non-critical features
- Automate retraining when drift detected
- Log all feature values for debugging

**Interview Follow-up:**
- "What causes feature drift?" → Distribution changes, data quality issues, seasonal effects
- "When should you retrain?" → Significant drift + model performance degradation
- "How to prevent false alarms?" → Multi-day smoothing, confidence intervals

---

### Q4: How do you ensure feature consistency between training and serving?

**Answer:**

**Problem:** Training-serving skew causes production bugs and performance degradation.

**Solutions:**

1. **Shared Feature Logic:**

```python
# features.py (shared between training and serving)
def compute_user_features(user_events):
    """Same code for training and serving"""
    return {
        'total_purchases': len([e for e in user_events if e['type'] == 'purchase']),
        'avg_purchase_value': mean([e['amount'] for e in user_events if e['type'] == 'purchase']),
        # ... more features
    }

# Training
training_features = compute_user_features(historical_events)

# Serving
serving_features = compute_user_features(recent_events)
```

2. **Feature Store Enforces Consistency:**
   - Single source of truth for feature definitions
   - Same retrieval API for training and serving
   - Versioned feature schemas

3. **Testing:**

```python
def test_feature_parity():
    """Test that training and serving produce identical features"""

    # Same input
    user_events = load_sample_events()

    # Training path
    training_features = training_pipeline.get_features(user_events)

    # Serving path
    serving_features = serving_api.get_features(user_events)

    # Assert equality
    assert training_features == serving_features
```

4. **Monitoring:**
   - Log feature values during inference
   - Compare inference features to training distributions
   - Alert on unexpected feature values

5. **Deployment Practices:**
   - Version feature transformations with models
   - Gradual rollout with A/B testing
   - Rollback capability if discrepancies found

**Common Pitfalls:**
- Different preprocessing libraries (pandas vs Spark)
- Time zone inconsistencies
- Different data sources (stale cache)
- Missing features in production

**Interview Follow-up:**
- "How to debug training-serving skew?" → Log and compare features, unit tests
- "Feature store benefits?" → Consistency, reusability, governance
- "What if features change?" → Feature versioning, gradual migration

---

## Summary

**Production ML Infrastructure enables:**
- ✅ Fast experimentation with reusable features
- ✅ Consistent training and serving
- ✅ Low-latency inference
- ✅ Scalable data processing
- ✅ Reliable production systems

**Key Technologies:**
- **Feature Stores:** Feast, Tecton, AWS SageMaker Feature Store
- **Data Infrastructure:** BigQuery, Snowflake, Delta Lake, S3
- **Orchestration:** Airflow, Prefect, Dagster
- **Streaming:** Kafka, Flink, Spark Streaming
- **Caching:** Redis, DynamoDB

**Next Steps:**
1. Set up Feast feature store locally
2. Build an Airflow DAG for feature pipelines
3. Implement streaming feature computation with Kafka
4. Deploy a production feature serving API

---

**Generated with Claude Code**

*Last Updated: 2025-10-25*
