# Cloud ML Platforms Guide

**Complete Guide to AWS SageMaker, GCP Vertex AI, and Azure ML**

*Last Updated: October 26, 2025*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Platform Comparison](#platform-comparison)
3. [AWS SageMaker](#aws-sagemaker)
4. [GCP Vertex AI](#gcp-vertex-ai)
5. [Azure Machine Learning](#azure-machine-learning)
6. [Platform Selection Guide](#platform-selection-guide)
7. [Cost Optimization](#cost-optimization)
8. [Best Practices](#best-practices)
9. [Migration Strategies](#migration-strategies)
10. [Interview Questions](#interview-questions)

---

## Introduction

Cloud ML platforms provide end-to-end tools for building, training, and deploying machine learning models at scale. They abstract infrastructure management and provide managed services for common ML workflows.

**Why Use Cloud ML Platforms:**
- **Scalability**: Train on massive datasets with distributed computing
- **Managed Infrastructure**: No need to manage servers, GPUs, or clusters
- **Cost Efficiency**: Pay only for what you use
- **Built-in Tools**: Experiment tracking, model registry, deployment, monitoring
- **Enterprise Features**: Security, compliance, governance

**Major Players:**
1. **AWS SageMaker** - Most mature, largest ecosystem
2. **GCP Vertex AI** - Best for TensorFlow/JAX, tight integration with BigQuery
3. **Azure ML** - Best for Microsoft-heavy organizations

---

## Platform Comparison

### Feature Matrix

| Feature | AWS SageMaker | GCP Vertex AI | Azure ML |
|---------|---------------|---------------|----------|
| **Training** | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| **AutoML** | ✅ Autopilot | ✅ Vertex AI AutoML | ✅ Automated ML |
| **Model Registry** | ✅ Model Registry | ✅ Model Registry | ✅ Model Registry |
| **Deployment** | ✅ Endpoints | ✅ Endpoints | ✅ Endpoints |
| **Monitoring** | ✅ Model Monitor | ✅ Model Monitoring | ✅ Data Drift |
| **Feature Store** | ✅ Feature Store | ✅ Feature Store | ⚠️ Limited |
| **Notebooks** | ✅ Studio | ✅ Workbench | ✅ Notebooks |
| **Distributed Training** | ✅ Multi-GPU/Multi-Node | ✅ Multi-GPU/Multi-Node | ✅ Multi-GPU/Multi-Node |
| **Pipelines** | ✅ Pipelines | ✅ Pipelines | ✅ Pipelines |
| **Batch Transform** | ✅ Yes | ✅ Batch Prediction | ✅ Batch Endpoints |
| **Edge Deployment** | ✅ Edge Manager | ✅ Edge AI | ⚠️ Limited |
| **Pricing Model** | Pay-per-use | Pay-per-use | Pay-per-use |
| **Free Tier** | 2 months | $300 credit | $200 credit |

### Market Share (2025)

- **AWS SageMaker**: ~40% (most widely used)
- **Azure ML**: ~30% (strong in enterprise)
- **GCP Vertex AI**: ~20% (growing fast)
- **Others**: ~10%

### When to Choose Which

**Choose AWS SageMaker if:**
- Already using AWS infrastructure
- Need largest ecosystem and third-party integrations
- Want most mature platform with extensive documentation
- Need comprehensive edge deployment

**Choose GCP Vertex AI if:**
- Using TensorFlow or JAX
- Need tight BigQuery integration for data
- Want best TPU support
- Prefer Google Cloud ecosystem

**Choose Azure ML if:**
- Already using Azure/Microsoft stack
- Need tight integration with Azure services
- Enterprise with Microsoft agreements
- Using .NET or C# for ML

---

## AWS SageMaker

### Overview

Amazon SageMaker is the most comprehensive cloud ML platform, offering end-to-end ML workflows from data prep to deployment.

**Key Components:**
- **SageMaker Studio**: Integrated development environment
- **SageMaker Training**: Distributed training jobs
- **SageMaker Inference**: Real-time and batch predictions
- **SageMaker Pipelines**: ML workflow orchestration
- **SageMaker Feature Store**: Centralized feature management
- **SageMaker Model Monitor**: Track model performance and drift

### 1. Training a Model

**Basic Training Job:**

```python
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker import get_execution_role

# Initialize session
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Define estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',  # Your training script
    role=role,
    instance_type='ml.m5.xlarge',
    framework_version='1.0-1',
    py_version='py3',
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
)

# Start training
sklearn_estimator.fit({'train': 's3://my-bucket/train.csv'})
```

**Training Script (train.py):**

```python
# train.py
import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--random_state', type=int, default=42)
    args, _ = parser.parse_known_args()

    # Load data from SageMaker
    train_data = pd.read_csv('/opt/ml/input/data/train/train.csv')

    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']

    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, os.path.join('/opt/ml/model', 'model.joblib'))

    print(f"Model trained successfully!")
```

### 2. Distributed Training (PyTorch)

```python
from sagemaker.pytorch import PyTorch

# PyTorch distributed training
pytorch_estimator = PyTorch(
    entry_point='train_pytorch.py',
    role=role,
    instance_type='ml.p3.8xlarge',  # 4 GPUs
    instance_count=2,  # 2 instances = 8 GPUs total
    framework_version='2.0.0',
    py_version='py310',
    distribution={
        'pytorchddp': {
            'enabled': True  # Enable DistributedDataParallel
        }
    },
    hyperparameters={
        'epochs': 10,
        'batch-size': 64,
        'learning-rate': 0.001
    }
)

pytorch_estimator.fit({'training': 's3://my-bucket/train/'})
```

**Distributed Training Script:**

```python
# train_pytorch.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train():
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')

    # Create model and wrap in DDP
    model = MyModel().to(device)
    model = DDP(model, device_ids=[device])

    # Training loop
    for epoch in range(args.epochs):
        for batch in train_loader:
            # Training step
            loss = train_step(model, batch)

            if rank == 0:  # Log only from main process
                print(f"Epoch {epoch}, Loss: {loss}")

if __name__ == '__main__':
    train()
```

### 3. Hyperparameter Tuning

```python
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

# Define hyperparameter ranges
hyperparameter_ranges = {
    'n_estimators': IntegerParameter(50, 300),
    'max_depth': IntegerParameter(5, 20),
    'learning_rate': ContinuousParameter(0.001, 0.1, scaling_type='Logarithmic')
}

# Create tuner
tuner = HyperparameterTuner(
    estimator=sklearn_estimator,
    objective_metric_name='validation:accuracy',
    objective_type='Maximize',
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=20,  # Total tuning jobs
    max_parallel_jobs=4  # Parallel jobs
)

# Start tuning
tuner.fit({'train': 's3://my-bucket/train.csv'})

# Get best hyperparameters
best_training_job = tuner.best_training_job()
print(f"Best hyperparameters: {tuner.best_estimator().hyperparameters()}")
```

### 4. Model Deployment

**Real-time Endpoint:**

```python
# Deploy model
predictor = sklearn_estimator.deploy(
    initial_instance_count=2,
    instance_type='ml.m5.xlarge',
    endpoint_name='my-sklearn-endpoint'
)

# Make predictions
import numpy as np
test_data = np.array([[1, 2, 3, 4, 5]])
prediction = predictor.predict(test_data)
print(f"Prediction: {prediction}")

# Auto-scaling configuration
import boto3

client = boto3.client('application-autoscaling')

# Register endpoint as scalable target
client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/my-sklearn-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=2,
    MaxCapacity=10
)

# Define scaling policy
client.put_scaling_policy(
    PolicyName='SageMakerEndpointInvocationScalingPolicy',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/my-sklearn-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,  # Target 70% invocations per instance
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        },
        'ScaleInCooldown': 300,
        'ScaleOutCooldown': 60
    }
)
```

**Batch Transform (Batch Predictions):**

```python
from sagemaker.transformer import Transformer

# Create transformer
transformer = Transformer(
    model_name='my-model',
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path='s3://my-bucket/batch-output/'
)

# Run batch transform
transformer.transform(
    data='s3://my-bucket/batch-input/test.csv',
    content_type='text/csv',
    split_type='Line'
)

# Wait for completion
transformer.wait()
```

### 5. SageMaker Pipelines

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, CreateModelStep, TransformStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterString

# Define pipeline parameters
instance_count = ParameterInteger(name="InstanceCount", default_value=1)
instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")

# Training step
train_step = TrainingStep(
    name="TrainModel",
    estimator=sklearn_estimator,
    inputs={'train': 's3://my-bucket/train.csv'}
)

# Model creation step
create_model_step = CreateModelStep(
    name="CreateModel",
    model=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    inputs=sagemaker.inputs.CreateModelInput(
        instance_type=instance_type
    )
)

# Create pipeline
pipeline = Pipeline(
    name="MyMLPipeline",
    parameters=[instance_count, instance_type],
    steps=[train_step, create_model_step]
)

# Create/update pipeline
pipeline.upsert(role_arn=role)

# Execute pipeline
execution = pipeline.start()
execution.wait()
```

### 6. SageMaker Feature Store

```python
from sagemaker.feature_store.feature_group import FeatureGroup
import time

# Create feature group
feature_group = FeatureGroup(
    name='customer-features',
    sagemaker_session=sagemaker_session
)

# Define features
feature_group.load_feature_definitions(data_frame=customer_df)

# Create feature group
feature_group.create(
    s3_uri=f's3://my-bucket/feature-store/',
    record_identifier_name='customer_id',
    event_time_feature_name='event_time',
    role_arn=role,
    enable_online_store=True
)

# Wait for feature group creation
while feature_group.describe()['FeatureGroupStatus'] == 'Creating':
    time.sleep(5)

# Ingest data
feature_group.ingest(data_frame=customer_df, max_workers=3, wait=True)

# Query features (online store for low-latency)
feature_store_runtime = boto3.client('sagemaker-featurestore-runtime')

response = feature_store_runtime.get_record(
    FeatureGroupName='customer-features',
    RecordIdentifierValueAsString='customer_123'
)

features = {item['FeatureName']: item['ValueAsString']
            for item in response['Record']}
```

### 7. Model Monitoring

```python
from sagemaker.model_monitor import DefaultModelMonitor, DataCaptureConfig

# Enable data capture
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,  # Capture 100% of data
    destination_s3_uri='s3://my-bucket/data-capture'
)

# Deploy with data capture
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='monitored-endpoint',
    data_capture_config=data_capture_config
)

# Create baseline from training data
from sagemaker.model_monitor import DefaultModelMonitor

monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    max_runtime_in_seconds=3600
)

# Generate baseline statistics
baseline_job = monitor.suggest_baseline(
    baseline_dataset='s3://my-bucket/train/train.csv',
    dataset_format={'csv': {'header': True}},
    output_s3_uri='s3://my-bucket/baseline'
)

baseline_job.wait()

# Create monitoring schedule
from sagemaker.model_monitor import CronExpressionGenerator

monitor.create_monitoring_schedule(
    endpoint_input=predictor.endpoint_name,
    schedule_cron_expression=CronExpressionGenerator.hourly(),  # Run hourly
    statistics=monitor.baseline_statistics(),
    constraints=monitor.suggested_constraints(),
    output_s3_uri='s3://my-bucket/monitoring-reports'
)
```

---

## GCP Vertex AI

### Overview

Google Cloud's Vertex AI provides a unified ML platform optimized for TensorFlow and JAX, with excellent integration with BigQuery.

**Key Components:**
- **Vertex AI Workbench**: Managed Jupyter notebooks
- **Vertex AI Training**: Distributed training with TPUs/GPUs
- **Vertex AI Prediction**: Online and batch prediction
- **Vertex AI Pipelines**: Kubeflow-based workflows
- **Vertex AI Feature Store**: Centralized feature repository
- **Vertex AI Model Monitoring**: Track drift and performance

### 1. Training a Model

**Basic Training:**

```python
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(
    project='my-project',
    location='us-central1',
    staging_bucket='gs://my-bucket'
)

# Custom training job
job = aiplatform.CustomTrainingJob(
    display_name='sklearn-training',
    script_path='train.py',
    container_uri='gcr.io/cloud-aiplatform/training/scikit-learn-cpu.1-0:latest',
    requirements=['scikit-learn==1.3.0', 'pandas==2.0.0'],
    model_serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/sklearn-cpu.1-0:latest'
)

# Run training
model = job.run(
    dataset=dataset,
    model_display_name='my-sklearn-model',
    machine_type='n1-standard-4',
    replica_count=1,
    args=['--n-estimators', '100', '--max-depth', '10']
)
```

**Training Script for Vertex AI:**

```python
# train.py
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from google.cloud import storage

def train_model(args):
    # Load data from GCS
    df = pd.read_csv(f'gs://{args.bucket}/train.csv')

    X = df.drop('target', axis=1)
    y = df['target']

    # Train model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    model.fit(X, y)

    # Save model locally
    joblib.dump(model, '/tmp/model.joblib')

    # Upload to GCS (Vertex AI Model Registry)
    storage_client = storage.Client()
    bucket = storage_client.bucket(args.bucket)
    blob = bucket.blob('models/model.joblib')
    blob.upload_from_filename('/tmp/model.joblib')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    args = parser.parse_args()

    train_model(args)
```

### 2. Distributed Training with TensorFlow

```python
# Distributed TensorFlow training
from google.cloud import aiplatform

job = aiplatform.CustomTrainingJob(
    display_name='distributed-tf-training',
    script_path='train_tf.py',
    container_uri='gcr.io/cloud-aiplatform/training/tf-gpu.2-12:latest',
    requirements=['tensorflow==2.12.0']
)

# Multi-GPU training
model = job.run(
    replica_count=1,
    machine_type='n1-standard-16',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=4,  # 4 GPUs
    args=['--epochs', '10', '--batch-size', '64']
)
```

**TensorFlow Distributed Training:**

```python
# train_tf.py
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

with strategy.scope():
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Train
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Save model
model.save('gs://my-bucket/models/tf-model')
```

### 3. Hyperparameter Tuning

```python
from google.cloud import aiplatform

# Define hyperparameter tuning job
job = aiplatform.HyperparameterTuningJob(
    display_name='sklearn-hp-tuning',
    custom_job=custom_job,
    metric_spec={
        'accuracy': 'maximize'
    },
    parameter_spec={
        'n_estimators': aiplatform.hyperparameter_tuning.IntegerParameterSpec(
            min=50, max=300, scale='linear'
        ),
        'max_depth': aiplatform.hyperparameter_tuning.IntegerParameterSpec(
            min=5, max=20, scale='linear'
        ),
        'learning_rate': aiplatform.hyperparameter_tuning.DoubleParameterSpec(
            min=0.001, max=0.1, scale='log'
        )
    },
    max_trial_count=20,
    parallel_trial_count=4
)

# Run tuning
job.run()

# Get best trial
best_trial = job.trials[0]  # Sorted by objective metric
print(f"Best hyperparameters: {best_trial.parameters}")
print(f"Best accuracy: {best_trial.final_measurement.metrics[0].value}")
```

### 4. Model Deployment

**Online Prediction Endpoint:**

```python
# Upload model
model = aiplatform.Model.upload(
    display_name='my-model',
    artifact_uri='gs://my-bucket/models/',
    serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/sklearn-cpu.1-0:latest'
)

# Deploy to endpoint
endpoint = model.deploy(
    deployed_model_display_name='sklearn-v1',
    machine_type='n1-standard-4',
    min_replica_count=2,
    max_replica_count=10,
    traffic_split={"0": 100},  # 100% traffic to this model
    accelerator_type=None  # CPU only
)

# Make predictions
instances = [[1, 2, 3, 4, 5]]
predictions = endpoint.predict(instances=instances)
print(predictions)

# A/B testing: Deploy new model version
new_model.deploy(
    endpoint=endpoint,
    deployed_model_display_name='sklearn-v2',
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=5,
    traffic_split={"0": 90, "1": 10}  # 90% v1, 10% v2
)
```

**Batch Prediction:**

```python
# Batch prediction job
batch_prediction_job = model.batch_predict(
    job_display_name='batch-prediction-job',
    gcs_source=['gs://my-bucket/batch-input/*.csv'],
    gcs_destination_prefix='gs://my-bucket/batch-output/',
    machine_type='n1-standard-4',
    starting_replica_count=1,
    max_replica_count=5
)

# Wait for completion
batch_prediction_job.wait()
```

### 5. Vertex AI Pipelines

```python
from kfp.v2 import dsl
from kfp.v2.dsl import component, pipeline

# Define components
@component(packages_to_install=['pandas', 'scikit-learn'])
def preprocess_data(input_path: str, output_path: str):
    import pandas as pd
    df = pd.read_csv(input_path)
    # Preprocessing logic
    df_processed = df.dropna()
    df_processed.to_csv(output_path, index=False)

@component(packages_to_install=['scikit-learn'])
def train_model(train_data: str, model_output: str):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    df = pd.read_csv(train_data)
    X = df.drop('target', axis=1)
    y = df['target']

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    joblib.dump(model, model_output)

# Define pipeline
@pipeline(name='ml-training-pipeline')
def ml_pipeline(input_data: str):
    preprocess_task = preprocess_data(
        input_path=input_data,
        output_path='gs://my-bucket/processed/data.csv'
    )

    train_task = train_model(
        train_data=preprocess_task.outputs['output_path'],
        model_output='gs://my-bucket/models/model.joblib'
    )

# Compile pipeline
from kfp.v2 import compiler

compiler.Compiler().compile(
    pipeline_func=ml_pipeline,
    package_path='ml_pipeline.json'
)

# Run pipeline
from google.cloud import aiplatform

job = aiplatform.PipelineJob(
    display_name='ml-training-pipeline',
    template_path='ml_pipeline.json',
    parameter_values={'input_data': 'gs://my-bucket/raw/data.csv'}
)

job.run()
```

### 6. Vertex AI Feature Store

```python
# Create feature store
from google.cloud import aiplatform

featurestore = aiplatform.Featurestore.create(
    featurestore_id='customer_featurestore',
    online_store_fixed_node_count=1
)

# Create entity type
entity_type = featurestore.create_entity_type(
    entity_type_id='customer',
    description='Customer entity'
)

# Create features
entity_type.create_feature(
    feature_id='age',
    value_type='INT64'
)

entity_type.create_feature(
    feature_id='total_purchases',
    value_type='INT64'
)

# Ingest features from BigQuery
from google.cloud import bigquery

feature_import_job = entity_type.batch_create_features(
    feature_configs={
        'age': {'value_type': 'INT64'},
        'total_purchases': {'value_type': 'INT64'},
        'avg_purchase_value': {'value_type': 'DOUBLE'}
    }
)

# Serve features (online)
feature_values = entity_type.read(
    entity_ids=['customer_123'],
    feature_ids=['age', 'total_purchases']
)
```

---

## Azure Machine Learning

### Overview

Azure ML provides comprehensive ML lifecycle management with strong enterprise features and Microsoft ecosystem integration.

**Key Components:**
- **Azure ML Studio**: Web-based ML environment
- **Azure ML Compute**: Managed compute for training/inference
- **Azure ML Pipelines**: Workflow orchestration
- **Azure ML Model Registry**: Centralized model management
- **Azure ML Endpoints**: Real-time and batch inference

### 1. Training a Model

**Basic Training:**

```python
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget, AmlCompute

# Connect to workspace
ws = Workspace.from_config()

# Create/get compute cluster
compute_name = 'cpu-cluster'
if compute_name not in ws.compute_targets:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_D2_V2',
        max_nodes=4,
        idle_seconds_before_scaledown=300
    )
    compute_target = ComputeTarget.create(ws, compute_name, compute_config)
    compute_target.wait_for_completion(show_output=True)
else:
    compute_target = ws.compute_targets[compute_name]

# Define environment
env = Environment.from_pip_requirements(
    name='sklearn-env',
    file_path='requirements.txt'
)

# Configure training run
config = ScriptRunConfig(
    source_directory='./src',
    script='train.py',
    compute_target=compute_target,
    environment=env,
    arguments=[
        '--n-estimators', 100,
        '--max-depth', 10
    ]
)

# Submit experiment
experiment = Experiment(workspace=ws, name='sklearn-experiment')
run = experiment.submit(config)
run.wait_for_completion(show_output=True)
```

**Training Script:**

```python
# train.py
import argparse
from azureml.core import Run
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Get run context
run = Run.get_context()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n-estimators', type=int, default=100)
parser.add_argument('--max-depth', type=int, default=10)
args = parser.parse_args()

# Load data
df = pd.read_csv('./data/train.csv')
X = df.drop('target', axis=1)
y = df['target']

# Train model
model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth
)
model.fit(X, y)

# Calculate metrics
accuracy = accuracy_score(y, model.predict(X))

# Log metrics to Azure ML
run.log('accuracy', accuracy)
run.log('n_estimators', args.n_estimators)
run.log('max_depth', args.max_depth)

# Save model
os.makedirs('./outputs', exist_ok=True)
joblib.dump(model, './outputs/model.joblib')
```

### 2. Distributed Training (PyTorch)

```python
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import PyTorchConfiguration

# Distributed training configuration
distributed_config = PyTorchConfiguration(
    node_count=2,  # 2 nodes
    process_count_per_node=4  # 4 GPUs per node = 8 GPUs total
)

# Configure training
config = ScriptRunConfig(
    source_directory='./src',
    script='train_distributed.py',
    compute_target=gpu_cluster,
    environment=pytorch_env,
    distributed_job_config=distributed_config,
    arguments=['--epochs', 10, '--batch-size', 64]
)

# Submit
run = experiment.submit(config)
```

### 3. Hyperparameter Tuning

```python
from azureml.train.hyperdrive import RandomParameterSampling, BanditPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice, uniform, loguniform

# Define parameter space
param_sampling = RandomParameterSampling({
    'n_estimators': choice(50, 100, 200, 300),
    'max_depth': choice(5, 10, 15, 20),
    'learning_rate': loguniform(-4, -1)  # 10^-4 to 10^-1
})

# Early termination policy
early_termination_policy = BanditPolicy(
    slack_factor=0.1,
    evaluation_interval=1,
    delay_evaluation=5
)

# HyperDrive configuration
hyperdrive_config = HyperDriveConfig(
    run_config=config,
    hyperparameter_sampling=param_sampling,
    policy=early_termination_policy,
    primary_metric_name='accuracy',
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=20,
    max_concurrent_runs=4
)

# Run hyperparameter tuning
hyperdrive_run = experiment.submit(hyperdrive_config)
hyperdrive_run.wait_for_completion(show_output=True)

# Get best run
best_run = hyperdrive_run.get_best_run_by_primary_metric()
print(f"Best accuracy: {best_run.get_metrics()['accuracy']}")
print(f"Best parameters: {best_run.get_details()['runDefinition']['arguments']}")
```

### 4. Model Deployment

**Real-time Endpoint:**

```python
from azureml.core.model import Model, InferenceConfig
from azureml.core.webservice import AciWebservice

# Register model
model = Model.register(
    workspace=ws,
    model_path='./outputs/model.joblib',
    model_name='sklearn-model'
)

# Create inference configuration
inference_config = InferenceConfig(
    entry_script='score.py',
    environment=env
)

# Deployment configuration
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    auth_enabled=True
)

# Deploy
service = Model.deploy(
    workspace=ws,
    name='sklearn-service',
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)

service.wait_for_deployment(show_output=True)

# Test endpoint
import json
test_data = json.dumps({'data': [[1, 2, 3, 4, 5]]})
prediction = service.run(test_data)
print(prediction)
```

**Scoring Script (score.py):**

```python
# score.py
import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('sklearn-model')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data)
        predictions = model.predict(data)
        return predictions.tolist()
    except Exception as e:
        return json.dumps({"error": str(e)})
```

### 5. Azure ML Pipelines

```python
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

# Define pipeline data
processed_data = PipelineData('processed', datastore=ws.get_default_datastore())
model_data = PipelineData('model', datastore=ws.get_default_datastore())

# Preprocessing step
preprocess_step = PythonScriptStep(
    name='preprocess',
    script_name='preprocess.py',
    outputs=[processed_data],
    compute_target=compute_target,
    source_directory='./src'
)

# Training step
train_step = PythonScriptStep(
    name='train',
    script_name='train.py',
    inputs=[processed_data],
    outputs=[model_data],
    compute_target=compute_target,
    source_directory='./src'
)

# Create pipeline
pipeline = Pipeline(workspace=ws, steps=[preprocess_step, train_step])

# Publish pipeline
published_pipeline = pipeline.publish(
    name='ML Training Pipeline',
    description='End-to-end ML training pipeline'
)

# Run pipeline
pipeline_run = published_pipeline.submit(ws, experiment_name='pipeline-run')
```

---

## Platform Selection Guide

### Decision Tree

```
Start Here
├─ Already using a cloud provider?
│  ├─ AWS → Use SageMaker
│  ├─ GCP → Use Vertex AI
│  └─ Azure → Use Azure ML
│
├─ Framework preference?
│  ├─ TensorFlow/JAX → Vertex AI (best TPU support)
│  ├─ PyTorch → Any (all support well)
│  └─ Scikit-learn → Any (all support well)
│
├─ Data location?
│  ├─ BigQuery → Vertex AI (seamless integration)
│  ├─ Redshift → SageMaker (seamless integration)
│  └─ Synapse → Azure ML (seamless integration)
│
├─ Budget?
│  ├─ Startup/small → Vertex AI (good free tier)
│  ├─ Enterprise → Azure ML (Microsoft agreements)
│  └─ Flexible → SageMaker (most options)
│
└─ Team expertise?
   ├─ Kubernetes experience → Any (all support)
   ├─ .NET developers → Azure ML
   └─ Python-first → Any
```

### Use Case Recommendations

| Use Case | Best Platform | Reason |
|----------|---------------|--------|
| Computer Vision | Vertex AI | Best TPU support, AutoML Vision |
| NLP/LLMs | SageMaker or Vertex AI | SageMaker: largest model zoo; Vertex AI: best for TF |
| Tabular Data | Any | All platforms handle well |
| Time Series | SageMaker | Built-in DeepAR, Forecast |
| Edge Deployment | SageMaker | Most mature Edge Manager |
| AutoML | Vertex AI | Best AutoML capabilities |
| MLOps/Pipelines | SageMaker | Most mature pipeline features |
| Enterprise Governance | Azure ML | Best enterprise features |

---

## Cost Optimization

### General Strategies

**1. Use Spot/Preemptible Instances:**

```python
# AWS SageMaker - Managed Spot Training
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.xlarge',
    use_spot_instances=True,  # Up to 90% cost savings
    max_wait=7200,  # Max wait for spot instances (seconds)
    max_run=3600  # Max training time
)

# GCP Vertex AI - Preemptible
job.run(
    machine_type='n1-standard-4',
    replica_count=1,
    reduction_server_replica_count=0,
    args=['--epochs', '10'],
    base_output_dir='gs://my-bucket/output',
    scheduling={
        'use_preemptible': True  # Up to 80% cost savings
    }
)

# Azure ML - Low Priority VMs
from azureml.core.compute import AmlCompute

compute_config = AmlCompute.provisioning_configuration(
    vm_size='STANDARD_D2_V2',
    max_nodes=4,
    vm_priority='lowpriority'  # Up to 80% cost savings
)
```

**2. Auto-scaling and Right-sizing:**

- Start with smaller instances, scale up if needed
- Use auto-scaling for endpoints (pay only for what you use)
- Shut down development environments when not in use

**3. Use Batch Processing When Possible:**

- Batch predictions are 10-20x cheaper than real-time
- Use serverless batch inference for sporadic workloads

**4. Optimize Data Storage:**

- Use lifecycle policies to move old data to cheaper storage
- Compress datasets before uploading
- Use columnar formats (Parquet) for better compression

**5. Monitor and Alert on Costs:**

```python
# AWS - Set up budget alerts
import boto3

budgets = boto3.client('budgets')

budgets.create_budget(
    AccountId='123456789012',
    Budget={
        'BudgetName': 'ML-Monthly-Budget',
        'BudgetLimit': {
            'Amount': '1000',
            'Unit': 'USD'
        },
        'TimeUnit': 'MONTHLY',
        'BudgetType': 'COST'
    },
    NotificationsWithSubscribers=[
        {
            'Notification': {
                'NotificationType': 'ACTUAL',
                'ComparisonOperator': 'GREATER_THAN',
                'Threshold': 80.0
            },
            'Subscribers': [
                {'SubscriptionType': 'EMAIL', 'Address': 'team@company.com'}
            ]
        }
    ]
)
```

### Cost Comparison (Approximate, 2025)

**Training (per hour):**

| Instance Type | AWS SageMaker | GCP Vertex AI | Azure ML |
|---------------|---------------|---------------|----------|
| 4 CPU, 16 GB | $0.23 | $0.22 | $0.24 |
| 1x V100 GPU | $3.06 | $2.48 | $3.06 |
| 4x V100 GPU | $12.24 | $9.92 | $12.24 |
| TPU v3-8 | N/A | $8.00 | N/A |

**Inference (per hour):**

| Instance Type | AWS | GCP | Azure |
|---------------|-----|-----|-------|
| 2 CPU, 4 GB | $0.05 | $0.04 | $0.05 |
| 1x T4 GPU | $0.50 | $0.35 | $0.50 |

*Prices vary by region and change frequently. Check official pricing.*

---

## Best Practices

### 1. Version Everything

```python
# Tag models with versions
model_name = f'my-model-{version}'
training_job_name = f'training-{version}-{timestamp}'

# Use Git SHA for reproducibility
import subprocess
git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

# Log to experiment tracking
mlflow.log_param('git_sha', git_sha)
mlflow.log_param('model_version', version)
```

### 2. Use Managed Services When Possible

- Managed notebooks instead of EC2/VM instances
- Managed training instead of custom clusters
- Managed endpoints instead of manual deployment

### 3. Implement Proper Logging

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Log to cloud logging
logger.info(f"Starting training with {len(X_train)} samples")
logger.info(f"Hyperparameters: {hyperparameters}")
logger.info(f"Final accuracy: {accuracy:.4f}")
```

### 4. Use Infrastructure as Code

```yaml
# terraform/sagemaker.tf
resource "aws_sagemaker_model" "model" {
  name               = "my-model"
  execution_role_arn = aws_iam_role.sagemaker.arn

  primary_container {
    image          = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-model:latest"
    model_data_url = "s3://my-bucket/models/model.tar.gz"
  }
}

resource "aws_sagemaker_endpoint_configuration" "config" {
  name = "my-endpoint-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.model.name
    initial_instance_count = 2
    instance_type          = "ml.m5.xlarge"
  }
}
```

### 5. Implement CI/CD for ML

```yaml
# .github/workflows/ml-deploy.yml
name: ML Model Deployment

on:
  push:
    branches: [main]

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: Train model
      run: python scripts/train.py

    - name: Run tests
      run: pytest tests/

    - name: Deploy to SageMaker
      run: python scripts/deploy.py
```

---

## Migration Strategies

### Migrating Between Platforms

**General Approach:**

1. **Assess Current State:**
   - Catalog all models, pipelines, endpoints
   - Document dependencies and integrations
   - Identify business-critical vs experimental workloads

2. **Plan Migration:**
   - Prioritize workloads (start with less critical)
   - Choose migration strategy (lift-and-shift vs re-architecture)
   - Set success criteria

3. **Execute Migration:**
   - Parallel run (old and new platforms simultaneously)
   - Validate results match
   - Gradually shift traffic

4. **Optimize:**
   - Leverage new platform features
   - Optimize costs
   - Improve performance

**Example: AWS SageMaker → GCP Vertex AI**

```python
# 1. Export model from SageMaker
import boto3
s3 = boto3.client('s3')
s3.download_file('my-bucket', 'models/model.tar.gz', 'model.tar.gz')

# 2. Extract and convert if needed
import tarfile
tar = tarfile.open('model.tar.gz')
tar.extractall('model/')

# 3. Upload to GCS
from google.cloud import storage
storage_client = storage.Client()
bucket = storage_client.bucket('my-gcp-bucket')
blob = bucket.blob('models/model.pkl')
blob.upload_from_filename('model/model.pkl')

# 4. Deploy to Vertex AI
from google.cloud import aiplatform

model = aiplatform.Model.upload(
    display_name='migrated-model',
    artifact_uri='gs://my-gcp-bucket/models/',
    serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/sklearn-cpu.1-0:latest'
)

endpoint = model.deploy(machine_type='n1-standard-4')
```

---

## Interview Questions

### Q1: What are the key differences between AWS SageMaker, GCP Vertex AI, and Azure ML?

**Answer:**

**AWS SageMaker:**
- **Strengths:** Most mature, largest ecosystem, most features, best documentation
- **Best for:** Organizations already on AWS, need comprehensive tooling
- **Unique features:** SageMaker Neo (edge optimization), SageMaker Clarify (bias detection)

**GCP Vertex AI:**
- **Strengths:** Best for TensorFlow/JAX, excellent TPU support, tight BigQuery integration
- **Best for:** TensorFlow users, need TPUs, have data in BigQuery
- **Unique features:** AutoML Tables, Vertex AI Matching Engine (vector search)

**Azure ML:**
- **Strengths:** Best enterprise features, tight Microsoft integration, strong governance
- **Best for:** Microsoft-heavy organizations, need enterprise governance
- **Unique features:** Responsible AI dashboard, tight integration with Azure Synapse

**When to choose:**
- **SageMaker:** Default choice for AWS users or need most features
- **Vertex AI:** Using TensorFlow, need TPUs, or data in BigQuery
- **Azure ML:** Already on Azure or need enterprise features

---

### Q2: How do you optimize costs when training ML models on cloud platforms?

**Answer:**

**Key Strategies:**

1. **Use Spot/Preemptible Instances (60-90% savings):**
   - Enable spot instances for fault-tolerant training jobs
   - Implement checkpointing to resume from interruptions
   - Use for experimentation, not production-critical training

2. **Right-size Compute Resources:**
   - Start small, scale up if needed
   - Profile GPU utilization (aim for >70%)
   - Use CPU for small models, GPU only when necessary

3. **Batch Processing:**
   - Use batch predictions instead of real-time endpoints (10-20x cheaper)
   - Schedule batch jobs during off-peak hours

4. **Auto-scaling:**
   - Configure auto-scaling for endpoints (scale to zero when idle)
   - Set appropriate min/max replicas

5. **Lifecycle Management:**
   - Delete unused endpoints, notebooks, compute clusters
   - Set up automatic shutdown for idle resources
   - Use lifecycle policies for storage (move to cheaper tiers)

6. **Distributed Training:**
   - Use data parallelism to reduce training time
   - Fewer hours at higher cost can be cheaper overall

**Example Cost Savings:**
- Spot instances: 60-90% savings
- Auto-scaling: 40-60% savings (vs always-on)
- Batch vs real-time: 90% savings
- Right-sizing: 30-50% savings

**Code Example:**
```python
# Enable spot training in SageMaker
estimator = PyTorch(
    entry_point='train.py',
    use_spot_instances=True,  # 60-90% cost savings
    max_wait=7200,
    max_run=3600,
    checkpoint_s3_uri='s3://my-bucket/checkpoints/'  # Resume from interruptions
)
```

---

### Q3: How would you deploy a model to production with A/B testing on a cloud platform?

**Answer:**

**Approach:**

1. **Deploy Multiple Model Versions:**
   - Deploy baseline model (v1) and new model (v2)
   - Configure traffic split (e.g., 90% v1, 10% v2)

2. **Monitor Metrics:**
   - Track business metrics (conversion, revenue)
   - Track model metrics (accuracy, latency)
   - Log predictions from both models

3. **Statistical Analysis:**
   - Run for sufficient duration (typically 1-2 weeks)
   - Calculate statistical significance
   - Decide whether to promote v2

4. **Gradual Rollout:**
   - If v2 wins: 10% → 25% → 50% → 100%
   - Monitor at each stage
   - Rollback if issues detected

**AWS SageMaker Example:**
```python
# Deploy model v1
predictor_v1 = model_v1.deploy(
    initial_instance_count=2,
    instance_type='ml.m5.xlarge',
    endpoint_name='my-endpoint'
)

# Deploy model v2 to same endpoint
predictor_v2 = model_v2.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='my-endpoint',
    update_endpoint=True
)

# Configure traffic split (90% v1, 10% v2)
import boto3
sm_client = boto3.client('sagemaker')

sm_client.update_endpoint_weights_and_capacities(
    EndpointName='my-endpoint',
    DesiredWeightsAndCapacities=[
        {'VariantName': 'variant-1', 'DesiredWeight': 90},
        {'VariantName': 'variant-2', 'DesiredWeight': 10}
    ]
)

# After analysis: shift to 100% v2
sm_client.update_endpoint_weights_and_capacities(
    EndpointName='my-endpoint',
    DesiredWeightsAndCapacities=[
        {'VariantName': 'variant-1', 'DesiredWeight': 0},
        {'VariantName': 'variant-2', 'DesiredWeight': 100}
    ]
)
```

**GCP Vertex AI Example:**
```python
# Deploy v2 with traffic split
new_model.deploy(
    endpoint=existing_endpoint,
    deployed_model_display_name='model-v2',
    machine_type='n1-standard-4',
    min_replica_count=1,
    traffic_split={
        'model-v1': 90,
        'model-v2': 10
    }
)

# Promote to 100% after validation
endpoint.update(
    traffic_split={
        'model-v1': 0,
        'model-v2': 100
    }
)
```

---

### Q4: What are the trade-offs between managed ML platforms and building custom infrastructure?

**Answer:**

**Managed Platforms (SageMaker, Vertex AI, Azure ML):**

**Pros:**
- Faster time to market (no infrastructure setup)
- Built-in best practices (monitoring, versioning)
- Automatic scaling and high availability
- Integrated tools (experiment tracking, pipelines)
- Lower operational overhead

**Cons:**
- Vendor lock-in
- Less flexibility/customization
- Potentially higher costs at scale
- Learning curve for platform-specific APIs

**Custom Infrastructure (Kubernetes + MLflow + Custom):**

**Pros:**
- Full control and customization
- Cloud-agnostic (portable)
- Potentially lower costs at very large scale
- Can optimize for specific use cases

**Cons:**
- High upfront cost (eng time)
- Requires DevOps/ML expertise
- Longer time to market
- Ongoing maintenance burden
- Need to build many features from scratch

**Decision Framework:**

**Use Managed Platform if:**
- Small-to-medium team (<20 ML engineers)
- Standard ML use cases
- Want to focus on models, not infrastructure
- Don't have dedicated MLOps team
- Need to move fast

**Use Custom Infrastructure if:**
- Very large scale (1000s of models)
- Unique requirements not met by platforms
- Have dedicated MLOps team (>5 people)
- Multi-cloud strategy required
- Long-term cost optimization critical

**Hybrid Approach (Common):**
- Use managed platform for most workflows
- Custom infrastructure for specific needs (e.g., specialized hardware)
- Best of both worlds, but adds complexity

---

### Q5: How do you handle model versioning and rollback in production on cloud platforms?

**Answer:**

**Model Versioning Strategy:**

1. **Semantic Versioning:**
   - Format: `v{major}.{minor}.{patch}`
   - Major: Breaking changes
   - Minor: New features
   - Patch: Bug fixes

2. **Tag with Metadata:**
   - Git commit SHA
   - Training data version
   - Hyperparameters
   - Performance metrics

3. **Use Model Registry:**
   - All platforms provide model registry
   - Register all models, even experimental
   - Tag with stage (dev, staging, production)

**Rollback Strategy:**

1. **Keep Previous Versions:**
   - Never delete production models
   - Keep at least 3 previous versions

2. **Blue-Green Deployment:**
   - Keep old endpoint (blue) while deploying new (green)
   - Switch traffic after validation
   - Instant rollback by switching back

3. **Automated Rollback:**
   - Monitor key metrics (error rate, latency, accuracy)
   - Automatic rollback if thresholds breached

**Implementation Example (AWS SageMaker):**

```python
from datetime import datetime

# Version naming
version = f"v1.2.3-{git_sha[:8]}-{datetime.now().strftime('%Y%m%d')}"

# Register model with metadata
from sagemaker.model import Model

model = Model(
    model_data='s3://my-bucket/models/model.tar.gz',
    role=role,
    name=f'my-model-{version}',
    image_uri=container,
    env={
        'MODEL_VERSION': version,
        'GIT_SHA': git_sha,
        'TRAINING_DATA_VERSION': data_version
    }
)

# Blue-green deployment
# Step 1: Deploy new model to new endpoint (green)
green_predictor = model.deploy(
    initial_instance_count=2,
    instance_type='ml.m5.xlarge',
    endpoint_name='my-model-green'
)

# Step 2: Validate green endpoint
test_predictions = green_predictor.predict(test_data)
if validate(test_predictions):
    # Step 3: Update DNS/load balancer to point to green
    update_traffic_routing('my-model', 'my-model-green')

    # Step 4: Keep blue for rollback
    # Delete blue after 1 week if no issues
else:
    # Validation failed, delete green
    green_predictor.delete_endpoint()

# Automated rollback based on metrics
import boto3

cloudwatch = boto3.client('cloudwatch')

# Create alarm for error rate
cloudwatch.put_metric_alarm(
    AlarmName='HighErrorRate',
    MetricName='ModelInvocationErrors',
    Namespace='AWS/SageMaker',
    Statistic='Sum',
    Period=300,  # 5 minutes
    EvaluationPeriods=2,
    Threshold=10,  # >10 errors in 10 min
    ComparisonOperator='GreaterThanThreshold',
    AlarmActions=['arn:aws:sns:us-west-2:123456789012:rollback-topic']
)

# SNS topic triggers Lambda for automatic rollback
# Lambda switches traffic back to blue endpoint
```

---

## Conclusion

Cloud ML platforms (AWS SageMaker, GCP Vertex AI, Azure ML) provide comprehensive tooling for production ML at scale. The choice depends on:

1. **Existing cloud infrastructure**
2. **Framework preferences** (TensorFlow → Vertex AI, any → SageMaker)
3. **Data location** (BigQuery → Vertex AI, Redshift → SageMaker)
4. **Team expertise**
5. **Budget and scale**

**Key Takeaways:**
- All three platforms are production-ready and feature-rich
- Start with your current cloud provider
- Use managed services to accelerate development
- Optimize costs with spot instances and auto-scaling
- Implement proper versioning and rollback strategies
- Monitor everything (model performance, drift, costs)

**Next Steps:**
- Get hands-on with free tiers
- Build end-to-end project on each platform
- Compare features for your specific use case
- Join platform-specific communities

---

## Additional Resources

**AWS SageMaker:**
- Documentation: https://docs.aws.amazon.com/sagemaker/
- Examples: https://github.com/aws/amazon-sagemaker-examples
- Course: AWS ML Specialty Certification

**GCP Vertex AI:**
- Documentation: https://cloud.google.com/vertex-ai/docs
- Examples: https://github.com/GoogleCloudPlatform/vertex-ai-samples
- Course: Google Cloud ML Engineer Certification

**Azure ML:**
- Documentation: https://docs.microsoft.com/en-us/azure/machine-learning/
- Examples: https://github.com/Azure/azureml-examples
- Course: Microsoft Azure AI Engineer Certification

**Comparison & Best Practices:**
- "Architecting ML on AWS" - O'Reilly
- "Building ML Pipelines" - Hannes Hapke & Catherine Nelson
- Cloud provider blogs (AWS ML Blog, Google Cloud Blog, Azure Blog)

---

*Last Updated: October 26, 2025*
*Version: 1.0*
*Maintained by: ML Engineering Team*
