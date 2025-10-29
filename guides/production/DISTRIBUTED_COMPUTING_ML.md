# Distributed Computing for Machine Learning

**Scale ML workloads from single machine to thousands of cores**

> 🚀 **Distributed Training** | 📊 **Apache Spark** | ⚡ **Ray** | 🔄 **Data Parallelism** | 🧮 **Model Parallelism**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Apache Spark for ML](#apache-spark-for-ml)
3. [Ray for Distributed ML](#ray-for-distributed-ml)
4. [Distributed Training with PyTorch](#distributed-training-with-pytorch)
5. [Model Parallelism](#model-parallelism)
6. [Hyperparameter Tuning at Scale](#hyperparameter-tuning-at-scale)
7. [Production Best Practices](#production-best-practices)
8. [Interview Questions](#interview-questions)

---

## 1. Introduction

### Why Distributed Computing for ML?

**Challenges that require distribution:**
- **Large datasets:** Can't fit in single machine memory (>100GB)
- **Long training times:** Single GPU takes days/weeks
- **Large models:** Models don't fit in single GPU memory (LLMs, Vision Transformers)
- **Hyperparameter search:** Need to try many configurations in parallel

### Types of Parallelism

**1. Data Parallelism**
- Split data across machines
- Each machine has full model copy
- Synchronize gradients
- **Use case:** Training on large datasets

**2. Model Parallelism**
- Split model across machines
- Each machine has part of model
- Forward/backward pass coordinated
- **Use case:** Models too large for single GPU

**3. Pipeline Parallelism**
- Split model into stages
- Each stage on different machine
- Process mini-batches in pipeline
- **Use case:** Very deep models

### Distributed ML Stack

| Layer | Tools | Purpose |
|-------|-------|---------|
| **Data Processing** | Spark, Dask | Distributed ETL |
| **Training Framework** | PyTorch DDP, Horovod, DeepSpeed | Distributed training |
| **Orchestration** | Ray, Kubernetes | Resource management |
| **Tuning** | Ray Tune, Optuna | Distributed HPO |
| **Serving** | Ray Serve, KServe | Distributed inference |

---

## 2. Apache Spark for ML

### Spark for Data Processing

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# Initialize Spark
spark = SparkSession.builder \
    .appName("ML Feature Engineering") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Read large dataset (distributed)
df = spark.read.parquet("s3://bucket/data/events/")

print(f"Dataset size: {df.count():,} rows")
print(f"Partitions: {df.rdd.getNumPartitions()}")

# Feature engineering at scale
features_df = df.groupBy("user_id").agg(
    F.count("*").alias("total_events"),
    F.countDistinct(F.when(F.col("event_type") == "purchase", F.col("event_id"))).alias("total_purchases"),
    F.avg(F.when(F.col("event_type") == "purchase", F.col("amount"))).alias("avg_purchase_value"),
    F.datediff(F.current_date(), F.max(F.when(F.col("event_type") == "purchase", F.col("date")))).alias("days_since_last_purchase"),
    F.collect_set(F.when(F.col("event_type") == "view", F.col("category"))).alias("viewed_categories")
)

# Explode categories and count
category_counts = features_df \
    .withColumn("category", F.explode("viewed_categories")) \
    .groupBy("user_id", "category").count() \
    .groupBy("user_id") \
    .agg(F.max_by("category", "count").alias("favorite_category"))

# Join back
final_features = features_df.join(category_counts, on="user_id", how="left")

# Save to data lake
final_features.write.mode("overwrite").parquet("s3://bucket/features/user_features/")
```

### Distributed ML Training with Spark MLlib

```python
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Prepare features
feature_cols = ["total_events", "total_purchases", "avg_purchase_value", "days_since_last_purchase"]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

# Index target variable
indexer = StringIndexer(
    inputCol="label",
    outputCol="indexed_label"
)

# Model
gbt = GBTClassifier(
    labelCol="indexed_label",
    featuresCol="features",
    maxIter=100,
    maxDepth=5
)

# Pipeline
pipeline = Pipeline(stages=[assembler, indexer, gbt])

# Hyperparameter tuning (distributed)
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5, 10, 15]) \
    .addGrid(gbt.maxIter, [50, 100, 150]) \
    .build()

# Cross-validation (distributed across cluster)
cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(labelCol="indexed_label"),
    numFolds=5,
    parallelism=4  # Run 4 models in parallel
)

# Train (distributed)
cvModel = cv.fit(train_df)

# Best model
best_model = cvModel.bestModel

# Predictions (distributed)
predictions = best_model.transform(test_df)

# Evaluate
evaluator = BinaryClassificationEvaluator(labelCol="indexed_label")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc:.4f}")
```

### Performance Optimization

```python
# Repartition for better parallelism
df_repartitioned = df.repartition(200, "user_id")

# Cache frequently accessed data
df_cached = df.cache()

# Persist with different storage levels
from pyspark import StorageLevel
df.persist(StorageLevel.MEMORY_AND_DISK)

# Broadcast small lookup tables
from pyspark.sql.functions import broadcast
result = large_df.join(broadcast(small_df), on="key")

# Optimize shuffles
spark.conf.set("spark.sql.shuffle.partitions", "200")
spark.conf.set("spark.sql.adaptive.enabled", "true")
```

---

## 3. Ray for Distributed ML

### Ray Core Basics

```python
import ray
import numpy as np

# Initialize Ray
ray.init()

# Distributed function
@ray.remote
def process_batch(batch_data):
    """Process a batch of data in parallel"""
    # Simulate expensive computation
    result = np.mean(batch_data) * 2
    return result

# Distribute work
data_batches = [np.random.rand(1000) for _ in range(100)]

# Parallel execution
futures = [process_batch.remote(batch) for batch in data_batches]

# Get results
results = ray.get(futures)

print(f"Processed {len(results)} batches in parallel")
```

### Distributed Data Processing with Ray

```python
import ray
import pandas as pd

ray.init()

@ray.remote
def process_file(file_path):
    """Process a single file"""
    df = pd.read_csv(file_path)

    # Feature engineering
    features = df.groupby('user_id').agg({
        'event': 'count',
        'amount': 'sum',
        'timestamp': 'max'
    }).reset_index()

    return features

# Process files in parallel
file_paths = [f"data/events_{i}.csv" for i in range(100)]

futures = [process_file.remote(path) for path in file_paths]
results = ray.get(futures)

# Combine results
final_df = pd.concat(results, ignore_index=True)
print(f"Processed {len(file_paths)} files, {len(final_df)} total rows")
```

### Distributed Training with Ray Train

```python
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn

def train_func(config):
    """Training function that runs on each worker"""

    # Model
    model = nn.Sequential(
        nn.Linear(100, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # Wrap for distributed training
    model = train.torch.prepare_model(model)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Data (each worker gets a subset)
    train_data = train.get_dataset_shard("train")

    # Training loop
    for epoch in range(config["epochs"]):
        for batch in train_data.iter_batches(batch_size=32):
            X = torch.tensor(batch["features"], dtype=torch.float32)
            y = torch.tensor(batch["label"], dtype=torch.long)

            # Forward pass
            outputs = model(X)
            loss = nn.CrossEntropyLoss()(outputs, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Report metrics
        train.report({"loss": loss.item(), "epoch": epoch})

# Configure distributed training
scaling_config = ScalingConfig(
    num_workers=4,  # 4 distributed workers
    use_gpu=True,
    resources_per_worker={"CPU": 2, "GPU": 1}
)

# Create trainer
trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=scaling_config,
    train_loop_config={"lr": 0.001, "epochs": 10}
)

# Train (distributed)
result = trainer.fit()

print(f"Training complete: {result.metrics}")
```

### Distributed Hyperparameter Tuning with Ray Tune

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

def train_model(config):
    """Training function for hyperparameter tuning"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Model
    model = nn.Sequential(
        nn.Linear(100, config["hidden_size"]),
        nn.ReLU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(config["hidden_size"], 10)
    )

    # Dummy data
    X_train = torch.randn(1000, 100)
    y_train = torch.randint(0, 10, (1000,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config["batch_size"])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Training
    for epoch in range(10):
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Report to Ray Tune
        tune.report(loss=total_loss / len(train_loader))

# Search space
config = {
    "hidden_size": tune.choice([64, 128, 256]),
    "dropout": tune.uniform(0.1, 0.5),
    "lr": tune.loguniform(1e-4, 1e-2),
    "batch_size": tune.choice([32, 64, 128])
}

# Scheduler (early stopping)
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=10,
    grace_period=1,
    reduction_factor=2
)

# Search algorithm
search_alg = OptunaSearch()

# Run tuning (distributed across cluster)
analysis = tune.run(
    train_model,
    config=config,
    num_samples=100,  # Try 100 configurations
    scheduler=scheduler,
    search_alg=search_alg,
    resources_per_trial={"cpu": 2, "gpu": 0.5}
)

# Best config
best_config = analysis.best_config
print(f"Best config: {best_config}")
```

---

## 4. Distributed Training with PyTorch

### Data Parallel Training (Single Node, Multi-GPU)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Model
model = nn.Sequential(
    nn.Linear(100, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Wrap model for multi-GPU (simple approach)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.cuda()

# Data
X = torch.randn(10000, 100)
y = torch.randint(0, 10, (10000,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch_X, batch_y in loader:
        batch_X = batch_X.cuda()
        batch_y = batch_y.cuda()

        # Forward pass (automatically distributed across GPUs)
        outputs = model(batch_X)
        loss = nn.CrossEntropyLoss()(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Distributed Data Parallel (Multi-Node, Multi-GPU)

```python
# train.py
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import os

def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train(rank, world_size):
    """Training function for each process"""

    # Setup
    setup(rank, world_size)

    # Model
    model = nn.Sequential(
        nn.Linear(100, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(rank)

    # Wrap with DDP
    model = DDP(model, device_ids=[rank])

    # Data
    X = torch.randn(10000, 100)
    y = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(X, y)

    # Distributed sampler (each process gets different data)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(rank)
            batch_y = batch_y.to(rank)

            # Forward pass
            outputs = model(batch_X)
            loss = nn.CrossEntropyLoss()(outputs, batch_y)

            # Backward pass (gradients automatically synchronized)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if rank == 0:  # Only log from main process
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Cleanup
    cleanup()

# Launch distributed training
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
```

### Launch Script

```bash
# Single node, 4 GPUs
python train.py

# Multi-node (torchrun)
# Node 0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=192.168.1.1 --master_port=1234 \
    train.py

# Node 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=192.168.1.1 --master_port=1234 \
    train.py
```

### DeepSpeed for Large Model Training

```python
# deepspeed_train.py
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

# DeepSpeed configuration
ds_config = {
    "train_batch_size": 128,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,  # ZeRO-2: Optimizer + Gradient partitioning
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}

# Model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)

# Training loop
for batch in train_loader:
    inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(model_engine.device) for k, v in inputs.items()}

    outputs = model_engine(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss

    model_engine.backward(loss)
    model_engine.step()

# Save model
model_engine.save_checkpoint("checkpoints/")
```

```bash
# Launch DeepSpeed training
deepspeed --num_gpus=4 deepspeed_train.py
```

---

## 5. Model Parallelism

### Tensor Parallelism (Split Layers)

```python
import torch
import torch.nn as nn

class TensorParallelLinear(nn.Module):
    """Split linear layer across GPUs"""

    def __init__(self, in_features, out_features, num_gpus=2):
        super().__init__()
        self.num_gpus = num_gpus
        self.out_features_per_gpu = out_features // num_gpus

        # Create split layers on different GPUs
        self.layers = nn.ModuleList([
            nn.Linear(in_features, self.out_features_per_gpu).to(f'cuda:{i}')
            for i in range(num_gpus)
        ])

    def forward(self, x):
        # Replicate input to all GPUs
        outputs = []
        for i, layer in enumerate(self.layers):
            x_gpu = x.to(f'cuda:{i}')
            out = layer(x_gpu)
            outputs.append(out)

        # Concatenate outputs on GPU 0
        result = torch.cat([out.to('cuda:0') for out in outputs], dim=-1)
        return result

# Model with tensor parallelism
class TensorParallelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = TensorParallelLinear(1024, 4096, num_gpus=2)
        self.relu = nn.ReLU()
        self.layer2 = TensorParallelLinear(4096, 1024, num_gpus=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

### Pipeline Parallelism

```python
import torch
import torch.nn as nn

class PipelineParallelModel(nn.Module):
    """Split model into stages across GPUs"""

    def __init__(self):
        super().__init__()

        # Stage 1 on GPU 0
        self.stage1 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU()
        ).to('cuda:0')

        # Stage 2 on GPU 1
        self.stage2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        ).to('cuda:1')

        # Stage 3 on GPU 2
        self.stage3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        ).to('cuda:2')

    def forward(self, x):
        # Move through pipeline
        x = x.to('cuda:0')
        x = self.stage1(x)

        x = x.to('cuda:1')
        x = self.stage2(x)

        x = x.to('cuda:2')
        x = self.stage3(x)

        return x

# Training with pipeline parallelism
model = PipelineParallelModel()

for batch_X, batch_y in loader:
    # Input on GPU 0
    batch_X = batch_X.to('cuda:0')

    # Forward through pipeline
    outputs = model(batch_X)

    # Loss on GPU 2 (where output is)
    batch_y = batch_y.to('cuda:2')
    loss = nn.CrossEntropyLoss()(outputs, batch_y)

    # Backward (gradients flow back through pipeline)
    loss.backward()
```

---

## 6. Hyperparameter Tuning at Scale

### Distributed Tuning with Optuna

```python
import optuna
from optuna.storages import RDBStorage
import torch
import torch.nn as nn

# Shared database for distributed tuning
storage = RDBStorage(
    url="postgresql://user:password@localhost/optuna_db"
)

def objective(trial):
    """Objective function for hyperparameter tuning"""

    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Model
    model = nn.Sequential(
        nn.Linear(100, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 10)
    )

    # Training (simplified)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ... training loop ...

    # Return validation accuracy
    return val_accuracy

# Create study (shared across processes)
study = optuna.create_study(
    study_name="distributed_hpo",
    storage=storage,
    direction="maximize",
    load_if_exists=True
)

# Run optimization (can run on multiple machines)
study.optimize(objective, n_trials=100)

# Best parameters
print(f"Best trial: {study.best_trial.params}")
```

### Launch Distributed Tuning

```bash
# Machine 1
python tune.py

# Machine 2 (same study, shared database)
python tune.py

# Machine 3
python tune.py

# All machines will contribute trials to the same study
```

---

## 7. Production Best Practices

### 1. Efficient Data Loading

```python
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

class EfficientDataset(Dataset):
    """Memory-efficient dataset"""

    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load data on-the-fly (not in memory)
        data = torch.load(self.file_paths[idx])
        return data

# Efficient data loading
loader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=8,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=2,  # Prefetch batches
    persistent_workers=True  # Keep workers alive
)
```

### 2. Gradient Accumulation

```python
# Simulate large batch size with gradient accumulation
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

optimizer.zero_grad()

for i, (batch_X, batch_y) in enumerate(loader):
    # Forward pass
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)

    # Normalize loss (important!)
    loss = loss / accumulation_steps

    # Backward pass
    loss.backward()

    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Gradient scaler for mixed precision
scaler = GradScaler()

for batch_X, batch_y in loader:
    optimizer.zero_grad()

    # Forward pass with automatic mixed precision
    with autocast():
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

    # Backward pass with scaled gradients
    scaler.scale(loss).backward()

    # Unscale and clip gradients
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Update weights
    scaler.step(optimizer)
    scaler.update()
```

### 4. Monitoring Distributed Training

```python
import wandb
import torch.distributed as dist

# Initialize on rank 0 only
if dist.get_rank() == 0:
    wandb.init(project="distributed-training")

for epoch in range(num_epochs):
    for batch_X, batch_y in loader:
        # Training step
        loss = train_step(batch_X, batch_y)

        # Log from rank 0 only
        if dist.get_rank() == 0:
            wandb.log({"loss": loss.item(), "epoch": epoch})
```

---

## 8. Interview Questions

### Q1: When would you use data parallelism vs model parallelism?

**Answer:**

**Data Parallelism:**
- **Use when:** Model fits in single GPU memory
- **How:** Replicate model on each GPU, split data
- **Example:** ResNet-50 training on ImageNet
- **Pros:** Simple to implement, linear speedup
- **Cons:** Memory overhead (model replicated N times)

**Model Parallelism:**
- **Use when:** Model too large for single GPU
- **How:** Split model layers across GPUs
- **Example:** GPT-3 (175B parameters)
- **Pros:** Can train arbitrarily large models
- **Cons:** Complex communication, potential bottlenecks

**Decision Matrix:**

```
Model Size | Data Size | Choice
-----------|-----------|-------
Small      | Large     | Data Parallelism
Large      | Small     | Model Parallelism
Large      | Large     | Hybrid (both)
```

**Interview Follow-up:**
- "What about pipeline parallelism?" → For very deep models, reduces memory
- "Communication overhead?" → Data parallel: all-reduce gradients, Model parallel: activations

---

### Q2: You have 100GB of training data and 8 GPUs. How would you design the training pipeline?

**Answer:**

**Architecture:**

```
Data Storage (S3/HDFS)
    ↓
Distributed Data Loading
    ├── Worker 0 (GPU 0)
    ├── Worker 1 (GPU 1)
    ├── ...
    └── Worker 7 (GPU 7)
    ↓
PyTorch DDP Training
    ↓
Checkpointing & Monitoring
```

**Implementation:**

1. **Data Sharding:**
```python
# Each worker loads different shard
sampler = DistributedSampler(
    dataset,
    num_replicas=8,
    rank=gpu_id
)
```

2. **Efficient Loading:**
```python
loader = DataLoader(
    dataset,
    batch_size=128,  # per GPU
    sampler=sampler,
    num_workers=4,   # 4 CPU workers per GPU
    pin_memory=True
)
```

3. **Training:**
```python
# Distributed Data Parallel
model = DDP(model, device_ids=[gpu_id])

# Gradients automatically synchronized
loss.backward()
optimizer.step()
```

**Optimization:**
- Prefetch data during computation
- Use mixed precision (2x faster)
- Gradient accumulation if memory limited
- Save checkpoints from rank 0 only

**Expected Performance:**
- Ideal speedup: 8x
- Realistic: 6-7x (communication overhead)

**Interview Follow-up:**
- "What if data doesn't fit in memory?" → Stream from S3, use Petastorm/WebDataset
- "Handling stragglers?" → Gradient accumulation, dynamic batching

---

### Q3: Explain how gradient synchronization works in distributed data parallel training.

**Answer:**

**Process:**

**Step 1: Forward Pass (Independent)**
```python
# Each GPU processes different batch
# GPU 0: batch[0:128]
# GPU 1: batch[128:256]
# ...
# No communication needed
```

**Step 2: Backward Pass (Compute Local Gradients)**
```python
# Each GPU computes gradients for its batch
# Still independent
```

**Step 3: All-Reduce Gradients (Synchronization)**
```python
# Average gradients across all GPUs
# Communication: Ring All-Reduce algorithm

# Example with 4 GPUs:
# GPU 0 gradient: g0 = [1.0, 2.0, 3.0]
# GPU 1 gradient: g1 = [1.5, 2.5, 3.5]
# GPU 2 gradient: g2 = [0.8, 1.8, 2.8]
# GPU 3 gradient: g3 = [1.2, 2.2, 3.2]

# After all-reduce:
# All GPUs have: g_avg = (g0 + g1 + g2 + g3) / 4
#                      = [1.125, 2.125, 3.125]
```

**Step 4: Update Weights (Synchronized)**
```python
# All GPUs update with same gradients
# Models stay synchronized
```

**Ring All-Reduce Visualization:**

```
4 GPUs, each with gradient of size N

Phase 1: Scatter-Reduce
  GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0
  Each sends 1/4 of gradient

Phase 2: All-Gather
  GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0
  Each sends reduced chunk

Bandwidth: O(N) per GPU
Communication: O(N * (GPUs-1) / GPUs) ≈ O(N)
```

**Optimization:**
- **Gradient Bucketing:** Group small tensors for efficient communication
- **Overlap:** Backward pass and all-reduce overlapped
- **Compression:** Gradient quantization (reduces bandwidth)

**Interview Follow-up:**
- "Synchronous vs asynchronous?" → Synchronous: wait for all, Asynchronous: update immediately
- "Handling slow GPUs?" → Timeout, ignore stragglers (with care)

---

## Summary

**Distributed Computing for ML enables:**
- ✅ Training on datasets larger than memory
- ✅ Faster training with multiple GPUs/machines
- ✅ Training models larger than single GPU memory
- ✅ Hyperparameter tuning at scale

**Key Technologies:**
- **Data Processing:** Apache Spark
- **Distributed Training:** PyTorch DDP, DeepSpeed
- **Orchestration:** Ray, Kubernetes
- **HPO:** Ray Tune, Optuna

**Best Practices:**
- Start simple (single GPU) → scale up as needed
- Profile to identify bottlenecks
- Monitor resource utilization
- Use mixed precision for 2x speedup
- Checkpoint frequently

**Next Steps:**
1. Practice PyTorch DDP on multi-GPU machine
2. Set up Ray cluster for distributed tuning
3. Learn Spark for large-scale data processing
4. Try DeepSpeed for large model training
5. Benchmark speedups (1 GPU → 4 GPU → 8 GPU)

---

**Generated with Claude Code**

*Last Updated: 2025-10-25*
