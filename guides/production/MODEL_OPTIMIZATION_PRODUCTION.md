# Model Optimization for Production

**Complete Guide to Model Compression and Inference Optimization**

*Last Updated: October 26, 2025*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quantization](#quantization)
3. [Pruning](#pruning)
4. [Knowledge Distillation](#knowledge-distillation)
5. [Model Architecture Optimization](#model-architecture-optimization)
6. [Serving Optimization](#serving-optimization)
7. [Hardware-Specific Optimization](#hardware-specific-optimization)
8. [Benchmarking & Profiling](#benchmarking--profiling)
9. [Production Deployment](#production-deployment)
10. [Interview Questions](#interview-questions)

---

## Introduction

Model optimization reduces model size and improves inference speed while maintaining accuracy. This is critical for production deployment.

**Why Optimize:**
- **Reduce Latency**: Faster predictions (ms → μs)
- **Lower Costs**: Smaller models need less compute
- **Enable Edge Deployment**: Models fit on mobile/IoT devices
- **Increase Throughput**: More predictions per second
- **Reduce Memory**: Deploy on constrained hardware

**Optimization Techniques:**

| Technique | Size Reduction | Speed Improvement | Accuracy Loss |
|-----------|----------------|-------------------|---------------|
| Quantization (INT8) | 4x | 2-4x | <1% |
| Pruning (50%) | 2x | 1.5-2x | <2% |
| Knowledge Distillation | Varies | 2-10x | 1-5% |
| Architecture Search | Varies | 2-5x | <1% |

**Optimization Workflow:**

```
1. Baseline Model (FP32)
   ↓
2. Profile (find bottlenecks)
   ↓
3. Apply Optimizations
   ├─ Quantization
   ├─ Pruning
   └─ Distillation
   ↓
4. Benchmark (latency, throughput, accuracy)
   ↓
5. Fine-tune if needed
   ↓
6. Deploy optimized model
```

---

## Quantization

Quantization reduces precision of weights and activations from FP32 (32-bit) to INT8 (8-bit) or lower.

**Benefits:**
- **4x smaller model** (FP32 → INT8)
- **2-4x faster inference** (hardware-dependent)
- **4x less memory bandwidth**

**Types of Quantization:**

1. **Post-Training Quantization (PTQ)**: Quantize after training (fast, slight accuracy loss)
2. **Quantization-Aware Training (QAT)**: Train with quantization (slower, better accuracy)

### 1. PyTorch Quantization

**Dynamic Quantization (Easiest):**

```python
import torch
import torch.quantization as quantization

# Original FP32 model
model_fp32 = MyModel()
model_fp32.load_state_dict(torch.load('model.pth'))
model_fp32.eval()

# Dynamic quantization (weights quantized, activations in FP32)
model_int8 = torch.ao.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.LSTM},  # Layers to quantize
    dtype=torch.qint8
)

# Save quantized model
torch.save(model_int8.state_dict(), 'model_int8.pth')

# Size comparison
import os
fp32_size = os.path.getsize('model.pth') / (1024 * 1024)  # MB
int8_size = os.path.getsize('model_int8.pth') / (1024 * 1024)
print(f"FP32 size: {fp32_size:.2f} MB")
print(f"INT8 size: {int8_size:.2f} MB")
print(f"Compression ratio: {fp32_size / int8_size:.2f}x")
```

**Static Quantization (Better Performance):**

```python
# Step 1: Fuse layers (Conv + BN + ReLU)
model_fp32.fuse_model()  # Custom method to fuse layers

# Step 2: Specify quantization configuration
model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')  # or 'qnnpack' for mobile

# Step 3: Prepare model for quantization
model_prepared = torch.ao.quantization.prepare(model_fp32)

# Step 4: Calibrate with representative data
with torch.no_grad():
    for batch in calibration_data_loader:
        model_prepared(batch)

# Step 5: Convert to quantized model
model_int8 = torch.ao.quantization.convert(model_prepared)

# Compare accuracy
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

fp32_acc = evaluate(model_fp32, test_loader)
int8_acc = evaluate(model_int8, test_loader)

print(f"FP32 accuracy: {fp32_acc:.4f}")
print(f"INT8 accuracy: {int8_acc:.4f}")
print(f"Accuracy drop: {fp32_acc - int8_acc:.4f}")
```

**Quantization-Aware Training (QAT) - Best Accuracy:**

```python
import torch.ao.quantization as quantization

# Prepare model for QAT
model_fp32.qconfig = quantization.get_default_qat_qconfig('x86')
model_prepared = quantization.prepare_qat(model_fp32.train())

# Training loop with quantization
optimizer = torch.optim.Adam(model_prepared.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(5):  # Fine-tune for a few epochs
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model_prepared(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Convert to quantized model
model_prepared.eval()
model_int8 = quantization.convert(model_prepared)

# Evaluate
int8_acc = evaluate(model_int8, test_loader)
print(f"QAT INT8 accuracy: {int8_acc:.4f}")
```

### 2. TensorFlow Lite Quantization

**Post-Training Quantization:**

```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('model.h5')

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide representative dataset for calibration
def representative_dataset():
    for data in calibration_dataset.batch(1).take(100):
        yield [tf.cast(data, tf.float32)]

converter.representative_dataset = representative_dataset

# Full integer quantization (INT8)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert
tflite_model = converter.convert()

# Save
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

# Size comparison
import os
original_size = os.path.getsize('model.h5') / (1024 * 1024)
quantized_size = os.path.getsize('model_quantized.tflite') / (1024 * 1024)
print(f"Original: {original_size:.2f} MB")
print(f"Quantized: {quantized_size:.2f} MB")
print(f"Compression: {original_size / quantized_size:.2f}x")
```

**Quantization-Aware Training (TensorFlow):**

```python
import tensorflow_model_optimization as tfmot

# Apply quantization-aware training
quantize_model = tfmot.quantization.keras.quantize_model

# Quantize entire model
q_aware_model = quantize_model(model)

# Compile and train
q_aware_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune
q_aware_model.fit(train_dataset, epochs=5, validation_data=val_dataset)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### 3. ONNX Runtime Quantization

```python
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Export to ONNX
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model_fp32,
    dummy_input,
    "model.onnx",
    opset_version=14,
    input_names=['input'],
    output_names=['output']
)

# Dynamic quantization
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QInt8
)

# Benchmark
import time
import numpy as np

def benchmark_onnx(model_path, num_runs=100):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        session.run(None, {input_name: dummy_input})
    elapsed = (time.time() - start) / num_runs * 1000  # ms

    return elapsed

fp32_latency = benchmark_onnx("model.onnx")
int8_latency = benchmark_onnx("model_quantized.onnx")

print(f"FP32 latency: {fp32_latency:.2f} ms")
print(f"INT8 latency: {int8_latency:.2f} ms")
print(f"Speedup: {fp32_latency / int8_latency:.2f}x")
```

---

## Pruning

Pruning removes unnecessary weights/neurons to reduce model size and improve speed.

**Types of Pruning:**
1. **Unstructured Pruning**: Remove individual weights (better compression, less speedup)
2. **Structured Pruning**: Remove entire neurons/filters (better speedup)

**Pruning Workflow:**
```
1. Train baseline model
2. Prune weights below threshold
3. Fine-tune pruned model
4. Repeat (iterative pruning)
5. Export sparse model
```

### 1. PyTorch Pruning

**Magnitude-based Pruning:**

```python
import torch.nn.utils.prune as prune

# Model
model = MyModel()

# Prune 50% of weights in each Linear layer
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.5)

# Check sparsity
def check_sparsity(model):
    zeros = 0
    total = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            zeros += (module.weight == 0).sum().item()
            total += module.weight.numel()

    return zeros / total

sparsity = check_sparsity(model)
print(f"Sparsity: {sparsity:.2%}")

# Fine-tune pruned model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(5):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Make pruning permanent
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.remove(module, 'weight')

# Save pruned model
torch.save(model.state_dict(), 'model_pruned.pth')
```

**Structured Pruning (Entire Filters):**

```python
import torch.nn.utils.prune as prune

# Prune 30% of filters in Conv2d layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(
            module,
            name='weight',
            amount=0.3,
            n=2,  # L2 norm
            dim=0  # Output channels
        )

# Fine-tune
# ...

# Remove pruning reparameterization
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')
```

**Iterative Pruning (Better Results):**

```python
def iterative_pruning(model, train_loader, initial_sparsity=0.0, final_sparsity=0.9, steps=10):
    """Gradually increase pruning from initial to final sparsity"""

    sparsity_schedule = torch.linspace(initial_sparsity, final_sparsity, steps)

    for step, target_sparsity in enumerate(sparsity_schedule):
        print(f"Step {step+1}/{steps}, Target sparsity: {target_sparsity:.2%}")

        # Prune
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=target_sparsity)

        # Fine-tune for a few epochs
        for epoch in range(2):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Evaluate
        accuracy = evaluate(model, val_loader)
        print(f"Accuracy: {accuracy:.4f}")

    # Make pruning permanent
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')

    return model

pruned_model = iterative_pruning(model, train_loader, final_sparsity=0.8)
```

### 2. TensorFlow Pruning

```python
import tensorflow_model_optimization as tfmot

# Define pruning schedule
pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=0.8,
    begin_step=0,
    end_step=1000
)

# Apply pruning to model
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model,
    pruning_schedule=pruning_schedule
)

# Compile
model_for_pruning.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Add pruning callbacks
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir='logs')
]

# Train
model_for_pruning.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=callbacks
)

# Export pruned model
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.save('pruned_model.h5')
```

---

## Knowledge Distillation

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model.

**Benefits:**
- **2-10x smaller model**
- **Better accuracy than training student from scratch**
- **Captures teacher's "soft targets"**

**Distillation Loss:**
```
Loss = α * CE(student, hard_targets) + (1-α) * KL(student, teacher)
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss"""

    def __init__(self, temperature=3.0, alpha=0.5):
        """
        Args:
            temperature: Softmax temperature (higher = softer)
            alpha: Weight for hard targets (1-alpha for soft targets)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # Hard target loss (student vs ground truth)
        hard_loss = self.ce_loss(student_logits, labels)

        # Soft target loss (student vs teacher)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return total_loss

# Teacher model (large)
teacher = LargeModel()
teacher.load_state_dict(torch.load('teacher.pth'))
teacher.eval()

# Student model (small)
student = SmallModel()

# Distillation training
distillation_loss = DistillationLoss(temperature=3.0, alpha=0.5)
optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

for epoch in range(20):
    for data, labels in train_loader:
        optimizer.zero_grad()

        # Get teacher predictions (no gradients)
        with torch.no_grad():
            teacher_logits = teacher(data)

        # Get student predictions
        student_logits = student(data)

        # Compute distillation loss
        loss = distillation_loss(student_logits, teacher_logits, labels)

        loss.backward()
        optimizer.step()

    # Evaluate
    student_acc = evaluate(student, val_loader)
    print(f"Epoch {epoch}, Student Accuracy: {student_acc:.4f}")

# Compare with baseline
baseline_student = SmallModel()
train_baseline(baseline_student, train_loader)  # Train from scratch

baseline_acc = evaluate(baseline_student, test_loader)
distilled_acc = evaluate(student, test_loader)
teacher_acc = evaluate(teacher, test_loader)

print(f"Teacher accuracy: {teacher_acc:.4f}")
print(f"Student (distilled) accuracy: {distilled_acc:.4f}")
print(f"Student (baseline) accuracy: {baseline_acc:.4f}")
print(f"Distillation gain: {distilled_acc - baseline_acc:.4f}")
```

### TensorFlow Implementation

```python
import tensorflow as tf

class Distiller(tf.keras.Model):
    """Knowledge Distillation Model"""

    def __init__(self, student, teacher, temperature=3.0, alpha=0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

    def compile(self, optimizer, metrics):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_tracker = tf.keras.metrics.Mean(name="student_loss")

    def train_step(self, data):
        x, y = data

        # Forward pass through teacher (no training)
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass through student
            student_predictions = self.student(x, training=True)

            # Hard loss (student vs ground truth)
            hard_loss = tf.keras.losses.sparse_categorical_crossentropy(
                y, student_predictions
            )

            # Soft loss (student vs teacher)
            soft_student = tf.nn.softmax(student_predictions / self.temperature)
            soft_teacher = tf.nn.softmax(teacher_predictions / self.temperature)
            soft_loss = tf.keras.losses.kl_divergence(soft_teacher, soft_student)
            soft_loss *= self.temperature ** 2

            # Combined loss
            loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        # Update student weights
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.student_loss_tracker.update_state(loss)

        return {"loss": self.student_loss_tracker.result()}

# Create teacher and student
teacher = create_teacher_model()
teacher.load_weights('teacher_weights.h5')

student = create_student_model()

# Create distiller
distiller = Distiller(student, teacher, temperature=3.0, alpha=0.5)

# Compile and train
distiller.compile(optimizer='adam', metrics=['accuracy'])
distiller.fit(train_dataset, epochs=20, validation_data=val_dataset)

# Extract student model
student_model = distiller.student
student_model.save('distilled_student.h5')
```

---

## Model Architecture Optimization

Optimize model architecture for better efficiency without changing functionality.

### 1. Operator Fusion

Combine multiple operations into one for better performance.

```python
# Before: Separate Conv, BN, ReLU
x = Conv2d(in_channels, out_channels)(x)
x = BatchNorm2d(out_channels)(x)
x = ReLU()(x)

# After: Fused Conv-BN-ReLU (faster)
x = ConvBNReLU(in_channels, out_channels)(x)

# PyTorch: Automatic fusion
model.eval()  # Fusion happens in eval mode
torch.jit.script(model)  # JIT compiler fuses ops
```

### 2. Depthwise Separable Convolutions

Replace standard convolutions with depthwise separable (MobileNet approach).

```python
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (MobileNet)

    Computational cost:
    - Standard Conv: H × W × C_in × C_out × K × K
    - Depthwise Separable: H × W × C_in × (K × K + C_out)

    Speedup: ~C_out / (K × K + C_out) ≈ 8-9x for typical values
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        # Depthwise: Convolve each input channel separately
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,  # Key: groups = in_channels
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise: 1x1 convolution to combine channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

# Replace standard Conv with Depthwise Separable
# Standard: Conv2d(64, 128, kernel_size=3)
# Optimized: DepthwiseSeparableConv(64, 128, kernel_size=3)
```

### 3. Inverted Residuals (MobileNetV2)

```python
class InvertedResidual(nn.Module):
    """Inverted Residual Block (MobileNetV2)

    1. Expand with 1x1 conv
    2. Depthwise conv
    3. Project back with 1x1 conv
    4. Skip connection if stride=1
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
```

---

## Serving Optimization

Optimize the serving layer for better throughput and latency.

### 1. Batch Processing

```python
import asyncio
from collections import deque
import numpy as np

class DynamicBatcher:
    """Dynamic batching for improved throughput"""

    def __init__(self, model, max_batch_size=32, max_wait_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000  # Convert to seconds
        self.queue = deque()
        self.batch_task = None

    async def predict(self, input_data):
        """Add prediction request to batch queue"""
        future = asyncio.Future()
        self.queue.append((input_data, future))

        # Start batch processing if not already running
        if self.batch_task is None or self.batch_task.done():
            self.batch_task = asyncio.create_task(self._process_batch())

        return await future

    async def _process_batch(self):
        """Process accumulated requests as a batch"""
        await asyncio.sleep(self.max_wait_ms)

        if not self.queue:
            return

        # Collect batch
        batch_inputs = []
        futures = []

        while self.queue and len(batch_inputs) < self.max_batch_size:
            input_data, future = self.queue.popleft()
            batch_inputs.append(input_data)
            futures.append(future)

        # Process batch
        try:
            batch_array = np.vstack(batch_inputs)
            predictions = self.model.predict(batch_array)

            # Return results
            for future, prediction in zip(futures, predictions):
                future.set_result(prediction)

        except Exception as e:
            for future in futures:
                future.set_exception(e)

# Usage with FastAPI
from fastapi import FastAPI

app = FastAPI()
batcher = DynamicBatcher(model, max_batch_size=32, max_wait_ms=10)

@app.post("/predict")
async def predict(features: List[float]):
    prediction = await batcher.predict(np.array([features]))
    return {"prediction": float(prediction[0])}
```

### 2. Model Caching

```python
from functools import lru_cache
import hashlib
import pickle

class PredictionCache:
    """Cache predictions for frequently requested inputs"""

    def __init__(self, max_size=10000, ttl=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0

    def _hash_input(self, input_data):
        """Create hash of input for cache key"""
        # Round to reduce sensitivity
        rounded = np.round(input_data, decimals=4)
        data_bytes = pickle.dumps(rounded)
        return hashlib.sha256(data_bytes).hexdigest()

    def get(self, input_data):
        """Get cached prediction"""
        key = self._hash_input(input_data)

        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def set(self, input_data, prediction):
        """Cache prediction"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (FIFO)
            self.cache.pop(next(iter(self.cache)))

        key = self._hash_input(input_data)
        self.cache[key] = prediction

    def stats(self):
        """Cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }

# Usage
cache = PredictionCache(max_size=10000)

def predict_with_cache(input_data):
    # Check cache
    cached = cache.get(input_data)
    if cached is not None:
        return cached

    # Make prediction
    prediction = model.predict(input_data)

    # Cache result
    cache.set(input_data, prediction)

    return prediction

# Monitor cache performance
@app.get("/cache-stats")
def get_cache_stats():
    return cache.stats()
```

### 3. Multi-Model Serving

```python
class MultiModelServer:
    """Serve multiple models efficiently"""

    def __init__(self):
        self.models = {}
        self.model_stats = {}

    def load_model(self, model_name, model_path):
        """Load model into memory"""
        import joblib
        self.models[model_name] = joblib.load(model_path)
        self.model_stats[model_name] = {'requests': 0, 'errors': 0}

    def unload_model(self, model_name):
        """Unload model from memory"""
        if model_name in self.models:
            del self.models[model_name]

    def predict(self, model_name, input_data):
        """Make prediction with specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        try:
            prediction = self.models[model_name].predict(input_data)
            self.model_stats[model_name]['requests'] += 1
            return prediction
        except Exception as e:
            self.model_stats[model_name]['errors'] += 1
            raise

    def get_stats(self):
        """Get serving statistics"""
        return self.model_stats

# Usage
server = MultiModelServer()
server.load_model('model_v1', 'models/v1/model.pkl')
server.load_model('model_v2', 'models/v2/model.pkl')

@app.post("/predict/{model_name}")
def predict(model_name: str, features: List[float]):
    prediction = server.predict(model_name, np.array([features]))
    return {"prediction": float(prediction[0])}
```

---

## Hardware-Specific Optimization

Optimize for specific hardware (CPU, GPU, TPU, mobile).

### 1. TensorRT (NVIDIA GPUs)

```python
import torch
import torch_tensorrt

# PyTorch model
model = torchvision.models.resnet50(pretrained=True).eval().cuda()

# Convert to TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions={torch.float, torch.half},  # FP32 and FP16
    workspace_size=1 << 30  # 1GB
)

# Benchmark
import time

dummy_input = torch.randn(1, 3, 224, 224).cuda()

# Warmup
for _ in range(10):
    _ = model(dummy_input)
    _ = trt_model(dummy_input)

# PyTorch
start = time.time()
for _ in range(100):
    _ = model(dummy_input)
torch.cuda.synchronize()
pytorch_time = (time.time() - start) / 100 * 1000

# TensorRT
start = time.time()
for _ in range(100):
    _ = trt_model(dummy_input)
torch.cuda.synchronize()
trt_time = (time.time() - start) / 100 * 1000

print(f"PyTorch latency: {pytorch_time:.2f} ms")
print(f"TensorRT latency: {trt_time:.2f} ms")
print(f"Speedup: {pytorch_time / trt_time:.2f}x")
```

### 2. Intel OpenVINO (Intel CPUs)

```python
from openvino.runtime import Core

# Convert PyTorch to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=14
)

# Load with OpenVINO
ie = Core()
model_onnx = ie.read_model(model="model.onnx")
compiled_model = ie.compile_model(model=model_onnx, device_name="CPU")

# Get input/output layers
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Inference
import numpy as np

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = compiled_model([input_data])[output_layer]
```

### 3. CoreML (Apple Devices)

```python
import coremltools as ct

# Convert PyTorch to CoreML
traced_model = torch.jit.trace(model, dummy_input)

coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(1, 3, 224, 224))],
    minimum_deployment_target=ct.target.iOS15
)

# Save
coreml_model.save("model.mlmodel")

# Optimize for specific hardware
coreml_model_optimized = ct.optimize_weights(
    coreml_model,
    quantization_mode="linear"
)
```

---

## Benchmarking & Profiling

Measure and optimize performance systematically.

### 1. Comprehensive Benchmarking

```python
import time
import torch
import numpy as np
from typing import Dict, List

class ModelBenchmark:
    """Comprehensive model benchmarking"""

    def __init__(self, model, input_shape, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.input_shape = input_shape
        self.device = device

    def measure_latency(self, num_runs=100, warmup=10):
        """Measure inference latency"""
        dummy_input = torch.randn(*self.input_shape).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)

        # Benchmark
        if self.device == 'cuda':
            torch.cuda.synchronize()

        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = self.model(dummy_input)

                if self.device == 'cuda':
                    torch.cuda.synchronize()

                latencies.append((time.time() - start) * 1000)  # ms

        return {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99)
        }

    def measure_throughput(self, batch_size=32, duration_seconds=10):
        """Measure throughput (samples/second)"""
        dummy_input = torch.randn(batch_size, *self.input_shape[1:]).to(self.device)

        num_batches = 0
        start_time = time.time()

        with torch.no_grad():
            while time.time() - start_time < duration_seconds:
                _ = self.model(dummy_input)
                num_batches += 1

                if self.device == 'cuda':
                    torch.cuda.synchronize()

        elapsed = time.time() - start_time
        throughput = (num_batches * batch_size) / elapsed

        return {
            'throughput_samples_per_sec': throughput,
            'batches_processed': num_batches
        }

    def measure_memory(self):
        """Measure model memory usage"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            dummy_input = torch.randn(*self.input_shape).to(self.device)

            with torch.no_grad():
                _ = self.model(dummy_input)

            memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            # CPU memory is harder to measure accurately
            memory_mb = None

        # Model size
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)

        return {
            'model_size_mb': model_size_mb,
            'peak_memory_mb': memory_mb
        }

    def count_parameters(self):
        """Count model parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_parameters_millions': total_params / 1e6
        }

    def profile_flops(self):
        """Estimate FLOPs"""
        try:
            from thop import profile
            dummy_input = torch.randn(*self.input_shape).to(self.device)
            flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)

            return {
                'flops': flops,
                'flops_billions': flops / 1e9
            }
        except ImportError:
            return {'flops': None, 'note': 'Install thop: pip install thop'}

    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("Running comprehensive benchmark...")

        results = {}

        # Latency
        print("  Measuring latency...")
        results['latency'] = self.measure_latency()

        # Throughput
        print("  Measuring throughput...")
        results['throughput'] = self.measure_throughput()

        # Memory
        print("  Measuring memory...")
        results['memory'] = self.measure_memory()

        # Parameters
        print("  Counting parameters...")
        results['parameters'] = self.count_parameters()

        # FLOPs
        print("  Estimating FLOPs...")
        results['flops'] = self.profile_flops()

        return results

# Usage
model = MyModel()
benchmark = ModelBenchmark(model, input_shape=(1, 3, 224, 224), device='cuda')
results = benchmark.run_full_benchmark()

# Print results
print("\n=== Benchmark Results ===")
print(f"Latency (p50): {results['latency']['p50_ms']:.2f} ms")
print(f"Latency (p99): {results['latency']['p99_ms']:.2f} ms")
print(f"Throughput: {results['throughput']['throughput_samples_per_sec']:.0f} samples/sec")
print(f"Model size: {results['memory']['model_size_mb']:.2f} MB")
print(f"Parameters: {results['parameters']['total_parameters_millions']:.2f}M")
if results['flops']['flops'] is not None:
    print(f"FLOPs: {results['flops']['flops_billions']:.2f}B")
```

### 2. Profiling Bottlenecks

```python
import torch.profiler as profiler

# Profile model
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with profiler.record_function("model_inference"):
        for _ in range(10):
            model(dummy_input)

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export to Chrome trace
prof.export_chrome_trace("trace.json")
# View at chrome://tracing
```

---

## Production Deployment

Deploy optimized models to production.

### Deployment Checklist

```markdown
## Pre-Deployment

- [ ] Benchmark baseline model (latency, throughput, memory)
- [ ] Apply optimizations (quantization, pruning, distillation)
- [ ] Benchmark optimized model
- [ ] Verify accuracy drop is acceptable (<1-2%)
- [ ] Test on production-like data
- [ ] Profile for bottlenecks
- [ ] Optimize serving layer (batching, caching)
- [ ] Load test (can handle peak traffic?)
- [ ] Set up monitoring (latency, throughput, errors)

## Deployment

- [ ] Deploy to staging environment
- [ ] Run integration tests
- [ ] Canary deployment (5% traffic)
- [ ] Monitor metrics for 24 hours
- [ ] Gradual rollout (25% → 50% → 100%)
- [ ] Keep previous version for rollback

## Post-Deployment

- [ ] Monitor performance daily
- [ ] Check for model drift
- [ ] A/B test vs previous version
- [ ] Optimize based on production data
- [ ] Document learnings
```

### Example Deployment Script

```python
# deploy_optimized_model.py
import torch
import joblib
from pathlib import Path
import json

def optimize_and_deploy(
    model_path: str,
    output_dir: str,
    quantize: bool = True,
    prune_amount: float = 0.0,
    benchmark: bool = True
):
    """Optimize model and prepare for deployment"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = torch.load(model_path)
    model.eval()

    # Baseline benchmark
    if benchmark:
        print("Benchmarking baseline...")
        baseline_benchmark = ModelBenchmark(model, (1, 3, 224, 224))
        baseline_results = baseline_benchmark.run_full_benchmark()

    # Apply optimizations
    if prune_amount > 0:
        print(f"Pruning {prune_amount:.0%} of weights...")
        model = apply_pruning(model, prune_amount)

    if quantize:
        print("Quantizing to INT8...")
        model = torch.ao.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )

    # Optimized benchmark
    if benchmark:
        print("Benchmarking optimized model...")
        opt_benchmark = ModelBenchmark(model, (1, 3, 224, 224))
        opt_results = opt_benchmark.run_full_benchmark()

        # Compare
        print("\n=== Optimization Results ===")
        print(f"Latency improvement: {baseline_results['latency']['p50_ms'] / opt_results['latency']['p50_ms']:.2f}x")
        print(f"Size reduction: {baseline_results['memory']['model_size_mb'] / opt_results['memory']['model_size_mb']:.2f}x")

        # Save comparison
        comparison = {
            'baseline': baseline_results,
            'optimized': opt_results
        }
        with open(output_dir / 'benchmark_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)

    # Save optimized model
    print(f"Saving optimized model to {output_dir}...")
    torch.save(model, output_dir / 'optimized_model.pth')

    # Save deployment metadata
    metadata = {
        'model_path': str(model_path),
        'optimizations': {
            'quantization': quantize,
            'pruning': prune_amount
        },
        'timestamp': pd.Timestamp.now().isoformat()
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Deployment preparation complete!")

    return model

# Run optimization
optimized_model = optimize_and_deploy(
    model_path='models/baseline.pth',
    output_dir='models/optimized',
    quantize=True,
    prune_amount=0.5,
    benchmark=True
)
```

---

## Interview Questions

### Q1: Explain the difference between post-training quantization and quantization-aware training.

**Answer:**

**Post-Training Quantization (PTQ):**
- Quantize model after training is complete
- **Pros:** Fast (no retraining), easy to apply
- **Cons:** Slight accuracy drop (1-2%)
- **When to use:** Quick optimization, accuracy loss acceptable

**Process:**
1. Train FP32 model normally
2. Collect calibration data
3. Compute quantization parameters (scale, zero-point)
4. Convert weights and activations to INT8

**Quantization-Aware Training (QAT):**
- Simulate quantization during training
- **Pros:** Better accuracy (< 0.5% drop), learns quantization error
- **Cons:** Slower (requires retraining)
- **When to use:** Need best accuracy, can afford training time

**Process:**
1. Insert fake quantization nodes in model
2. Train/fine-tune with quantization simulation
3. Model learns to be robust to quantization errors
4. Convert to actual INT8 model

**Code Example:**
```python
# PTQ: Fast, slight accuracy loss
model_int8 = torch.ao.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# QAT: Slower, better accuracy
model_qat = quantization.prepare_qat(model_fp32)
# Train for a few epochs
model_int8 = quantization.convert(model_qat)
```

**Rule of Thumb:**
- Start with PTQ (fast, good enough for most cases)
- Use QAT if accuracy drop > 1%

---

### Q2: What is knowledge distillation and when would you use it?

**Answer:**

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model.

**How it Works:**

1. **Soft Targets:** Teacher's output probabilities contain more information than hard labels
   - Hard label: [0, 0, 1, 0] (cat)
   - Soft targets: [0.05, 0.1, 0.8, 0.05] (mostly cat, slight dog similarity)

2. **Loss Function:**
   ```
   Loss = α × CE(student, hard_labels) + (1-α) × KL(student, teacher)
   ```

3. **Temperature:** Softens probabilities for better knowledge transfer
   ```python
   soft_targets = F.softmax(logits / temperature, dim=1)
   ```

**Benefits:**
- **2-10x smaller model**
- **Better than training student from scratch** (gains 2-5% accuracy)
- **Captures teacher's learned representations**

**When to Use:**

✅ **Good for:**
- Deploying to mobile/edge (need smaller models)
- Teacher is ensemble (distill to single model)
- Have large teacher, need fast inference
- Teacher is BERT, want faster student

❌ **Not good for:**
- Teacher and student same size (no compression)
- Teacher accuracy is poor (garbage in, garbage out)
- No trained teacher available

**Real-World Example:**
- DistilBERT: 6-layer student of 12-layer BERT
- Result: 40% smaller, 60% faster, 97% of BERT's performance

**Code:**
```python
# Distillation loss
def distillation_loss(student_logits, teacher_logits, labels, T=3, alpha=0.5):
    # Soft loss (student learns from teacher)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)

    # Hard loss (student learns from labels)
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * hard_loss + (1 - alpha) * soft_loss
```

---

### Q3: How do you optimize model inference latency for production?

**Answer:**

**Multi-Level Optimization Strategy:**

**1. Model-Level (2-10x speedup):**
- **Quantization:** FP32 → INT8 (4x smaller, 2-4x faster)
- **Pruning:** Remove 50-80% of weights (2x smaller, 1.5-2x faster)
- **Knowledge Distillation:** Use smaller architecture (2-10x faster)

**2. Architecture-Level (2-5x speedup):**
- **Depthwise Separable Convolutions:** Replace standard conv (8-9x FLOPs reduction)
- **MobileNet/EfficientNet:** Use efficient architectures
- **Reduce depth/width:** Smaller models if accuracy allows

**3. Serving-Level (2-4x speedup):**
- **Batch Processing:** Dynamic batching (2-3x throughput)
- **Caching:** Cache frequent predictions (instant for cache hits)
- **Model Compilation:** TorchScript, ONNX Runtime (10-30% speedup)

**4. Hardware-Level (2-10x speedup):**
- **TensorRT (NVIDIA GPUs):** Optimize for CUDA (2-5x faster)
- **OpenVINO (Intel CPUs):** Optimize for Intel (2-3x faster)
- **CoreML (Apple):** Optimize for iPhone/Mac (3-10x faster)

**5. System-Level:**
- **Multi-threading:** Parallelize requests
- **Load balancing:** Distribute across instances
- **GPU vs CPU:** Use GPU for large models

**Practical Example:**

```python
# Baseline: 100ms latency
model_fp32 = ResNet50()

# Step 1: Quantization (→ 50ms, 2x speedup)
model_int8 = quantize_dynamic(model_fp32)

# Step 2: TensorRT (→ 20ms, 2.5x speedup)
model_trt = torch_tensorrt.compile(model_int8)

# Step 3: Batching (→ 5ms/sample at batch=32, 4x throughput)
batcher = DynamicBatcher(model_trt, max_batch_size=32)

# Total: 100ms → 5ms/sample = 20x improvement
```

**Decision Framework:**

| Target Latency | Approach |
|----------------|----------|
| <10ms | Quantization + TensorRT + Batching |
| <50ms | Quantization + ONNX Runtime |
| <100ms | Model compression only |
| Batch | Any optimization + large batches |

**Measure Everything:**
```python
# Always benchmark before/after
baseline_latency = benchmark(model_fp32)
optimized_latency = benchmark(model_optimized)
speedup = baseline_latency / optimized_latency
```

---

### Q4: What are the trade-offs between unstructured and structured pruning?

**Answer:**

**Unstructured Pruning:**
- **What:** Remove individual weights below threshold
- **Sparsity Pattern:** Irregular (any weight can be zero)
- **Compression:** High (can remove 80-95% of weights)
- **Speedup:** Low (1.1-1.3x) without specialized hardware
- **Why slow:** Sparse matrix operations not well-supported on GPUs

**Structured Pruning:**
- **What:** Remove entire neurons/filters/channels
- **Sparsity Pattern:** Regular (entire structures removed)
- **Compression:** Moderate (typically remove 30-60%)
- **Speedup:** High (1.5-3x) on standard hardware
- **Why fast:** Results in smaller dense matrices

**Comparison Table:**

| Aspect | Unstructured | Structured |
|--------|--------------|------------|
| Compression | 4-10x | 1.5-2.5x |
| Speedup (GPU) | 1.1-1.3x | 1.5-3x |
| Speedup (specialized HW) | 2-4x | 1.5-3x |
| Accuracy preservation | Better | Worse |
| Implementation | Complex | Simple |

**Code Examples:**

```python
# Unstructured: Remove individual weights
prune.l1_unstructured(module, name='weight', amount=0.8)
# Result: 80% of weights are zero (scattered)

# Structured: Remove entire filters
prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)
# Result: 50% of filters removed completely
```

**When to Use:**

**Unstructured if:**
- Have specialized hardware (NVIDIA Ampere A100, Apple Neural Engine)
- Care more about compression than speed
- Can accept complex deployment

**Structured if:**
- Deploying on standard hardware (GPUs, CPUs)
- Need actual inference speedup
- Want simpler deployment

**Best Practice:**
- Start with structured pruning (easier, real speedup)
- Use unstructured only if you have specialized hardware

---

### Q5: How would you optimize a BERT model for production deployment?

**Answer:**

**Optimization Strategy for BERT:**

BERT-base: 110M parameters, ~140MB, ~70ms latency
Goal: <20ms latency, <50MB size

**Approach:**

**1. Knowledge Distillation (DistilBERT):**
```
BERT-base (12 layers) → DistilBERT (6 layers)
- 40% smaller
- 60% faster
- 97% of BERT's performance
```

**2. Quantization:**
```
FP32 (140MB) → INT8 (35MB)
- 4x smaller
- 2-3x faster
```

**3. ONNX Runtime:**
```
PyTorch → ONNX → ONNX Runtime
- 1.5-2x faster
- Better CPU performance
```

**4. Optimize Serving:**
```
- Dynamic batching (batch up to 32)
- Caching for repeated inputs
- Async processing
```

**Implementation:**

```python
# Step 1: Distillation (use DistilBERT)
from transformers import DistilBertModel
student_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
# Or train your own student

# Step 2: Quantization
from transformers import BertModel
import torch

model = BertModel.from_pretrained('distilbert-base-uncased')
model.eval()

model_int8 = torch.ao.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Step 3: Export to ONNX
dummy_input = torch.ones(1, 128, dtype=torch.long)
torch.onnx.export(
    model_int8,
    (dummy_input,),
    "distilbert_int8.onnx",
    opset_version=14
)

# Step 4: ONNX Runtime inference
import onnxruntime as ort

session = ort.InferenceSession("distilbert_int8.onnx")
input_name = session.get_inputs()[0].name

# Inference
result = session.run(None, {input_name: dummy_input.numpy()})
```

**Results:**

| Stage | Size | Latency | Accuracy |
|-------|------|---------|----------|
| BERT-base | 440MB | 70ms | 100% |
| DistilBERT | 260MB | 28ms | 97% |
| + Quantization | 65MB | 14ms | 96% |
| + ONNX RT | 65MB | 10ms | 96% |
| + Batching (32) | 65MB | 0.5ms/sample | 96% |

**Alternative Approaches:**

1. **TinyBERT:** Even smaller (4 layers, 14M params, 7x faster)
2. **ALBERT:** Parameter sharing (reduce params, same accuracy)
3. **Pruning:** Remove attention heads/layers (structured pruning)

**Production Deployment:**
```python
from fastapi import FastAPI
import onnxruntime as ort

app = FastAPI()
session = ort.InferenceSession("distilbert_int8.onnx")

@app.post("/predict")
async def predict(text: str):
    # Tokenize
    inputs = tokenizer(text, return_tensors="np")

    # Predict
    outputs = session.run(None, {
        'input_ids': inputs['input_ids']
    })

    return {"embedding": outputs[0].tolist()}
```

**Key Takeaway:** Combined optimizations achieve 7x speedup and 7x size reduction with only 4% accuracy drop.

---

## Conclusion

Model optimization is essential for production ML deployment. Key techniques:

1. **Quantization:** 4x smaller, 2-4x faster (INT8)
2. **Pruning:** 2x smaller, 1.5-2x faster (structured)
3. **Distillation:** 2-10x faster (smaller architecture)
4. **Serving:** 2-4x throughput (batching, caching)
5. **Hardware:** 2-10x faster (TensorRT, OpenVINO, CoreML)

**Best Practices:**
- Start simple (PTQ quantization)
- Measure everything (benchmark before/after)
- Combine techniques (quantization + pruning + batching)
- Test on production data
- Monitor in production

**Typical Production Stack:**
```
Model (quantized + pruned)
  ↓
ONNX Runtime / TensorRT
  ↓
Dynamic Batching
  ↓
Caching Layer
  ↓
FastAPI / TorchServe
  ↓
Kubernetes + Load Balancer
```

This achieves 10-100x latency reduction while maintaining >95% accuracy.

---

*Last Updated: October 26, 2025*
*Version: 1.0*
*Maintained by: ML Engineering Team*
