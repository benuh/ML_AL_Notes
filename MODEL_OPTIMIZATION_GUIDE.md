# ⚡ Model Optimization Guide

## Complete Guide to Making ML Models Faster, Smaller, and More Efficient

This comprehensive guide covers all techniques for optimizing machine learning models for production deployment - from quantization to pruning to knowledge distillation.

---

## 📋 Table of Contents

1. [Why Optimize Models](#why-optimize-models)
2. [Quantization](#quantization)
3. [Pruning](#pruning)
4. [Knowledge Distillation](#knowledge-distillation)
5. [Architecture Optimization](#architecture-optimization)
6. [Inference Optimization](#inference-optimization)
7. [Hardware Acceleration](#hardware-acceleration)
8. [Benchmarking & Profiling](#benchmarking--profiling)

---

## 🎯 Why Optimize Models

### The Optimization Trilemma

```
Pick 2 of 3:
┌─────────────┐
│   Accuracy  │
└─────────────┘
       ╱ ╲
      ╱   ╲
     ╱     ╲
┌────────┐ ┌────────┐
│ Speed  │ │  Size  │
└────────┘ └────────┘

Reality: Need all 3 for production!
Solution: Optimization techniques
```

### Why It Matters

**1. Latency Requirements**
```
Use Case              Target Latency
─────────────────────────────────────
Real-time video       < 33ms (30 FPS)
Voice assistant       < 100ms
Search ranking        < 50ms
Recommendation        < 100ms
Batch processing      Minutes-Hours OK
```

**2. Cost Savings**
```
Scenario: Image classification service
- 1M requests/day
- Without optimization: 100ms per request on GPU
  Cost: $500/month GPU compute

- With optimization: 20ms per request on CPU
  Cost: $50/month CPU compute

Savings: $450/month = $5,400/year
```

**3. Edge Deployment**
```
Device Constraints:
- Mobile: Limited battery, memory, compute
- IoT: Tiny processors, no GPU
- Embedded: Real-time requirements

Solution: Optimized models (quantized, pruned, distilled)
```

### Optimization Metrics

**Model Size:**
- FP32: 4 bytes per parameter
- FP16: 2 bytes per parameter
- INT8: 1 byte per parameter
- Binary: 1 bit per parameter

**Inference Speed:**
- Latency: Time per prediction
- Throughput: Predictions per second
- FLOPs: Floating point operations

**Accuracy:**
- Test accuracy
- Precision/Recall
- Task-specific metrics

---

## 🔢 Quantization

### What is Quantization?

**Concept:** Reduce numerical precision of weights and activations

```python
# Full Precision (FP32)
weight = 0.123456789  # 32 bits, very precise

# Half Precision (FP16)
weight = 0.1235  # 16 bits, 2x smaller

# 8-bit Integer (INT8)
weight = 31  # 8 bits, 4x smaller
# Represents: 31/127 ≈ 0.244

# Binary (1-bit)
weight = 1  # 1 bit, 32x smaller
# Represents: +1 or -1
```

### Types of Quantization

**1. Post-Training Quantization (PTQ)**
```python
import torch

# Train model normally (FP32)
model = train_model()

# Quantize after training (no retraining needed)
model_int8 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize Linear layers
    dtype=torch.qint8
)

# Compare sizes
fp32_size = get_model_size(model)  # 100 MB
int8_size = get_model_size(model_int8)  # 25 MB

print(f"Size reduction: {fp32_size / int8_size:.1f}x")
# Output: Size reduction: 4.0x
```

**Benefits:**
- ✅ No retraining needed
- ✅ Fast to apply
- ✅ Works well for large models

**Limitations:**
- ❌ Some accuracy loss (typically 1-2%)
- ❌ Not as accurate as QAT

**2. Quantization-Aware Training (QAT)**
```python
import torch.quantization

# Prepare model for QAT
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model)

# Train with quantization simulation
for epoch in range(epochs):
    for data, target in train_loader:
        # Forward pass simulates quantization
        output = model_prepared(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Convert to actual quantized model
model_quantized = torch.quantization.convert(model_prepared)
```

**Benefits:**
- ✅ Minimal accuracy loss (<0.5%)
- ✅ Better than PTQ
- ✅ Model learns to be robust to quantization

**Limitations:**
- ❌ Requires retraining
- ❌ More complex

**3. Dynamic vs Static Quantization**

**Dynamic Quantization:**
```python
# Quantizes weights only (activations stay FP32)
model_dynamic = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# Use case: NLP models (variable sequence length)
# Benefits: Easy, no calibration needed
# Speedup: 2-3x on CPU
```

**Static Quantization:**
```python
# Quantizes both weights AND activations

# Step 1: Fuse operations
model_fused = torch.quantization.fuse_modules(
    model,
    [['conv1', 'bn1', 'relu1']]  # Fuse conv-bn-relu
)

# Step 2: Set quantization config
model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Step 3: Calibrate (run representative data)
model_prepared = torch.quantization.prepare(model_fused)

with torch.no_grad():
    for data, _ in calibration_loader:
        model_prepared(data)  # Collect statistics

# Step 4: Convert to quantized model
model_quantized = torch.quantization.convert(model_prepared)

# Use case: Computer vision (fixed input size)
# Benefits: Maximum speedup (4-5x)
# Requires: Calibration data
```

### Quantization in Practice

**Example: Quantize ResNet50**
```python
import torch
import torchvision.models as models
from torch.quantization import quantize_dynamic, quantize_static

# Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Dynamic quantization (easiest)
model_dynamic = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Static quantization (best performance)
# Step 1: Fuse layers
model.fuse_model()

# Step 2: Configure
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)

# Step 3: Calibrate
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, _ in data_loader:
            model(image)

calibrate(model_prepared, calibration_loader)

# Step 4: Convert
model_quantized = torch.quantization.convert(model_prepared)

# Evaluate
def benchmark(model, data_loader):
    import time
    model.eval()

    start = time.time()
    with torch.no_grad():
        for image, _ in data_loader:
            _ = model(image)
    end = time.time()

    return end - start

fp32_time = benchmark(model, test_loader)
int8_time = benchmark(model_quantized, test_loader)

print(f"FP32: {fp32_time:.2f}s")
print(f"INT8: {int8_time:.2f}s")
print(f"Speedup: {fp32_time / int8_time:.1f}x")

# Typical output:
# FP32: 15.23s
# INT8: 3.87s
# Speedup: 3.9x
```

### Advanced Quantization

**Mixed Precision Quantization:**
```python
# Quantize different layers to different precisions

# Sensitive layers: Keep FP32 or FP16
# - First layer (high information)
# - Last layer (critical for accuracy)
# - Attention layers (precision important)

# Insensitive layers: INT8 or lower
# - Middle layers
# - Large linear layers

# Example configuration
qconfig_dict = {
    '': torch.quantization.get_default_qconfig('fbgemm'),  # Default INT8
    'module.layer1': None,  # Keep FP32
    'module.layer_last': None,  # Keep FP32
}

model_prepared = quantization.prepare(model, qconfig_dict)
```

**Custom Quantization:**
```python
# Define custom quantization scheme

class CustomQuantize(torch.nn.Module):
    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        self.scale = None
        self.zero_point = None

    def forward(self, x):
        if self.scale is None:
            # Calibration phase
            self.scale = (x.max() - x.min()) / (2**self.bits - 1)
            self.zero_point = x.min()

        # Quantize
        x_quant = torch.round((x - self.zero_point) / self.scale)
        x_quant = torch.clamp(x_quant, 0, 2**self.bits - 1)

        # Dequantize (for training)
        x_dequant = x_quant * self.scale + self.zero_point

        return x_dequant
```

---

## ✂️ Pruning

### What is Pruning?

**Concept:** Remove unnecessary weights/neurons to reduce model size and computation

```
Dense Network:          Pruned Network:
[●][●][●][●][●]        [●][ ][●][ ][●]
[●][●][●][●][●]  →     [ ][●][ ][●][ ]
[●][●][●][●][●]        [●][ ][●][ ][●]

50% weights removed
50% faster inference (with sparse operations)
```

### Types of Pruning

**1. Magnitude-Based Pruning**
```python
import torch
import torch.nn.utils.prune as prune

def magnitude_prune(model, amount=0.3):
    """
    Prune weights with smallest magnitude

    amount: Percentage to prune (0.3 = 30%)
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(
                module,
                name='weight',
                amount=amount
            )

    return model

# Example
model = MyModel()
model_pruned = magnitude_prune(model, amount=0.5)  # Remove 50%

# Check sparsity
def check_sparsity(model):
    zeros = 0
    total = 0

    for param in model.parameters():
        zeros += torch.sum(param == 0).item()
        total += param.numel()

    sparsity = zeros / total
    print(f"Sparsity: {sparsity:.1%}")
    return sparsity

check_sparsity(model_pruned)
# Output: Sparsity: 50.0%
```

**2. Structured Pruning**
```python
# Remove entire channels/neurons (easier to accelerate)

def structured_prune_channels(model, amount=0.3):
    """
    Prune entire channels based on L1 norm
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(
                module,
                name='weight',
                amount=amount,
                n=1,  # L1 norm
                dim=0  # Prune output channels
            )

    return model

# Benefits:
# - Actually reduces computation (unlike unstructured)
# - No need for sparse matrix operations
# - Works on any hardware

# Drawbacks:
# - Less fine-grained
# - May need higher pruning ratio for same compression
```

**3. Gradual Magnitude Pruning**
```python
# Iteratively prune during training

def iterative_prune(model, train_loader, epochs=10, final_sparsity=0.9):
    """
    Gradually increase sparsity during training
    """
    initial_sparsity = 0.0

    for epoch in range(epochs):
        # Calculate current target sparsity
        progress = epoch / epochs
        current_sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * progress

        # Prune
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=current_sparsity
                )

        # Train
        train_epoch(model, train_loader)

        # Remove pruning reparameterization (make permanent)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.remove(module, 'weight')

    return model

# Benefits:
# - Better accuracy than one-shot pruning
# - Model adapts to sparsity
# - Can achieve 90%+ sparsity with minimal accuracy loss
```

**4. Lottery Ticket Hypothesis**
```python
# Find "winning ticket" - subnetwork that can be trained from scratch

def find_winning_ticket(model, train_loader, prune_rate=0.2, rounds=5):
    """
    Iterative magnitude pruning to find winning ticket
    """
    # Save initial weights
    initial_weights = {name: param.clone()
                      for name, param in model.named_parameters()}

    for round in range(rounds):
        # Train
        train_full(model, train_loader)

        # Prune
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, 'weight', amount=prune_rate)

        # Reset to initial weights (but keep mask)
        for name, param in model.named_parameters():
            if 'weight' in name and name.replace('_orig', '') in initial_weights:
                param.data = initial_weights[name.replace('_orig', '')]

    return model

# Result: Sparse network that trains as well as dense network!
```

### Pruning in Practice

**Complete Pruning Pipeline:**
```python
import torch
import torch.nn.utils.prune as prune

class PruningPipeline:
    def __init__(self, model, target_sparsity=0.8):
        self.model = model
        self.target_sparsity = target_sparsity

    def iterative_prune_and_finetune(self, train_loader, val_loader, steps=10):
        """
        Gradually prune and fine-tune
        """
        best_acc = 0
        sparsity_step = self.target_sparsity / steps

        for step in range(steps):
            current_sparsity = sparsity_step * (step + 1)

            # Prune
            self.apply_pruning(current_sparsity)

            # Fine-tune
            print(f"Step {step+1}: Sparsity {current_sparsity:.1%}")
            acc = self.finetune(train_loader, val_loader, epochs=3)

            if acc > best_acc:
                best_acc = acc
                self.save_checkpoint(f'pruned_{current_sparsity:.0%}.pth')

            print(f"Accuracy: {acc:.2%}, Best: {best_acc:.2%}\n")

        return self.model

    def apply_pruning(self, amount):
        """Apply magnitude pruning"""
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                prune.l1_unstructured(module, name='weight', amount=amount)

    def finetune(self, train_loader, val_loader, epochs=3):
        """Fine-tune pruned model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            # Training
            self.model.train()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

        # Validation
        return self.evaluate(val_loader)

    def evaluate(self, val_loader):
        """Evaluate model"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        return correct / total

    def make_permanent(self):
        """Remove pruning reparameterization"""
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass
        return self.model

# Usage
pipeline = PruningPipeline(model, target_sparsity=0.9)
pruned_model = pipeline.iterative_prune_and_finetune(train_loader, val_loader)
pruned_model = pipeline.make_permanent()

# Result: 90% sparse model with minimal accuracy loss
```

---

## 🎓 Knowledge Distillation

### What is Knowledge Distillation?

**Concept:** Train small "student" model to mimic large "teacher" model

```
Teacher Model               Student Model
(Large, Accurate)          (Small, Fast)
     ↓                           ↓
Predictions              Learns from teacher
  [0.7, 0.2, 0.1]   →    [0.65, 0.25, 0.1]

Key: Student learns soft targets (probabilities)
Not just hard labels (0, 1, 0)
```

### Basic Distillation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        """
        alpha: Weight for distillation loss vs. ground truth loss
        temperature: Soften probability distributions
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, labels):
        # Distillation loss (KL divergence)
        distill_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Ground truth loss (cross entropy)
        student_loss = F.cross_entropy(student_logits, labels)

        # Combine
        total_loss = (
            self.alpha * distill_loss +
            (1 - self.alpha) * student_loss
        )

        return total_loss

# Training loop
def distill_model(teacher, student, train_loader, epochs=10):
    teacher.eval()  # Teacher in eval mode
    student.train()

    criterion = DistillationLoss(alpha=0.7, temperature=3.0)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for data, labels in train_loader:
            # Get teacher predictions (no gradient)
            with torch.no_grad():
                teacher_logits = teacher(data)

            # Get student predictions
            student_logits = student(data)

            # Compute loss
            loss = criterion(student_logits, teacher_logits, labels)

            # Update student
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return student

# Example: Distill ResNet50 → MobileNet
teacher = models.resnet50(pretrained=True)
student = models.mobilenet_v2(pretrained=False)

student_distilled = distill_model(teacher, student, train_loader)
```

### Advanced Distillation Techniques

**1. Feature-Based Distillation**
```python
# Match intermediate features, not just outputs

class FeatureDistillation(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student

        # Hooks to extract features
        self.teacher_features = []
        self.student_features = []

        # Register hooks
        self.teacher.layer3.register_forward_hook(
            lambda m, i, o: self.teacher_features.append(o)
        )
        self.student.layer3.register_forward_hook(
            lambda m, i, o: self.student_features.append(o)
        )

    def forward(self, x, labels):
        # Forward pass
        teacher_out = self.teacher(x)
        student_out = self.student(x)

        # Output distillation
        output_loss = F.kl_div(
            F.log_softmax(student_out, dim=1),
            F.softmax(teacher_out.detach(), dim=1)
        )

        # Feature distillation
        teacher_feat = self.teacher_features[-1]
        student_feat = self.student_features[-1]

        # Match feature dimensions if needed
        if teacher_feat.shape != student_feat.shape:
            student_feat = F.adaptive_avg_pool2d(
                student_feat,
                teacher_feat.shape[-2:]
            )

        feature_loss = F.mse_loss(student_feat, teacher_feat.detach())

        # Ground truth loss
        gt_loss = F.cross_entropy(student_out, labels)

        # Combine
        total_loss = 0.5 * output_loss + 0.3 * feature_loss + 0.2 * gt_loss

        # Clear features
        self.teacher_features.clear()
        self.student_features.clear()

        return total_loss
```

**2. Self-Distillation**
```python
# Use model as its own teacher

def self_distillation(model, train_loader, epochs=10):
    """
    Train model to match its own predictions
    (with data augmentation for diversity)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for data, labels in train_loader:
            # Original augmentation
            data_aug1 = augment(data, strength='weak')

            # Strong augmentation
            data_aug2 = augment(data, strength='strong')

            # Teacher predictions (weak aug, no grad)
            model.eval()
            with torch.no_grad():
                teacher_logits = model(data_aug1)

            # Student predictions (strong aug, with grad)
            model.train()
            student_logits = model(data_aug2)

            # Distillation loss
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=1),
                F.softmax(teacher_logits, dim=1)
            )

            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
```

**3. Multi-Teacher Distillation**
```python
# Ensemble of teachers → One student

def multi_teacher_distillation(teachers, student, train_loader):
    """
    Distill knowledge from multiple teachers
    """
    # Set teachers to eval mode
    for teacher in teachers:
        teacher.eval()

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    for data, labels in train_loader:
        # Get predictions from all teachers
        teacher_logits_list = []
        with torch.no_grad():
            for teacher in teachers:
                logits = teacher(data)
                teacher_logits_list.append(F.softmax(logits, dim=1))

        # Average teacher predictions
        avg_teacher_probs = torch.stack(teacher_logits_list).mean(dim=0)

        # Student prediction
        student_logits = student(data)

        # Distillation loss
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            avg_teacher_probs,
            reduction='batchmean'
        )

        # Update student
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return student

# Example: 3 ResNets → 1 MobileNet
teachers = [
    models.resnet50(pretrained=True),
    models.resnet101(pretrained=True),
    models.resnext50_32x4d(pretrained=True)
]
student = models.mobilenet_v2()

student = multi_teacher_distillation(teachers, student, train_loader)
```

---

## 🏗️ Architecture Optimization

### Efficient Architectures

**1. MobileNet: Depthwise Separable Convolutions**
```python
# Standard Convolution: H×W×C → H×W×C'
# Params: C × C' × K × K
# FLOPs: H × W × C × C' × K × K

# Depthwise Separable = Depthwise + Pointwise
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        # Depthwise: Each channel processed separately
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            groups=in_channels,  # Key: groups = in_channels
            padding=kernel_size//2
        )

        # Pointwise: 1×1 convolution to mix channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Computation Reduction:
# Standard: C × C' × K × K
# Depthwise Sep: C × K × K + C × C'
# Ratio: (C × K × K + C × C') / (C × C' × K × K)
#      ≈ 1/C' + 1/K²
# For K=3, C'=256: ~8-9x reduction!
```

**2. EfficientNet: Compound Scaling**
```python
# Scale depth, width, resolution together

def compound_scaling(base_model, phi=1.0):
    """
    phi: Compound coefficient

    depth = α^phi
    width = β^phi
    resolution = γ^phi

    subject to: α × β² × γ² ≈ 2
    """
    alpha = 1.2  # Depth scaling
    beta = 1.1   # Width scaling
    gamma = 1.15 # Resolution scaling

    depth_mult = alpha ** phi
    width_mult = beta ** phi
    resolution_mult = gamma ** phi

    # Scale base model
    scaled_model = scale_model(
        base_model,
        depth_multiplier=depth_mult,
        width_multiplier=width_mult,
        resolution=int(224 * resolution_mult)
    )

    return scaled_model

# EfficientNet-B0: Baseline
# EfficientNet-B7: phi=2.0 (much larger, more accurate)
```

**3. Neural Architecture Search (NAS)**
```python
# Automatically find efficient architectures

# Simplified NAS example
def search_architecture(search_space, train_loader, val_loader):
    """
    Search for best architecture in given space
    """
    best_arch = None
    best_score = 0

    for arch_config in search_space:
        # Build model from config
        model = build_from_config(arch_config)

        # Train briefly
        train_quick(model, train_loader, epochs=5)

        # Evaluate
        acc = evaluate(model, val_loader)

        # Score = accuracy / FLOPs (efficiency)
        flops = count_flops(model)
        score = acc / (flops / 1e9)  # Normalize

        if score > best_score:
            best_score = score
            best_arch = arch_config

    # Train best architecture fully
    final_model = build_from_config(best_arch)
    train_full(final_model, train_loader)

    return final_model
```

### Model Compression

**1. Low-Rank Factorization**
```python
# Decompose weight matrix into smaller matrices

def low_rank_decomposition(layer, rank):
    """
    W (m×n) ≈ U (m×r) × V (r×n)

    Params: m×n → m×r + r×n
    Reduction: (m×r + r×n) / (m×n) = r(m+n) / (m×n)
    """
    W = layer.weight.data
    m, n = W.shape

    # SVD
    U, S, V = torch.svd(W)

    # Keep top-r singular values
    U_r = U[:, :rank]
    S_r = torch.diag(S[:rank])
    V_r = V[:, :rank].t()

    # Create two smaller layers
    layer1 = nn.Linear(n, rank, bias=False)
    layer2 = nn.Linear(rank, m, bias=layer.bias is not None)

    layer1.weight.data = (V_r @ torch.sqrt(S_r)).t()
    layer2.weight.data = torch.sqrt(S_r) @ U_r.t()

    if layer.bias is not None:
        layer2.bias.data = layer.bias.data

    return nn.Sequential(layer1, layer2)

# Example: Compress large FC layer
large_layer = nn.Linear(1024, 1024)  # 1M params
compressed = low_rank_decomposition(large_layer, rank=128)
# (1024×128) + (128×1024) = 262K params (4x reduction)
```

---

## ⚡ Inference Optimization

### Model Export Formats

**1. ONNX (Open Neural Network Exchange)**
```python
import torch
import onnx
import onnxruntime as ort

# Export PyTorch model to ONNX
def export_to_onnx(model, input_shape, filepath='model.onnx'):
    dummy_input = torch.randn(input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,  # Optimize
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    # Verify
    onnx_model = onnx.load(filepath)
    onnx.checker.check_model(onnx_model)
    print(f"Exported to {filepath}")

# Inference with ONNX Runtime (faster!)
def onnx_inference(onnx_path, input_data):
    session = ort.InferenceSession(onnx_path)

    # Get input name
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(
        None,
        {input_name: input_data.numpy()}
    )

    return outputs[0]

# Benchmark
model = models.resnet50(pretrained=True)
model.eval()

# Export
export_to_onnx(model, (1, 3, 224, 224))

# Compare speed
import time

input_tensor = torch.randn(1, 3, 224, 224)

# PyTorch
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(input_tensor)
pytorch_time = time.time() - start

# ONNX
start = time.time()
for _ in range(100):
    _ = onnx_inference('model.onnx', input_tensor)
onnx_time = time.time() - start

print(f"PyTorch: {pytorch_time:.3f}s")
print(f"ONNX: {onnx_time:.3f}s")
print(f"Speedup: {pytorch_time/onnx_time:.2f}x")
# Typical: 1.5-2x speedup
```

**2. TensorRT (NVIDIA)**
```python
# Maximum GPU inference speed

import tensorrt as trt

def convert_to_tensorrt(onnx_path, engine_path):
    """
    Convert ONNX to TensorRT engine
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    # Configure
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)  # FP16 precision

    # Build engine
    engine = builder.build_engine(network, config)

    # Serialize
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"TensorRT engine saved to {engine_path}")

# Typical speedups: 2-5x over ONNX, 5-10x over PyTorch
```

**3. TFLite (Mobile/Edge)**
```python
import tensorflow as tf

def convert_to_tflite(model_path, output_path):
    """
    Convert to TensorFlow Lite for mobile/edge
    """
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Convert
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    # Convert
    tflite_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {output_path}")
```

### Operator Fusion

```python
# Fuse operations for efficiency

# Before: Conv → BatchNorm → ReLU (3 operations)
conv = nn.Conv2d(3, 64, 3)
bn = nn.BatchNorm2d(64)
relu = nn.ReLU()

# Fused: Single operation (faster!)
fused = torch.quantization.fuse_modules(
    model,
    [['conv', 'bn', 'relu']]
)

# Benefits:
# - Fewer memory reads/writes
# - Better cache utilization
# - 20-30% speedup
```

### Batching Strategies

```python
# Dynamic batching for variable-size inputs

class DynamicBatcher:
    def __init__(self, model, max_batch_size=32, max_wait_ms=50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []

    async def predict(self, input_data):
        """
        Buffer requests and batch them
        """
        # Add to queue
        future = asyncio.Future()
        self.queue.append((input_data, future))

        # If batch full or timeout, process
        if len(self.queue) >= self.max_batch_size:
            await self.process_batch()
        else:
            # Wait for more requests (up to max_wait_ms)
            await asyncio.sleep(self.max_wait_ms / 1000)
            if self.queue:
                await self.process_batch()

        return await future

    async def process_batch(self):
        """
        Process batched requests
        """
        if not self.queue:
            return

        # Get batch
        batch_data = [item[0] for item in self.queue]
        futures = [item[1] for item in self.queue]
        self.queue = []

        # Batch inference
        batch_tensor = torch.stack(batch_data)
        with torch.no_grad():
            batch_outputs = self.model(batch_tensor)

        # Distribute results
        for output, future in zip(batch_outputs, futures):
            future.set_result(output)

# Throughput improvement: 3-5x with batching
```

---

## 🖥️ Hardware Acceleration

### GPU Optimization

```python
# Optimize for GPU inference

# 1. Use mixed precision
from torch.cuda.amp import autocast

@autocast()
def forward_pass(model, data):
    return model(data)

# 2. Optimize memory layout
# Channels-first (NCHW) faster on GPU
data = data.permute(0, 3, 1, 2)  # NHWC → NCHW

# 3. Use fused operations
model = torch.jit.script(model)  # TorchScript

# 4. Optimize batch size
# Find max batch size that fits in GPU memory
def find_optimal_batch_size(model, input_shape):
    batch_size = 1
    while True:
        try:
            data = torch.randn(batch_size, *input_shape).cuda()
            _ = model(data)
            batch_size *= 2
        except RuntimeError:  # OOM
            return batch_size // 2
```

### CPU Optimization

```python
# Optimize for CPU inference

# 1. Use OpenMP threads
torch.set_num_threads(4)  # Physical cores

# 2. Use Intel MKL-DNN
# Automatically used if available

# 3. Quantize
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 4. Use ONNX Runtime
# 2-3x faster than PyTorch on CPU
```

---

## 📊 Benchmarking & Profiling

### Comprehensive Benchmarking

```python
import time
import torch
from torch.profiler import profile, ProfilerActivity

class ModelBenchmark:
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape
        self.device = next(model.parameters()).device

    def measure_latency(self, num_runs=100):
        """Measure inference latency"""
        dummy_input = torch.randn(self.input_shape).to(self.device)

        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)

        # Measure
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(num_runs):
            _ = self.model(dummy_input)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        end = time.time()

        avg_latency = (end - start) / num_runs * 1000  # ms
        return avg_latency

    def measure_throughput(self, batch_size=32, duration=10):
        """Measure throughput (samples/sec)"""
        dummy_input = torch.randn(batch_size, *self.input_shape[1:]).to(self.device)

        count = 0
        start = time.time()

        while time.time() - start < duration:
            _ = self.model(dummy_input)
            count += batch_size

        throughput = count / duration
        return throughput

    def count_parameters(self):
        """Count model parameters"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    def measure_memory(self):
        """Measure memory usage"""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            dummy_input = torch.randn(self.input_shape).to(self.device)
            _ = self.model(dummy_input)
            memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            return memory_mb
        else:
            # CPU memory harder to measure accurately
            return None

    def profile(self, num_steps=10):
        """Profile model execution"""
        dummy_input = torch.randn(self.input_shape).to(self.device)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            for _ in range(num_steps):
                _ = self.model(dummy_input)

        # Print summary
        print(prof.key_averages().table(
            sort_by="cuda_time_total" if self.device.type == 'cuda' else "cpu_time_total",
            row_limit=10
        ))

        return prof

    def full_benchmark(self):
        """Run all benchmarks"""
        print("=" * 60)
        print("MODEL BENCHMARK RESULTS")
        print("=" * 60)

        # Parameters
        params = self.count_parameters()
        print(f"\nParameters:")
        print(f"  Total: {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,}")
        print(f"  Model size: {params['total'] * 4 / 1024**2:.2f} MB (FP32)")

        # Latency
        latency = self.measure_latency()
        print(f"\nLatency:")
        print(f"  Average: {latency:.2f} ms")
        print(f"  FPS: {1000/latency:.1f}")

        # Throughput
        throughput = self.measure_throughput()
        print(f"\nThroughput:")
        print(f"  {throughput:.0f} samples/sec")

        # Memory
        memory = self.measure_memory()
        if memory:
            print(f"\nMemory:")
            print(f"  Peak: {memory:.2f} MB")

        # FLOPs
        from thop import profile as count_ops
        dummy_input = torch.randn(self.input_shape).to(self.device)
        flops, _ = count_ops(self.model, inputs=(dummy_input,))
        print(f"\nFLOPs:")
        print(f"  {flops/1e9:.2f} GFLOPs")

        print("=" * 60)

# Usage
model = models.resnet50(pretrained=True).cuda().eval()
benchmark = ModelBenchmark(model, (1, 3, 224, 224))
benchmark.full_benchmark()
```

---

## 🎯 Optimization Strategy

### Step-by-Step Guide

**1. Baseline Measurement**
```python
# Measure current performance
benchmark = ModelBenchmark(model, input_shape)
baseline_latency = benchmark.measure_latency()
baseline_size = benchmark.count_parameters()['total'] * 4 / 1024**2

print(f"Baseline: {baseline_latency:.2f}ms, {baseline_size:.2f}MB")
```

**2. Set Targets**
```
Target: 10x faster, 5x smaller, <1% accuracy loss

Allocation:
- Quantization: 4x smaller, 2x faster
- Pruning: 2x smaller, 1.5x faster
- Distillation: Better accuracy preservation
- Operator fusion: 1.5x faster
```

**3. Apply Techniques (in order)**
```python
# Step 1: Distillation (preserve accuracy)
student = smaller_architecture()
student = distill(teacher=model, student=student)

# Step 2: Pruning (reduce size/compute)
student = iterative_prune(student, sparsity=0.8)

# Step 3: Quantization (reduce precision)
student_q8 = quantize_model(student, dtype='int8')

# Step 4: Export (inference optimization)
export_to_onnx(student_q8, 'optimized.onnx')

# Step 5: Verify
final_latency = benchmark.measure_latency()
final_size = get_model_size(student_q8)
final_acc = evaluate(student_q8, test_loader)

print(f"Final: {final_latency:.2f}ms, {final_size:.2f}MB, {final_acc:.2%}")
print(f"Speedup: {baseline_latency/final_latency:.1f}x")
print(f"Compression: {baseline_size/final_size:.1f}x")
```

---

## 📚 Resources

**Tools:**
- PyTorch Model Optimization: pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- ONNX: onnx.ai
- TensorRT: developer.nvidia.com/tensorrt
- Neural Compressor: github.com/intel/neural-compressor

**Papers:**
- "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018)
- "The Lottery Ticket Hypothesis" (Frankle & Carbin, 2019)
- "EfficientNet: Rethinking Model Scaling" (Tan & Le, 2019)

---

**Remember: Optimization is about tradeoffs. Always measure, experiment, and validate on your specific use case!**
