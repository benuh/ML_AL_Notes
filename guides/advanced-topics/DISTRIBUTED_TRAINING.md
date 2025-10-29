# Distributed Training Guide

Scale ML training to multiple GPUs, nodes, and clusters for massive models and datasets.

## Table of Contents
1. [Why Distributed Training](#why-distributed-training)
2. [Data Parallelism](#data-parallelism)
3. [Model Parallelism](#model-parallelism)
4. [Advanced Techniques](#advanced-techniques)
5. [Frameworks and Tools](#frameworks-and-tools)
6. [Best Practices](#best-practices)

---

## Why Distributed Training

### When You Need Distributed Training

**Use distributed training when:**
1. **Model too large** for single GPU memory
2. **Dataset too large** - training takes too long
3. **Need faster iteration** - reduce training time from days to hours
4. **Scaling experiments** - test more hyperparameters

**Types of Parallelism:**
- **Data Parallelism** - Split data across GPUs, replicate model
- **Model Parallelism** - Split model across GPUs
- **Pipeline Parallelism** - Split model layers across GPUs, pipeline batches
- **Hybrid** - Combine multiple strategies

**Trade-offs:**
- **Communication overhead** - Syncing gradients/activations
- **Memory vs Computation** - More GPUs = more memory, but communication cost
- **Complexity** - Harder to debug

---

## Data Parallelism

### PyTorch DistributedDataParallel (DDP)

**Most common and efficient** for data parallelism in PyTorch.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """Initialize distributed training"""
    import os

    # Set up distributed process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # or 'gloo' for CPU
        rank=rank,
        world_size=world_size
    )

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_ddp(rank, world_size, model, train_dataset, epochs=10):
    """Training function for each GPU"""

    # Setup
    setup(rank, world_size)

    # Move model to GPU
    model = model.to(rank)

    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[rank])

    # Create distributed sampler
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # DataLoader with distributed sampler
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)

    # Training loop
    ddp_model.train()
    for epoch in range(epochs):
        # Set epoch for shuffling
        sampler.set_epoch(epoch)

        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        # Average loss across all processes
        avg_loss = epoch_loss / len(train_loader)

        if rank == 0:  # Only log from rank 0
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

        # Save checkpoint (only from rank 0)
        if rank == 0 and (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddp_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_epoch_{epoch}.pt')

    cleanup()

# Launch distributed training
def main():
    world_size = torch.cuda.device_count()  # Number of GPUs

    # Create model and dataset
    model = YourModel()
    train_dataset = YourDataset()

    # Spawn processes for each GPU
    mp.spawn(
        train_ddp,
        args=(world_size, model, train_dataset),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()
```

**Using torchrun (recommended):**

```python
# train_ddp.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # Initialize from environment variables set by torchrun
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create model
    model = YourModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Training loop
    # ... (same as above)

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

```bash
# Launch with torchrun
torchrun --nproc_per_node=4 train_ddp.py

# Multi-node training
torchrun --nproc_per_node=4 \
         --nnodes=2 \
         --node_rank=0 \
         --master_addr="192.168.1.1" \
         --master_port=12355 \
         train_ddp.py
```

---

### Gradient Accumulation

**Simulate larger batch sizes** with limited GPU memory.

```python
class GradientAccumulationTrainer:
    """Training with gradient accumulation"""

    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            # Forward pass
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)

            # Normalize loss (average over accumulation steps)
            loss = loss / self.accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps

        return total_loss / len(train_loader)

# Usage
trainer = GradientAccumulationTrainer(
    model=model,
    optimizer=optimizer,
    accumulation_steps=4  # Effective batch size = 4 * actual batch size
)

loss = trainer.train_epoch(train_loader)
```

---

### Mixed Precision Training

**Reduce memory and speed up training** with FP16/BF16.

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    """Training with mixed precision (FP16)"""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()  # Gradient scaler for FP16

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()

            self.optimizer.zero_grad()

            # Forward pass with autocast
            with autocast():
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)

            # Backward pass with scaled gradients
            self.scaler.scale(loss).backward()

            # Unscale gradients and update weights
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(train_loader)

# Usage
trainer = MixedPrecisionTrainer(model=model, optimizer=optimizer)
loss = trainer.train_epoch(train_loader)
```

---

## Model Parallelism

### Tensor Parallelism (Megatron-LM)

**Split tensors across GPUs** for large models.

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """Column-parallel linear layer"""

    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        self.rank = dist.get_rank()

        # Each GPU gets a slice of columns
        self.out_features_per_partition = out_features // world_size

        self.weight = nn.Parameter(
            torch.randn(self.out_features_per_partition, in_features)
        )
        self.bias = nn.Parameter(
            torch.randn(self.out_features_per_partition)
        )

    def forward(self, x):
        # Local matrix multiplication
        output_parallel = F.linear(x, self.weight, self.bias)

        # All-gather to combine results
        output_list = [torch.zeros_like(output_parallel) for _ in range(self.world_size)]
        dist.all_gather(output_list, output_parallel)

        # Concatenate along feature dimension
        output = torch.cat(output_list, dim=-1)

        return output

class RowParallelLinear(nn.Module):
    """Row-parallel linear layer"""

    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        self.rank = dist.get_rank()

        # Each GPU gets a slice of rows
        self.in_features_per_partition = in_features // world_size

        self.weight = nn.Parameter(
            torch.randn(out_features, self.in_features_per_partition)
        )
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # Split input across GPUs
        input_parallel = x.chunk(self.world_size, dim=-1)[self.rank]

        # Local matrix multiplication
        output_parallel = F.linear(input_parallel, self.weight)

        # All-reduce to sum partial results
        dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)

        # Add bias (only on one GPU)
        if self.rank == 0:
            output_parallel = output_parallel + self.bias

        return output_parallel

class TensorParallelTransformer(nn.Module):
    """Transformer with tensor parallelism"""

    def __init__(self, d_model, nhead, world_size):
        super().__init__()

        # Column-parallel for QKV
        self.qkv_proj = ColumnParallelLinear(d_model, 3 * d_model, world_size)

        # Row-parallel for output
        self.out_proj = RowParallelLinear(d_model, d_model, world_size)

        # Column-parallel for FFN
        self.fc1 = ColumnParallelLinear(d_model, 4 * d_model, world_size)
        self.fc2 = RowParallelLinear(4 * d_model, d_model, world_size)

    def forward(self, x):
        # Self-attention with tensor parallelism
        qkv = self.qkv_proj(x)
        # ... attention computation ...
        attn_output = self.out_proj(attn_output)

        # FFN with tensor parallelism
        ffn_output = self.fc1(attn_output)
        ffn_output = F.relu(ffn_output)
        ffn_output = self.fc2(ffn_output)

        return ffn_output
```

---

### Pipeline Parallelism

**Split model layers across GPUs**, pipeline batches.

```python
class PipelineParallel(nn.Module):
    """Simple pipeline parallelism"""

    def __init__(self, model_layers, num_gpus):
        super().__init__()
        self.num_gpus = num_gpus

        # Split layers across GPUs
        layers_per_gpu = len(model_layers) // num_gpus
        self.partitions = []

        for gpu_idx in range(num_gpus):
            start_idx = gpu_idx * layers_per_gpu
            end_idx = start_idx + layers_per_gpu if gpu_idx < num_gpus - 1 else len(model_layers)

            partition = nn.Sequential(*model_layers[start_idx:end_idx])
            partition = partition.to(f'cuda:{gpu_idx}')
            self.partitions.append(partition)

    def forward(self, x):
        # Sequential execution across GPUs
        for gpu_idx, partition in enumerate(self.partitions):
            x = x.to(f'cuda:{gpu_idx}')
            x = partition(x)

        return x

# Usage
layers = [nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10)]
model = PipelineParallel(layers, num_gpus=2)

# Forward pass moves data across GPUs
output = model(input_data)
```

**GPipe (Micro-batching for Pipeline):**

```python
class GPipe(nn.Module):
    """GPipe with micro-batching"""

    def __init__(self, model, num_gpus, chunks=4):
        super().__init__()
        self.model = model
        self.num_gpus = num_gpus
        self.chunks = chunks

    def forward(self, x):
        # Split batch into micro-batches
        micro_batches = x.chunk(self.chunks, dim=0)

        outputs = []
        for micro_batch in micro_batches:
            # Forward pass through pipeline
            output = self.model(micro_batch)
            outputs.append(output)

        # Concatenate results
        return torch.cat(outputs, dim=0)

# Training with GPipe
def train_gpipe(model, train_loader, optimizer):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass (pipelined)
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)

        # Backward pass (pipelined)
        loss.backward()
        optimizer.step()
```

---

## Advanced Techniques

### ZeRO (Zero Redundancy Optimizer)

**DeepSpeed's ZeRO** - Reduce memory by partitioning optimizer states, gradients, and parameters.

```python
import deepspeed

# DeepSpeed config
ds_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,  # ZeRO-2: Partition optimizer states + gradients
        "offload_optimizer": {
            "device": "cpu",  # Offload to CPU
            "pin_memory": True
        }
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
}

# Initialize DeepSpeed
model_engine, optimizer, train_loader, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
    training_data=train_dataset,
    config=ds_config
)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        loss = model_engine(batch)

        # Backward pass
        model_engine.backward(loss)

        # Optimizer step
        model_engine.step()
```

**ZeRO Stages:**
- **Stage 1**: Partition optimizer states (4x memory reduction)
- **Stage 2**: Partition optimizer states + gradients (8x memory reduction)
- **Stage 3**: Partition optimizer states + gradients + parameters (Linear memory scaling)

---

### Fully Sharded Data Parallel (FSDP)

**PyTorch's FSDP** - Similar to ZeRO-3, built into PyTorch.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

def train_fsdp(rank, world_size):
    # Setup
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Model
    model = YourLargeModel()

    # Auto-wrap policy (shard layers > 100M parameters)
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=100_000_000
    )

    # Wrap with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=True,
        device_id=rank
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(epochs):
        for batch in train_loader:
            data, target = batch[0].to(rank), batch[1].to(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

    # Save checkpoint
    if rank == 0:
        torch.save(model.state_dict(), 'fsdp_checkpoint.pt')

    dist.destroy_process_group()

# Launch
mp.spawn(train_fsdp, args=(world_size,), nprocs=world_size)
```

---

### 3D Parallelism (Data + Tensor + Pipeline)

**Combine all parallelism strategies** for massive models.

```python
class Parallel3D:
    """3D parallelism: Data + Tensor + Pipeline"""

    def __init__(self, model, data_parallel_size, tensor_parallel_size, pipeline_parallel_size):
        self.dp_size = data_parallel_size
        self.tp_size = tensor_parallel_size
        self.pp_size = pipeline_parallel_size

        # Create process groups
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Data parallel group
        self.dp_group = self._create_dp_group(rank)

        # Tensor parallel group
        self.tp_group = self._create_tp_group(rank)

        # Pipeline parallel group
        self.pp_group = self._create_pp_group(rank)

        # Wrap model with parallelism
        model = self._apply_tensor_parallelism(model, self.tp_group)
        model = self._apply_pipeline_parallelism(model, self.pp_group)
        model = DDP(model, process_group=self.dp_group)

        self.model = model

    def _create_dp_group(self, rank):
        # Create data parallel groups
        # ...
        pass

    def _create_tp_group(self, rank):
        # Create tensor parallel groups
        # ...
        pass

    def _create_pp_group(self, rank):
        # Create pipeline parallel groups
        # ...
        pass

# Example: 8 GPUs = 2 (DP) x 2 (TP) x 2 (PP)
parallel_3d = Parallel3D(
    model=gpt3_model,
    data_parallel_size=2,
    tensor_parallel_size=2,
    pipeline_parallel_size=2
)
```

---

## Frameworks and Tools

### Accelerate (HuggingFace)

**Simplify distributed training** with minimal code changes.

```python
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator(
    mixed_precision='fp16',
    gradient_accumulation_steps=4
)

# Prepare for distributed training
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

# Training loop (same as single GPU!)
for epoch in range(epochs):
    for batch in train_loader:
        with accelerator.accumulate(model):
            outputs = model(batch['input_ids'])
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

# Save model
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
accelerator.save(unwrapped_model.state_dict(), 'model.pt')
```

**Launch:**
```bash
accelerate launch --multi_gpu --num_processes=4 train.py
```

---

### Horovod

**Uber's distributed training framework** (TensorFlow, PyTorch, MXNet).

```python
import horovod.torch as hvd

# Initialize Horovod
hvd.init()

# Pin to local GPU
torch.cuda.set_device(hvd.local_rank())

# Build model
model = YourModel().cuda()

# Horovod: scale learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.01 * hvd.size())

# Horovod: wrap optimizer
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters()
)

# Horovod: broadcast initial parameters
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Training loop
for epoch in range(epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

# Launch with horovodrun
# horovodrun -np 4 -H localhost:4 python train.py
```

---

## Best Practices

### 1. Profiling and Bottleneck Analysis

```python
import torch.profiler as profiler

def profile_training():
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA
        ],
        schedule=profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=profiler.tensorboard_trace_handler('./log'),
        with_stack=True
    ) as prof:
        for step, (data, target) in enumerate(train_loader):
            if step >= (1 + 1 + 3):
                break

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            prof.step()

# Analyze with TensorBoard
# tensorboard --logdir=./log
```

---

### 2. Gradient Checkpointing

**Trade compute for memory** - recompute activations during backward pass.

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 1000)
        self.layer3 = nn.Linear(1000, 10)

    def forward(self, x):
        # Checkpoint layer1 and layer2
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = self.layer3(x)
        return x

# Reduces memory by ~50% but increases training time by ~20%
```

---

### 3. Communication Optimization

```python
class CommunicationOptimizer:
    """Optimize communication in distributed training"""

    @staticmethod
    def overlap_communication_computation(model, optimizer):
        """Overlap gradient communication with backward pass"""
        # Register gradient hooks
        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(lambda grad: dist.all_reduce(grad, async_op=True))

    @staticmethod
    def gradient_compression(gradients, compression_ratio=0.01):
        """Compress gradients before communication"""
        # Top-k sparsification
        k = int(compression_ratio * gradients.numel())
        _, indices = torch.topk(gradients.abs().view(-1), k)

        compressed = torch.zeros_like(gradients).view(-1)
        compressed[indices] = gradients.view(-1)[indices]

        return compressed.view_as(gradients)

    @staticmethod
    def hierarchical_all_reduce(tensor, world_size):
        """Hierarchical all-reduce for multi-node"""
        # Reduce within node
        node_rank = dist.get_rank() // 8  # Assuming 8 GPUs per node

        # Intra-node reduce
        # ...

        # Inter-node reduce
        # ...
```

---

## Summary

| Strategy | Memory Efficiency | Speed | Complexity | Use Case |
|----------|-------------------|-------|------------|----------|
| **DDP** | Low | High | Low | Standard multi-GPU |
| **FSDP/ZeRO-3** | Very High | Medium | Medium | Large models |
| **Tensor Parallel** | High | High | High | Very large layers |
| **Pipeline Parallel** | Medium | Medium | High | Very deep models |
| **3D Parallel** | Very High | High | Very High | Trillion parameter models |

---

## Key Takeaways

1. **Start with DDP** for standard multi-GPU training
2. **Use FSDP/ZeRO** for models that don't fit in single GPU
3. **Tensor parallelism** for models with huge layers (e.g., LLMs)
4. **Pipeline parallelism** for very deep models
5. **Mixed precision** reduces memory and speeds up training
6. **Gradient accumulation** simulates larger batch sizes
7. **Profile your training** to find bottlenecks
8. **Accelerate/DeepSpeed** simplify distributed training

**Common Issues:**
- **OOM errors**: Use FSDP, gradient checkpointing, or smaller batch size
- **Slow training**: Check GPU utilization, communication overhead
- **Hanging**: Ensure all processes reach collective operations
- **NaN losses**: Check learning rate, gradient clipping, mixed precision settings

**Next Steps:**
- Implement DDP for your model
- Profile training to identify bottlenecks
- Try FSDP for large models
- Experiment with mixed precision training
- Scale to multi-node clusters
