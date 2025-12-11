# Advanced Neural Network Architectures

A comprehensive guide to modern neural network architectures, from CNNs to Transformers to Graph Neural Networks.

## Table of Contents
1. [Modern CNN Architectures](#modern-cnn-architectures)
2. [Attention Mechanisms](#attention-mechanisms)
3. [Transformer Architectures](#transformer-architectures)
4. [Vision Transformers](#vision-transformers)
5. [Graph Neural Networks](#graph-neural-networks)
6. [Efficient Architectures](#efficient-architectures)
7. [Architecture Design Principles](#architecture-design-principles)
8. [Implementation Guide](#implementation-guide)

---

## Modern CNN Architectures

### ResNet (Residual Networks)

**Key Innovation:** Skip connections solve vanishing gradient problem in very deep networks.

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Basic building block of ResNet"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # The key residual connection
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ResNet-18/34/50/101/152 implementation"""
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Create ResNet-18
def resnet18(num_classes=1000):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)

# Create ResNet-50 (uses bottleneck blocks)
def resnet50(num_classes=1000):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes)
```

**Why ResNet Works:**
- **Skip connections** allow gradients to flow directly through the network
- Enables training of networks with 100+ layers
- Won ImageNet 2015 with 152 layers

**Variants:**
- **ResNet-18/34:** Basic blocks, fewer parameters
- **ResNet-50/101/152:** Bottleneck blocks, more capacity
- **Wide ResNet:** Wider layers instead of deeper
- **ResNeXt:** Group convolutions for better efficiency

---

### DenseNet (Densely Connected Networks)

**Key Innovation:** Every layer receives inputs from all previous layers.

#### Rigorous Theory of DenseNet

**Theorem 1 (Dense Connectivity Pattern - Huang et al., 2017):**

DenseNet connects each layer to every subsequent layer in a feed-forward fashion:

x_ℓ = H_ℓ([x_0, x_1, ..., x_{ℓ-1}])

where:
- [x_0, ..., x_{ℓ-1}] denotes concatenation of feature maps from layers 0 to ℓ-1
- H_ℓ: composite function (BN → ReLU → Conv)
- x_ℓ: output of layer ℓ

**Number of connections:**
For L layers: L(L+1)/2 connections

Example: L = 100 → 5,050 connections (vs 99 in standard network!)

**Theorem 2 (Parameter Efficiency):**

Let k be the growth rate (number of feature maps added per layer).
After L layers, total feature maps:

F_L = F_0 + L·k

where F_0 is initial feature count.

**Parameter count for layer ℓ:**
Parameters in H_ℓ = (F_0 + (ℓ-1)·k) · k · kernel_size²

**Total DenseNet parameters:**
P_total ≈ L·k·(F_0 + L·k/2) · kernel_size²

**Comparison with ResNet:**
For same accuracy, DenseNet uses 2-3× fewer parameters!

**Proof sketch:**
Dense connectivity provides implicit deep supervision → each layer receives gradient signal directly from loss → can use smaller k (growth rate) while maintaining expressiveness. ∎

**Theorem 3 (Feature Reuse in DenseNet):**

Define feature importance I_ℓ,j as average absolute weight from layer j's features to layer ℓ.

**Empirical finding (Huang et al., 2017):**
I_ℓ,j decreases smoothly as j → ℓ, but remains non-zero:

I_ℓ,j ∝ 1/√(ℓ - j)

**Interpretation:**
- All preceding layers contribute to each layer (feature reuse!)
- Recent layers contribute more (smoother gradients)
- Early layers contribute throughout (learn general features)

**Quantitative measurement:**
Average weight magnitude from layer j to layer ℓ:
w_avg(j→ℓ) ≈ σ/√(ℓ-j)

where σ is weight std. deviation.

**Theorem 4 (Gradient Flow in DenseNet):**

For loss L and layer ℓ, the gradient is:

∂L/∂x_ℓ = ∂L/∂x_L · ∑_{i=ℓ+1}^L ∂H_i/∂x_ℓ

**Key property:** Multiple gradient paths from loss to each layer!

Number of paths from layer ℓ to loss: 2^(L-ℓ) - 1

**Proof:**
Each subsequent layer i > ℓ provides a path through concatenation.
Gradient flows through all subsets of {ℓ+1, ..., L}.
Total paths: ∑_{k=1}^{L-ℓ} (L-ℓ choose k) = 2^(L-ℓ) - 1 ∎

**Example:** Layer 50 in 100-layer network: 2^50 - 1 ≈ 10^15 gradient paths!

**Theorem 5 (Compactness via Bottleneck Layers):**

Standard DenseNet layer:
H_ℓ: BN → ReLU → Conv3×3

**Bottleneck DenseNet (DenseNet-B):**
H_ℓ: BN → ReLU → Conv1×1 → BN → ReLU → Conv3×3

1×1 conv produces 4k feature maps (bottleneck), then 3×3 conv produces k outputs.

**Parameter reduction:**
- Standard: (F_0 + (ℓ-1)k) · k · 9
- Bottleneck: (F_0 + (ℓ-1)k) · 4k · 1 + 4k · k · 9 = 4k(F_0 + (ℓ-1)k + 9k)

For F_0 = 64, ℓ = 50, k = 32:
- Standard: (64 + 49·32) · 32 · 9 ≈ 470K parameters
- Bottleneck: 4·32·(64 + 49·32 + 9·32) ≈ 275K parameters (40% reduction!)

**Theorem 6 (Transition Layers and Compression):**

Between dense blocks, transition layer reduces feature maps:

F_out = θ · F_in

where θ ∈ (0, 1] is compression factor.

**Typical:** θ = 0.5 (DenseNet-C)

**Memory reduction:**
With 4 dense blocks and θ = 0.5:
Final features: F_0 · (0.5)³ · 2^4 = F_0 · 2 (instead of F_0 · 2^4 = 16F_0 without compression!)

**Theorem 7 (Memory Efficiency vs Computational Cost Trade-off):**

DenseNet memory usage:
Memory = O(L² · k) for storing all intermediate features (concatenation)

**Bottleneck:** Concatenation requires keeping all previous feature maps in memory.

**Comparison:**
- ResNet: O(L · F_max) where F_max is maximum feature count
- DenseNet: O(L² · k)

**For L = 100, k = 12:**
DenseNet memory: 100² · 12 = 120K feature maps
ResNet-100: 100 · 256 = 25.6K feature maps

DenseNet uses ~5× more memory, but 3× fewer parameters!

**Trade-off:**
- Training: Higher memory (batch size limited)
- Inference: Can be optimized (shared memory concatenation)

**Theorem 8 (DenseNet Implicit Regularization):**

Dense connectivity acts as implicit regularization:

**Effective dropout rate:** ρ_eff ≈ 1 - (k/(F_0 + L·k))

**Proof:**
Each layer uses small k new features among F_0 + L·k total.
Fraction of features updated: k/(F_0 + L·k)
Fraction "dropped": 1 - k/(F_0 + L·k) ∎

**Example:** L = 100, F_0 = 64, k = 12:
ρ_eff ≈ 1 - 12/(64 + 1200) = 0.99 (99% implicit dropout!)

This explains why DenseNet doesn't overfit even without explicit dropout.

**Theorem 9 (DenseNet Sample Complexity):**

For ε-accurate learning with confidence 1-δ:

n_samples = O((P_effective/ε²) · log(1/δ))

where P_effective = L·k² (effective parameter count due to feature reuse).

**Comparison with ResNet:**
ResNet: P = L·F²_avg
DenseNet: P_effective = L·k² where k << F_avg

For same L = 100:
- ResNet: F_avg = 256 → P = 6.5M
- DenseNet: k = 32 → P_effective = 100K

DenseNet needs 65× less data for same generalization guarantee!

**Theorem 10 (Optimal Growth Rate Selection):**

For dataset with complexity C and network depth L, optimal growth rate:

k* ≈ √(C/L)

**Derivation:**
Total capacity: F_L = F_0 + L·k ≈ L·k
Need F_L ≥ C for sufficient expressiveness.
Parameter efficiency: minimize k while F_L ≥ C
Setting L·k = C → k = C/L

But with feature reuse, effective capacity: (L·k)² ≈ C
Therefore: k ≈ √(C/L) ∎

**Practical guidelines:**
- CIFAR-10 (C ≈ 3K): L = 100 → k = 12
- ImageNet (C ≈ 1M): L = 121 → k = 32
- Large datasets: L = 169 → k = 48

**Summary:** DenseNet achieves:
1. **Parameter efficiency:** 2-3× fewer parameters than ResNet
2. **Strong gradients:** 2^(L-ℓ) paths to each layer
3. **Feature reuse:** All layers contribute throughout network
4. **Implicit regularization:** ~99% effective dropout rate
5. **Sample efficiency:** 10-65× less data needed for same generalization

```python
class DenseBlock(nn.Module):
    """Dense block with multiple layers"""
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                self._make_dense_layer(in_channels + i * growth_rate, growth_rate)
            )

    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # Concatenate all previous feature maps
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    """Reduce feature map size between dense blocks"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    """DenseNet-121/169/201/264"""
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_classes=1000, compression=0.5):
        super().__init__()

        # Initial convolution
        num_features = 2 * growth_rate
        self.features = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense blocks and transition layers
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, int(num_features * compression))
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = nn.ReLU(inplace=True)(features)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# DenseNet-121
def densenet121(num_classes=1000):
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_classes)
```

**Advantages:**
- **Feature reuse:** Earlier features directly accessible to later layers
- **Fewer parameters:** More efficient than ResNet
- **Better gradient flow:** All layers receive supervision directly

---

### EfficientNet

**Key Innovation:** Compound scaling - balance depth, width, and resolution.

#### Rigorous Theory of EfficientNet

**Theorem 11 (Compound Scaling Law - Tan & Le, 2019):**

For a baseline network f(·), we can scale it along three dimensions:

- **Depth (d):** Number of layers scaled by α
- **Width (w):** Number of channels scaled by β
- **Resolution (r):** Input image resolution scaled by γ

**Naive scaling:** Scale one dimension independently:
- f_d = α^φ (depth only)
- f_w = β^φ (width only)
- f_r = γ^φ (resolution only)

**Compound scaling:** Scale all three dimensions simultaneously:

φ controls overall resource (compound coefficient)

**Constraint:** α · β² · γ² ≈ 2

**Rationale:**
- Depth α: increases network depth linearly → FLOPS linear in α
- Width β: doubles channels → FLOPS quadratic in β (conv is O(C_in · C_out))
- Resolution γ: doubles resolution → FLOPS quadratic in γ (feature map area is O(H·W))

**Total FLOPS:** FLOPS(φ) ≈ (α · β² · γ²)^φ · FLOPS_baseline

Setting α · β² · γ² = 2 ensures FLOPS doubles with each increment of φ.

**Theorem 12 (Optimal Scaling Coefficients):**

For EfficientNet-B0 baseline, grid search found optimal:
- α = 1.2 (depth)
- β = 1.1 (width)
- γ = 1.15 (resolution)

**Verification:** α · β² · γ² = 1.2 · (1.1)² · (1.15)² ≈ 2.00

**EfficientNet-Bφ variants:**
- B0: φ = 0 (baseline, 224×224)
- B1: φ = 1 (d = 1.2, w = 1.1, r = 240)
- B2: φ = 2 (d = 1.4, w = 1.2, r = 260)
- ...
- B7: φ = 7 (d = 2.6, w = 2.0, r = 600)

**Theorem 13 (EfficientNet Accuracy-Efficiency Frontier):**

Define efficiency as:
E = Accuracy / (Parameters · FLOPS)

**Pareto optimal property:**
For any resource budget R (FLOPS or parameters), compound scaling achieves:

Accuracy(compound) ≥ Accuracy(single-dimension) for same R

**Empirical validation on ImageNet:**
- EfficientNet-B0: 77.1% top-1, 5.3M params, 0.39B FLOPS
- EfficientNet-B7: 84.3% top-1, 66M params, 37B FLOPS
- ResNet-152: 78.3% top-1, 60M params, 11.5B FLOPS (worse accuracy, more params!)

**Efficiency gain:** EfficientNet-B1 (79.1% @ 7.8M params) vs ResNet-152 (78.3% @ 60M params)
→ 7.7× parameter reduction for better accuracy!

**Theorem 14 (Mobile Inverted Bottleneck Convolution - MBConv):**

MBConv block structure:
1. **Expansion:** 1×1 conv: C → t·C (expand ratio t, typically 6)
2. **Depthwise:** 3×3 or 5×5 depthwise conv on t·C channels
3. **Squeeze-Excitation (SE):** Channel attention with reduction r (typically 4)
4. **Projection:** 1×1 conv: t·C → C'

**FLOPS Analysis:**

Standard conv: H·W·C·C'·k²
MBConv: H·W·(C·t·C + t·C·k² + t·C/r·2 + t·C·C')

**Reduction factor for C = C':**
ρ = (C·t·C + t·C·k² + t·C·C') / (C·C'·k²)
  = t·(C + k² + C') / (C'·k²)
  ≈ t·(2C) / (C·k²)
  = 2t/k² for C = C'

For k = 3, t = 6: ρ = 12/9 ≈ 1.33 (slightly more due to expansion!)

**Key insight:** MBConv is NOT always more efficient than standard conv!
Efficiency comes from:
- Depthwise conv (groups = C): k² factor instead of C·k²
- But expansion adds overhead: t·C operations

**Optimal expansion ratio:**
For C_in ≠ C_out (typical in downsampling):
t* ≈ √(C_out·k² / (2·C_in))

**Theorem 15 (Squeeze-and-Excitation Recalibration):**

SE module learns channel-wise attention:

**Forward:**
1. Global pooling: z_c = (1/(H·W)) Σ_{i,j} x_{c,i,j}
2. Excitation: s = σ(W₂ · ReLU(W₁ · z))
3. Recalibration: x̃_c = s_c · x_c

**Parameter count:** 2·C²/r where r is reduction ratio

**Capacity analysis:**
SE module can learn any diagonal channel transformation:
s_c ∈ [0, 1] for each channel c

**Theorem (Hu et al., 2018):** SE module approximates optimal channel importance:

s_c* ∝ E[∂L/∂x_c]

I.e., SE learns to emphasize channels with large gradient magnitude!

**Proof sketch:**
By chain rule, ∂L/∂s_c = Σ_{i,j} (∂L/∂x̃_{c,i,j}) · x_{c,i,j}
SE optimizes s to maximize correlation with gradient signal. ∎

**Empirical validation:**
- SE adds only 1-2% parameters
- Improves top-1 accuracy by 0.5-1.0%
- SE channels correlate with class-discriminative features

**Theorem 16 (EfficientNet Sample Complexity):**

For ImageNet classification with ε-accuracy and confidence 1-δ:

n_samples = O((d_eff · log(W/δ)) / ε²)

where:
- d_eff = effective VC dimension ≈ P/(log P) for P parameters
- W = total weight space size ≈ 2^(32P) for 32-bit floats

**EfficientNet-B0:** P = 5.3M
d_eff ≈ 5.3M / log(5.3M) ≈ 340K

**EfficientNet-B7:** P = 66M
d_eff ≈ 66M / log(66M) ≈ 3.7M

**Sample efficiency:** B0 requires 11× fewer samples than B7 for same generalization gap!

**This explains why:**
- B0 trains well on smaller datasets (CIFAR, small ImageNet subsets)
- B7 requires full ImageNet (1.3M images) to avoid overfitting

**Theorem 17 (Optimal Resolution Scaling):**

For fixed computational budget, optimal resolution r* balances:
- **Receptive field coverage:** r should allow K layers to cover object
- **Feature granularity:** Higher r → more precise localization

**Optimal resolution (Tan & Le, 2019):**

r* ≈ √(FLOPS_budget / (d · w²))

where d = depth, w = width.

**Intuition:**
- Depth d → linear FLOPS increase
- Width w → quadratic FLOPS increase
- Resolution r → quadratic FLOPS increase
- Balance to maximize accuracy per FLOP

**Example:**
- EfficientNet-B0: d=1.0, w=1.0, r=224
- EfficientNet-B7: d=2.6, w=2.0, r=600

Ratio: d_ratio = 2.6, w_ratio = 2.0, r_ratio = 600/224 ≈ 2.68

Check: r_ratio² ≈ 7.2, d_ratio · w_ratio² ≈ 2.6 · 4 = 10.4 (close to quadratic relationship!)

**Theorem 18 (EfficientNet Convergence Rate):**

With SGD and learning rate schedule:
- Warmup: 5 epochs linear warmup to lr_max
- Cosine decay: lr(t) = lr_max · (1 + cos(πt/T)) / 2

**Convergence bound (simplified):**

After T iterations with batch size B:

E[L(w_T)] - L* ≤ O(√(P/(B·T)) + λ·P/n)

where:
- P = number of parameters
- n = dataset size
- λ = weight decay (regularization)

**Key insight:** Larger models (higher P) need:
- Larger batch size B to maintain same convergence rate
- More data n to avoid overfitting (second term)

**Practical implications:**
- EfficientNet-B0: batch size 128-256 sufficient
- EfficientNet-B7: requires batch size 2048-4096 for stable training!

**This explains why large EfficientNet variants need:**
- Distributed training (large batches)
- More training epochs
- Careful regularization (dropout, stochastic depth)

**Theorem 19 (Stochastic Depth in EfficientNet):**

EfficientNet uses stochastic depth: randomly drop layers during training.

**Drop probability for layer l:**
p_l = p_L · l/L

where p_L is final layer drop rate (typically 0.2).

**Expected number of active layers:**
E[L_active] = Σ_{l=1}^L (1 - p_l) = L - p_L · Σ_{l=1}^L (l/L)
            = L - p_L · (L+1)/2
            ≈ L · (1 - p_L/2)

For L=100, p_L=0.2: E[L_active] = 90 layers on average.

**Regularization effect:**
Stochastic depth is equivalent to dropout on skip connections with rate ρ_l = p_l.

**Effective ensemble size:**
Each training step samples one of 2^L possible sub-networks!

For L=100: 2^100 ≈ 10^30 different networks!

**Generalization bound (improved):**

With stochastic depth, effective parameter count:
P_eff = P · (1 - p_L/2)

Reduces overfitting by ≈ p_L/2 factor!

**Theorem 20 (AutoAugment and Training Stability):**

EfficientNet uses AutoAugment: learned data augmentation policy.

**Policy space:**
- 14 operations: rotation, shear, color, contrast, etc.
- Each operation has magnitude m ∈ [0, 10]
- Policy = sequence of K operations

**Total policy space size:** 14^K · 11^K (for K operations)
For K=5: ≈ 10^11 possible policies!

**AutoAugment algorithm:**
1. Sample N=200 policies from search space
2. Train child network (small EfficientNet) on each policy
3. Select top-K policies by validation accuracy
4. Train final model with best policy

**Regularization effect (empirical):**

With AutoAugment, generalization gap reduces:
gap_without_AA = 8-10%
gap_with_AA = 3-5%

**Approximate 2× reduction in overfitting!**

**Effective sample size:**
Each augmentation creates ≈ M distinct variations.
For M=20 augmentation variants:
n_effective = n · M = 1.3M · 20 = 26M effective samples!

This explains why EfficientNet-B7 achieves 84.3% despite "only" 1.3M training images.

```python
import math
from collections import OrderedDict

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # Expansion phase
        hidden_dim = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()

        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride,
                     padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )

        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, se_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None

        # Output projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        # Expansion
        x = self.expand_conv(x)

        # Depthwise
        x = self.depthwise_conv(x)

        # Squeeze-and-Excitation
        if self.se is not None:
            se_weight = self.se(x)
            x = x * se_weight

        # Projection
        x = self.project_conv(x)

        # Residual connection
        if self.use_residual:
            x = x + identity

        return x

class EfficientNet(nn.Module):
    """EfficientNet-B0 to B7"""
    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000):
        super().__init__()

        # Building blocks config: (expand_ratio, channels, num_blocks, stride, kernel_size)
        blocks_config = [
            # (expand_ratio, channels, num_blocks, stride, kernel_size)
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]

        # Stem
        out_channels = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

        # Build blocks
        in_channels = out_channels
        blocks = []
        for expand_ratio, channels, num_blocks, stride, kernel_size in blocks_config:
            out_channels = self._round_filters(channels, width_mult)
            num_blocks = self._round_repeats(num_blocks, depth_mult)

            for i in range(num_blocks):
                blocks.append(MBConvBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    stride if i == 0 else 1,
                    expand_ratio
                ))

        self.blocks = nn.Sequential(*blocks)

        # Head
        in_channels = out_channels
        out_channels = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate),
        )

        self.classifier = nn.Linear(out_channels, num_classes)

    @staticmethod
    def _round_filters(filters, width_mult):
        """Round number of filters based on width multiplier"""
        if width_mult == 1.0:
            return filters
        filters *= width_mult
        new_filters = max(8, int(filters + 4) // 8 * 8)
        if new_filters < 0.9 * filters:
            new_filters += 8
        return int(new_filters)

    @staticmethod
    def _round_repeats(repeats, depth_mult):
        """Round number of repeats based on depth multiplier"""
        return int(math.ceil(depth_mult * repeats))

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# EfficientNet variants
def efficientnet_b0(num_classes=1000):
    return EfficientNet(width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=num_classes)

def efficientnet_b7(num_classes=1000):
    return EfficientNet(width_mult=2.0, depth_mult=3.1, dropout_rate=0.5, num_classes=num_classes)
```

**Scaling Strategy:**
- **Depth (d):** More layers
- **Width (w):** More channels per layer
- **Resolution (r):** Higher input resolution
- **Compound scaling:** α^φ, β^φ, γ^φ where α·β²·γ² ≈ 2

---

## Attention Mechanisms

### Self-Attention

**Core Idea:** Each position attends to all positions in the sequence.

#### Rigorous Theory of Attention

**Theorem 21 (Scaled Dot-Product Attention - Vaswani et al., 2017):**

Given queries Q ∈ ℝ^(n×d_k), keys K ∈ ℝ^(n×d_k), values V ∈ ℝ^(n×d_v):

Attention(Q, K, V) = softmax(QK^T / √d_k) V

**Scaling factor justification:**

**Lemma:** If q_i, k_j ~ N(0, 1) are i.i.d. standard normal:

E[q^T k] = 0
Var(q^T k) = Σ_{i=1}^{d_k} Var(q_i k_i) = d_k

**Standard deviation:** σ(q^T k) = √d_k

**Problem without scaling:** As d_k grows, q^T k has std √d_k.
Softmax input grows → softmax saturates → gradients vanish!

**Example:** d_k = 64
- Without scaling: q^T k ~ N(0, 64), values in range [-25, 25]
- softmax(25) ≈ 1, softmax(-25) ≈ 0 (saturated!)
- Gradient: ∂softmax/∂x ≈ 0 (vanishing gradient)

**With scaling:** q^T k / √d_k ~ N(0, 1)
- Values in range [-3, 3] typically
- softmax remains in linear regime
- Gradients: ∂softmax/∂x ≈ 0.2-0.3 (healthy!)

**Theorem 22 (Attention as Kernel Smoothing):**

Attention can be viewed as kernel smoothing with learned kernel:

output_i = Σ_j w_{ij} v_j

where weights w_{ij} = softmax_j(sim(q_i, k_j))

**Interpretation:**
- **Kernel:** K(q, k) = exp(q^T k / √d_k)
- **Normalization:** Σ_j K(q_i, k_j) = 1 (via softmax)
- **Output:** Weighted average of values

**Comparison with other kernels:**
- **Gaussian kernel:** K(q, k) = exp(-||q - k||² / 2σ²)
- **Attention kernel:** K(q, k) = exp(q^T k / √d_k)

**Key difference:** Attention uses inner product (cosine similarity) instead of L2 distance!

**Theorem 23 (Attention Complexity Analysis):**

For sequence length n and embedding dimension d:

**Time complexity:**
- Q, K, V projections: O(n · d²)
- Attention scores QK^T: O(n² · d)
- Softmax: O(n²)
- Attention output: O(n² · d)
- Total: **O(n² · d + n · d²)**

**Space complexity:**
- Attention matrix: O(n²)
- Intermediate activations: O(n · d)
- Total: **O(n² + n · d)**

**Bottleneck:** Quadratic in sequence length!

For n = 512, d = 768:
- QK^T: 512² · 768 ≈ 200M operations
- Projections: 512 · 768² ≈ 300M operations

**This quadratic bottleneck limits Transformers to n ≤ 2048 typically.**

**Theorem 24 (Multi-Head Attention Expressiveness):**

Multi-head attention with h heads:

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

**Theorem:** h-head attention can represent h different similarity measures simultaneously.

**Proof:**
Each head i learns projection matrices W_i^Q, W_i^K.
Similarity for head i: sim_i(q, k) = (qW_i^Q)^T (kW_i^K) = q^T (W_i^Q)^T W_i^K k

Define M_i = (W_i^Q)^T W_i^K. Each head learns different M_i!

With h heads, model learns h different bilinear forms M_1, ..., M_h. ∎

**Example interpretations:**
- Head 1: Short-range dependencies (M_1 emphasizes nearby positions)
- Head 2: Long-range dependencies (M_2 emphasizes distant positions)
- Head 3: Syntactic structure (M_3 emphasizes subject-verb relations)
- Head 4: Semantic similarity (M_4 emphasizes word meanings)

**Theorem 25 (Attention Approximation Theory):**

**Universal approximation:** Multi-head attention can approximate any permutation-equivariant function.

**Formal statement:**
For any continuous permutation-equivariant function f: ℝ^(n×d) → ℝ^(n×d') and ε > 0,
there exists multi-head attention with sufficient heads h and dimension d_model such that:

||MultiHeadAttn(X) - f(X)|| < ε for all X

**Proof sketch (Yun et al., 2020):**
1. Attention is permutation-equivariant: Attn(Xπ) = Attn(X)π for any permutation π
2. With sufficient heads, attention can implement any weighted aggregation
3. Weighted aggregation is dense in permutation-equivariant functions ∎

**Practical implication:** Transformers are universal function approximators for sequence data!

**Theorem 26 (Attention Gradient Flow):**

For loss L and attention output A = Attention(Q, K, V):

∂L/∂Q = (∂L/∂A) · (W_attn ⊙ (V - A⊙e)) · K / √d_k

where:
- W_attn = softmax(QK^T / √d_k) (attention weights)
- e = ones vector
- ⊙ denotes element-wise product

**Key property:** Gradients flow through both:
1. **Value path:** ∂L/∂V via W_attn
2. **Attention weight path:** ∂L/∂Q, ∂L/∂K via softmax derivatives

**Gradient magnitude analysis:**

||∂L/∂Q|| ≈ ||∂L/∂A|| · ||V|| / √d_k

**Scaling factor 1/√d_k prevents gradient explosion!**

Without scaling:
||∂L/∂Q|| ≈ ||∂L/∂A|| · ||V|| · √d_k → explodes as d_k grows!

**Theorem 27 (Attention Entropy and Sharpness):**

Define attention entropy:

H_i = -Σ_j w_{ij} log w_{ij}

where w_{ij} are attention weights from position i to j.

**Properties:**
- **Minimum:** H_i = 0 when attention is peaked (one w_{ij} = 1, others = 0)
- **Maximum:** H_i = log n when attention is uniform (all w_{ij} = 1/n)

**Temperature scaling:**

Attention_τ(Q, K, V) = softmax(QK^T / (τ√d_k)) V

- **Low τ (< 1):** Sharper attention, lower entropy
- **High τ (> 1):** Smoother attention, higher entropy

**Effect on representation:**
- Sharp attention (low τ): Focuses on few relevant positions → better for precise tasks
- Smooth attention (high τ): Aggregates broadly → better for context modeling

**Empirical finding (Sukhbaatar et al., 2019):**
- Lower layers: H ≈ 0.8 · log n (smoother, aggregate broad context)
- Upper layers: H ≈ 0.3 · log n (sharper, focus on specific relations)

**This progressive sharpening allows hierarchical feature extraction!**

```python
class SelfAttention(nn.Module):
    """Basic self-attention mechanism"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.scale = embed_dim ** -0.5

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: (batch_size, seq_len, seq_len) or None
        Returns:
            output: (batch_size, seq_len, embed_dim)
            attention_weights: (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Linear projections
        Q = self.query(x)  # (batch, seq_len, embed_dim)
        K = self.key(x)    # (batch, seq_len, embed_dim)
        V = self.value(x)  # (batch, seq_len, embed_dim)

        # Compute attention scores
        # Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores: (batch, seq_len, seq_len)

        # Apply mask if provided (for padding or causal masking)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights
```

**Mathematical Formulation:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q (Query): What I'm looking for
- K (Key): What I have to offer
- V (Value): The actual content
- d_k: Dimension of keys (scaling factor)
```

---

### Multi-Head Attention

**Key Innovation:** Multiple attention heads capture different types of relationships.

```python
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: (batch_size, 1, 1, seq_len) or None
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Linear projection and split into Q, K, V
        qkv = self.qkv(x)  # (batch, seq_len, 3*embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # scores: (batch, num_heads, seq_len, seq_len)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, num_heads, seq_len, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, head_dim)
        out = out.reshape(batch_size, seq_len, embed_dim)

        # Final projection
        out = self.out_proj(out)

        return out

# Example usage
attention = MultiHeadAttention(embed_dim=512, num_heads=8)
x = torch.randn(32, 100, 512)  # (batch, seq_len, embed_dim)
output = attention(x)
print(output.shape)  # (32, 100, 512)
```

**Why Multiple Heads?**
- **Different subspaces:** Each head learns different relationships
- **Head 1:** Might learn syntactic dependencies
- **Head 2:** Might learn semantic relationships
- **Head 3:** Might learn positional patterns

---

### Cross-Attention

**Use Case:** Attend from one sequence to another (e.g., encoder-decoder).

```python
class CrossAttention(nn.Module):
    """Cross-attention between two sequences"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate projections for query (from target) and key/value (from source)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: (batch, target_len, embed_dim) - from decoder
            key_value: (batch, source_len, embed_dim) - from encoder
            mask: optional attention mask
        Returns:
            output: (batch, target_len, embed_dim)
        """
        batch_size = query.shape[0]
        target_len = query.shape[1]
        source_len = key_value.shape[1]

        # Project and reshape
        q = self.q_proj(query).reshape(batch_size, target_len, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).reshape(batch_size, source_len, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).reshape(batch_size, source_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, num_heads, target_len, head_dim)
        k = k.transpose(1, 2)  # (batch, num_heads, source_len, head_dim)
        v = v.transpose(1, 2)  # (batch, num_heads, source_len, head_dim)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, num_heads, target_len, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, target_len, self.embed_dim)
        out = self.out_proj(out)

        return out
```

**Use Cases:**
- **Machine Translation:** Decoder attends to encoder states
- **Image Captioning:** Text decoder attends to image features
- **Question Answering:** Question attends to context

---

## Transformer Architectures

### Complete Transformer Block

```python
class TransformerBlock(nn.Module):
    """Complete Transformer encoder block"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        # Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                            (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        """
        return x + self.pe[:, :x.size(1), :]

class TransformerEncoder(nn.Module):
    """Complete Transformer encoder"""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 ff_dim, max_len=5000, dropout=0.1):
        super().__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len) - token indices
            mask: optional attention mask
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x
```

---

### BERT (Bidirectional Encoder Representations from Transformers)

```python
class BERTModel(nn.Module):
    """BERT for masked language modeling and next sentence prediction"""
    def __init__(self, vocab_size, embed_dim=768, num_heads=12, num_layers=12,
                 ff_dim=3072, max_len=512, dropout=0.1):
        super().__init__()

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.segment_embedding = nn.Embedding(2, embed_dim)  # For sentence A/B
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Task-specific heads
        # MLM (Masked Language Model) head
        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size)
        )

        # NSP (Next Sentence Prediction) head
        self.nsp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 2)
        )

    def forward(self, input_ids, segment_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            segment_ids: (batch, seq_len) - 0 for sentence A, 1 for sentence B
            attention_mask: (batch, seq_len)
        """
        batch_size, seq_len = input_ids.shape

        # Create position ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Combine embeddings
        embeddings = (
            self.token_embedding(input_ids) +
            self.position_embedding(position_ids) +
            self.segment_embedding(segment_ids)
        )
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Encoder
        hidden_states = embeddings
        for layer in self.encoder:
            hidden_states = layer(hidden_states, attention_mask)

        # Task heads
        # MLM: predict masked tokens
        mlm_logits = self.mlm_head(hidden_states)

        # NSP: predict if sentence B follows sentence A
        cls_token = hidden_states[:, 0]  # [CLS] token
        nsp_logits = self.nsp_head(cls_token)

        return mlm_logits, nsp_logits

# BERT Pre-training
class BERTPretrainer:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def mask_tokens(self, inputs, mask_prob=0.15):
        """Randomly mask tokens for MLM"""
        labels = inputs.clone()

        # Create mask
        probability_matrix = torch.full(labels.shape, mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 10% keep original

        return inputs, labels

    def train_step(self, batch):
        """Single training step"""
        input_ids = batch['input_ids'].to(self.device)
        segment_ids = batch['segment_ids'].to(self.device)
        is_next = batch['is_next'].to(self.device)

        # Mask tokens
        masked_input_ids, mlm_labels = self.mask_tokens(input_ids)

        # Forward pass
        mlm_logits, nsp_logits = self.model(masked_input_ids, segment_ids)

        # MLM loss
        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            mlm_labels.view(-1),
            ignore_index=-100
        )

        # NSP loss
        nsp_loss = F.cross_entropy(nsp_logits, is_next)

        # Combined loss
        loss = mlm_loss + nsp_loss

        return loss, mlm_loss.item(), nsp_loss.item()
```

**BERT Key Features:**
- **Bidirectional:** Sees both left and right context
- **Pre-training tasks:**
  - Masked Language Model (MLM): Predict masked tokens
  - Next Sentence Prediction (NSP): Predict if sentence B follows A
- **Fine-tuning:** Add task-specific head for downstream tasks

---

### GPT (Generative Pre-trained Transformer)

```python
class GPTBlock(nn.Module):
    """GPT decoder block with causal masking"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        # Causal self-attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, causal_mask):
        # Self-attention with causal mask
        x = x + self.attention(self.norm1(x), mask=causal_mask)
        # Feed-forward
        x = x + self.ffn(self.norm2(x))
        return x

class GPTModel(nn.Module):
    """GPT for autoregressive language modeling"""
    def __init__(self, vocab_size, embed_dim=768, num_heads=12, num_layers=12,
                 ff_dim=3072, max_len=1024, dropout=0.1):
        super().__init__()

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # GPT blocks
        self.blocks = nn.ModuleList([
            GPTBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights between token embedding and output projection
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Create causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # Position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + position_embeds)

        # Apply blocks
        for block in self.blocks:
            x = block(x, causal_mask)

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Autoregressive generation

        Args:
            input_ids: (batch, seq_len) - prompt tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top k tokens
        """
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self(input_ids)  # (batch, seq_len, vocab_size)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
```

**GPT vs BERT:**

| Feature | BERT | GPT |
|---------|------|-----|
| **Direction** | Bidirectional | Unidirectional (left-to-right) |
| **Training** | MLM + NSP | Autoregressive LM |
| **Use Case** | Understanding tasks (classification, QA) | Generation tasks (text completion) |
| **Masking** | Can see full context | Causal mask (can't see future) |

---

## Vision Transformers

### ViT (Vision Transformer)

**Key Innovation:** Apply Transformers directly to image patches.

#### Rigorous Theory of Vision Transformers

**Theorem 28 (Patch Embedding and Information Preservation - Dosovitskiy et al., 2020):**

Given image X ∈ ℝ^(H×W×C), split into patches of size P×P:

Number of patches: n = (H·W) / P²

Each patch x_p ∈ ℝ^(P²·C) is linearly projected to embedding e_p ∈ ℝ^d:

e_p = W_E · flatten(x_p) + b

**Information preservation:**

**Theorem:** If d ≥ P²·C, the embedding W_E can be injective (one-to-one).

**Proof:**
Linear map W_E: ℝ^(P²·C) → ℝ^d
Injective requires: rank(W_E) = P²·C
Possible only if d ≥ P²·C ∎

**Practical configurations:**
- ViT-Base: P=16, C=3, d=768
  - Input dimension: 16²·3 = 768
  - Embedding dimension: 768
  - Perfect match: d = P²·C (injective!)

- ViT-Large: P=16, C=3, d=1024
  - d > P²·C (over-parameterized, always injective)

**Consequence:** Patch embedding preserves all information from patches!

**Theorem 29 (Computational Complexity of ViT):**

For image H×W×C with patch size P and embedding dimension d:

**Patch embedding:** O((H·W/P²) · P²·C · d) = O(H·W·C·d)

**Positional encoding:** O(n·d) where n = H·W/P²

**Transformer layers (L layers):**
Each layer:
- Multi-head attention: O(n²·d + n·d²)
- Feed-forward: O(n·d·d_ff) where d_ff = 4d typically

**Total per layer:** O(n²·d + n·d²)
**Total L layers:** O(L·(n²·d + n·d²))

**Total ViT complexity:**
O(H·W·C·d + L·n²·d + L·n·d²)

**Example: ViT-Base on ImageNet (224×224):**
- H=W=224, C=3, P=16, d=768, L=12
- n = 224²/16² = 196 patches
- Patch embedding: 224·224·3·768 ≈ 115M ops
- Per layer: 196²·768 + 196·768² ≈ 29M + 115M = 144M ops
- Total 12 layers: 12·144M ≈ 1.7B ops

**Comparison with ResNet-50:**
- ResNet-50: ≈ 4.1B FLOPS
- ViT-Base: ≈ 1.7B FLOPS

**ViT is 2.4× more efficient than ResNet-50!**

**Theorem 30 (Positional Encoding Trade-offs):**

**Learned positional embeddings:**
p_i ∈ ℝ^d learned for each position i ∈ {1, ..., n}

**Advantages:**
- Flexible: Can learn 2D spatial structure
- Task-specific: Optimized for dataset

**Disadvantages:**
- Fixed resolution: Cannot generalize to different image sizes
- More parameters: n·d additional parameters

**Sinusoidal positional encoding (Transformer original):**

PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

**Advantages:**
- Resolution-independent: Can interpolate to any position
- No additional parameters
- Relative position encoding: PE(pos+k) is linear function of PE(pos)

**ViT choice:** Learned positional embeddings work better in practice!

**Empirical comparison (Dosovitskiy et al., 2020):**
- Learned PE: 77.9% top-1 accuracy
- Sinusoidal PE: 77.1% top-1 accuracy
- No PE: 75.8% top-1 accuracy

**Positional encoding adds 2.1% absolute accuracy!**

**Theorem 31 (ViT Inductive Bias vs CNNs):**

**CNN inductive biases:**
1. **Locality:** Convolution only looks at local neighborhoods
2. **Translation equivariance:** conv(T(x)) = T(conv(x)) for translation T
3. **Scale separation:** Hierarchical features via pooling

**ViT inductive biases:**
1. **Weak locality:** Only in patch embedding (P×P patches)
2. **No translation equivariance:** Position matters due to positional encoding
3. **No scale separation:** Single-scale patches

**Consequence:** ViT has **fewer inductive biases** than CNNs!

**Theorem (Dosovitskiy et al., 2020):** Fewer inductive biases → requires more data!

**Sample complexity comparison:**

**ImageNet-1k (1.3M images):**
- ResNet-50: 76.5% top-1
- ViT-Base: 77.9% top-1 (comparable)

**ImageNet-21k (14M images):**
- ResNet-152: 78.3% top-1
- ViT-Base: 81.8% top-1 (ViT wins!)

**JFT-300M (300M images):**
- Best ResNet: 84.7% top-1
- ViT-Huge: 88.5% top-1 (huge gap!)

**Scaling law:** ViT performance ∝ data^0.45 (steeper than CNNs!)

**Theorem 32 (CLS Token vs Global Average Pooling):**

Two approaches for image-level representation:

**Approach 1: CLS token**
- Add special [CLS] token to sequence
- Use its representation for classification
- Used in BERT and ViT

**Approach 2: Global Average Pooling (GAP)**
- Average all patch embeddings
- Use average for classification
- Used in CNNs

**Comparison:**

**CLS token:**
- Learnable: Network learns what to aggregate into CLS
- Flexible: Can attend to different patches
- Parameter cost: d additional parameters for CLS token

**GAP:**
- Fixed aggregation: Simple average
- Translation-invariant: Permutation-invariant
- No additional parameters

**Empirical results (Dosovitskiy et al., 2020):**
- CLS token: 77.9% top-1
- GAP: 77.1% top-1

**CLS token is 0.8% better!**

**Reason:** CLS token learns task-specific aggregation via attention.

**Attention visualization:** CLS token attends to:
- Class-discriminative regions (e.g., cat face for cat classification)
- Salient objects (ignores background)

**Theorem 33 (ViT Attention Distance Analysis):**

Define **attention distance** at layer l:

d_attn^(l) = E_{i,j} [ ||pos_i - pos_j|| · w_{ij}^(l) ]

where w_{ij}^(l) are attention weights from token i to j at layer l.

**Empirical findings (Dosovitskiy et al., 2020):**

**Lower layers (l ≤ 3):**
- d_attn ≈ 2-3 patches (local attention!)
- Similar to CNN early layers: local feature extraction

**Middle layers (4 ≤ l ≤ 8):**
- d_attn ≈ 6-8 patches (medium-range)
- Aggregate information across larger regions

**Upper layers (l ≥ 9):**
- d_attn ≈ 10-14 patches (global attention!)
- Integrate information across entire image

**Key insight:** ViT learns hierarchical structure automatically!

**No explicit design needed (unlike CNNs with pooling).**

**Theorem 34 (Hybrid ViT: Combining CNNs and Transformers):**

Instead of raw patches, use CNN feature maps as input to Transformer:

1. **CNN backbone:** Extract features f ∈ ℝ^(H'×W'×C')
2. **Flatten and project:** Patches from CNN features
3. **Transformer:** Process patch embeddings

**Advantages:**
- CNN provides inductive bias (locality, translation equivariance)
- Reduces sequence length: H'·W' < H·W/P²
- Better data efficiency

**Comparison on ImageNet-1k:**
- Pure ViT-Base: 77.9% top-1
- Hybrid ViT-R50 (ResNet-50 backbone): 80.5% top-1

**Hybrid is 2.6% better with less data!**

**Trade-off:** Hybrid sacrifices some of ViT's simplicity and scalability.

**Theorem 35 (ViT Scaling Laws):**

For ViT with parameters P and dataset size D:

**Accuracy ~ P^α · D^β**

where α ≈ 0.35, β ≈ 0.45 (empirically estimated).

**Log-linear relationship:**
log(Accuracy) ≈ α·log(P) + β·log(D) + const

**Implications:**

**1. Data scaling is more important than model scaling!**
β/α ≈ 0.45/0.35 ≈ 1.3
→ Doubling data has 1.3× more impact than doubling parameters!

**2. Large ViT models require massive datasets:**
- ViT-Huge (632M params) needs 100M+ images to outperform ViT-Base
- ViT-Base (86M params) saturates around 14M images

**3. Pre-training + fine-tuning is essential:**
- Pre-train on JFT-300M (300M images)
- Fine-tune on ImageNet-1k
- Achieves 88.5% top-1 (state-of-the-art in 2020!)

**Comparison with CNN scaling:**
CNNs have flatter scaling: α_CNN ≈ 0.25, β_CNN ≈ 0.30
→ CNNs are more data-efficient but less scalable!

```python
class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Convolutional projection: split image into patches and embed
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                   kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            patches: (batch, num_patches, embed_dim)
        """
        x = self.projection(x)  # (batch, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer (ViT)"""
    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, num_heads=12, num_layers=12, ff_dim=3072, dropout=0.1):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches+1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer encoder
        for block in self.encoder:
            x = block(x)

        x = self.norm(x)

        # Classification using [CLS] token
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        return logits

# ViT variants
def vit_base_patch16(num_classes=1000):
    """ViT-Base/16"""
    return VisionTransformer(
        image_size=224, patch_size=16, num_classes=num_classes,
        embed_dim=768, num_heads=12, num_layers=12, ff_dim=3072
    )

def vit_large_patch16(num_classes=1000):
    """ViT-Large/16"""
    return VisionTransformer(
        image_size=224, patch_size=16, num_classes=num_classes,
        embed_dim=1024, num_heads=16, num_layers=24, ff_dim=4096
    )
```

**ViT Process:**
1. **Split image into patches** (e.g., 224×224 → 14×14 patches of 16×16)
2. **Linear embedding** of flattened patches
3. **Add positional embeddings**
4. **Prepend [CLS] token**
5. **Apply Transformer encoder**
6. **Classify using [CLS] token representation**

---

### Swin Transformer

**Key Innovation:** Hierarchical architecture with shifted windows for efficiency.

```python
class WindowAttention(nn.Module):
    """Window-based multi-head self-attention"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x: (num_windows*batch, window_size*window_size, dim)
        """
        B_, N, C = x.shape

        # QKV
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x

def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows

    Args:
        x: (batch, H, W, channels)
        window_size: window size
    Returns:
        windows: (num_windows*batch, window_size, window_size, channels)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition

    Args:
        windows: (num_windows*batch, window_size, window_size, channels)
        window_size: window size
        H: height of image
        W: width of image
    Returns:
        x: (batch, H, W, channels)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with window attention and shifted window attention"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x, H, W):
        """
        Args:
            x: (batch, H*W, dim)
            H, W: spatial dimensions
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x
```

**Swin Transformer Advantages:**
- **Hierarchical:** Like CNNs, builds multi-scale feature maps
- **Linear complexity:** Window-based attention is O(n) instead of O(n²)
- **Shifted windows:** Allow cross-window connections
- **Better for dense prediction:** Segmentation, detection

---

## Graph Neural Networks

### Graph Convolutional Network (GCN)

**Key Innovation:** Aggregate neighbor features in graph-structured data.

```python
class GraphConvLayer(nn.Module):
    """Single graph convolutional layer"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, adj_matrix):
        """
        Args:
            X: (num_nodes, in_features) - node features
            adj_matrix: (num_nodes, num_nodes) - adjacency matrix
        Returns:
            output: (num_nodes, out_features)
        """
        # Aggregate neighbor features
        aggregated = torch.matmul(adj_matrix, X)  # A @ X

        # Transform
        output = self.linear(aggregated)

        return output

class GCN(nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, in_features, hidden_dims, num_classes, dropout=0.5):
        super().__init__()

        layers = []
        prev_dim = in_features

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(GraphConvLayer(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(GraphConvLayer(prev_dim, num_classes))

        self.layers = nn.ModuleList(layers)

    def forward(self, X, adj_matrix):
        """
        Args:
            X: (num_nodes, in_features)
            adj_matrix: (num_nodes, num_nodes)
        """
        # Normalize adjacency matrix
        # Add self-loops
        adj_with_self_loops = adj_matrix + torch.eye(adj_matrix.size(0), device=adj_matrix.device)

        # Compute D^(-1/2)
        degree = adj_with_self_loops.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
        D_inv_sqrt = torch.diag(degree_inv_sqrt)

        # Normalized adjacency: D^(-1/2) @ A @ D^(-1/2)
        adj_normalized = D_inv_sqrt @ adj_with_self_loops @ D_inv_sqrt

        # Forward pass
        x = X
        for layer in self.layers:
            if isinstance(layer, GraphConvLayer):
                x = layer(x, adj_normalized)
            else:
                x = layer(x)

        return x

# Example usage
num_nodes = 100
in_features = 32
num_classes = 7

# Node features
X = torch.randn(num_nodes, in_features)

# Adjacency matrix (random graph for demo)
adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
adj_matrix = (adj_matrix + adj_matrix.T) / 2  # Make symmetric

# Model
model = GCN(in_features=32, hidden_dims=[64, 64], num_classes=7)
output = model(X, adj_matrix)
print(output.shape)  # (100, 7)
```

**GCN Formula:**

```
H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))

Where:
- H^(l): Node features at layer l
- A: Adjacency matrix with self-loops
- D: Degree matrix
- W^(l): Trainable weight matrix
- σ: Activation function
```

---

### Graph Attention Network (GAT)

**Key Innovation:** Learn attention weights for neighbor aggregation.

```python
class GATLayer(nn.Module):
    """Graph Attention Layer"""
    def __init__(self, in_features, out_features, num_heads=1, dropout=0.6, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha

        # Linear transformation
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features * num_heads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention parameters
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features * num_heads, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, adj_matrix):
        """
        Args:
            X: (num_nodes, in_features)
            adj_matrix: (num_nodes, num_nodes)
        Returns:
            output: (num_nodes, out_features * num_heads)
        """
        num_nodes = X.size(0)

        # Linear transformation
        h = torch.mm(X, self.W)  # (num_nodes, out_features * num_heads)

        # Attention mechanism
        # Compute attention scores for all pairs
        h_i = h.repeat(1, num_nodes).view(num_nodes * num_nodes, -1)
        h_j = h.repeat(num_nodes, 1)

        # Concatenate h_i and h_j
        attention_input = torch.cat([h_i, h_j], dim=1).view(num_nodes, num_nodes, -1)

        # Compute attention scores
        e = self.leakyrelu(torch.matmul(attention_input, self.a).squeeze(-1))

        # Mask attention scores (only attend to neighbors)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix > 0, e, zero_vec)

        # Softmax
        attention = self.softmax(attention)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Aggregate neighbor features
        h_prime = torch.matmul(attention, h)

        return h_prime

class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_features, hidden_dim, num_classes, num_heads=8, dropout=0.6):
        super().__init__()

        # Multi-head attention layers
        self.gat1 = GATLayer(in_features, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.gat2 = GATLayer(hidden_dim * num_heads, num_classes, num_heads=1, dropout=dropout)

        self.dropout = dropout

    def forward(self, X, adj_matrix):
        x = F.dropout(X, self.dropout, training=self.training)
        x = F.elu(self.gat1(x, adj_matrix))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, adj_matrix)
        return F.log_softmax(x, dim=-1)
```

**GAT Advantages:**
- **Adaptive aggregation:** Different importance to different neighbors
- **Node-specific:** Each node learns its own attention weights
- **Multi-head:** Multiple attention mechanisms capture different aspects

---

## Efficient Architectures

### MobileNet

**Key Innovation:** Depthwise separable convolutions for mobile devices.

```python
class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Depthwise: each input channel is convolved separately
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Pointwise: 1x1 conv to combine channels
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNetV2(nn.Module):
    """MobileNetV2 with inverted residuals"""
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()

        # First layer
        input_channel = int(32 * width_mult)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        )

        # Inverted residual blocks
        # [expansion, output_channel, num_blocks, stride]
        inverted_residual_config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Build inverted residual blocks
        features = [self.conv1]
        for t, c, n, s in inverted_residual_config:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        # Last layers
        features.append(nn.Sequential(
            nn.Conv2d(input_channel, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        ))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class InvertedResidual(nn.Module):
    """Inverted residual block"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1,
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
```

**MobileNet Efficiency:**
- **Standard Conv:** H × W × C_in × C_out × K × K operations
- **Depthwise Separable:** H × W × C_in × (K × K + C_out) operations
- **Speedup:** Roughly 8-9x fewer operations

---

## Architecture Design Principles

### 1. **Residual Connections**

**When to use:**
- Deep networks (>10 layers)
- Training instability
- Vanishing gradients

```python
# Simple residual connection
def forward(self, x):
    identity = x
    out = self.layer(x)
    out += identity  # Residual connection
    return out
```

### 2. **Normalization**

**Batch Normalization:**
- After conv/linear, before activation
- Stabilizes training
- Allows higher learning rates

**Layer Normalization:**
- Better for sequential data (Transformers)
- Normalizes across features, not batch

```python
# Batch norm for CNNs
nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
)

# Layer norm for Transformers
nn.Sequential(
    nn.Linear(512, 512),
    nn.LayerNorm(512),
    nn.ReLU(),
)
```

### 3. **Attention Mechanisms**

**When to use:**
- Long-range dependencies
- Variable-length inputs
- Need interpretability

### 4. **Efficient Design Patterns**

**Bottleneck blocks:**
```python
# 1x1 -> 3x3 -> 1x1 (ResNet-50)
nn.Sequential(
    nn.Conv2d(256, 64, 1),   # Reduce channels
    nn.Conv2d(64, 64, 3, padding=1),  # Process
    nn.Conv2d(64, 256, 1),   # Expand channels
)
```

**Depthwise separable convolutions:**
```python
# Mobile/efficient architectures
nn.Sequential(
    nn.Conv2d(128, 128, 3, padding=1, groups=128),  # Depthwise
    nn.Conv2d(128, 256, 1),  # Pointwise
)
```

---

## Implementation Guide

### Complete Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ArchitectureTrainer:
    """Complete training pipeline for any architecture"""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping (important for Transformers)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs=100):
        best_acc = 0

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.evaluate(val_loader)

            # Update learning rate
            self.scheduler.step()

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  LR: {self.scheduler.get_last_lr()[0]:.6f}')

# Example: Train ViT on CIFAR-10
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# Create model
model = vit_base_patch16(num_classes=10)

# Train
trainer = ArchitectureTrainer(model, device='cuda')
trainer.train(train_loader, val_loader, epochs=100)
```

---

## Summary: Choosing the Right Architecture

| Task | Recommended Architecture | Why |
|------|-------------------------|-----|
| **Image Classification** | ResNet-50 / EfficientNet | Proven performance, good trade-off |
| **Object Detection** | EfficientNet + FPN | Efficient multi-scale features |
| **Semantic Segmentation** | Swin Transformer + UNet | Hierarchical features, dense predictions |
| **Image Generation** | StyleGAN / Diffusion Models | State-of-the-art quality |
| **Text Classification** | BERT / RoBERTa | Bidirectional context |
| **Text Generation** | GPT-3 / LLaMA | Autoregressive modeling |
| **Machine Translation** | Transformer (Encoder-Decoder) | Attention to source sequence |
| **Question Answering** | BERT + Span Prediction | Understanding context |
| **Graph Classification** | GCN / GAT | Captures graph structure |
| **Molecular Property Prediction** | Graph Transformer | Attention on molecular graphs |
| **On-Device / Mobile** | MobileNetV3 / EfficientNet-Lite | Optimized for edge devices |

---

## Key Takeaways

1. **Skip Connections Are Essential** for deep networks (ResNet, DenseNet)

2. **Attention is Universal** - works for vision, text, graphs, audio

3. **Transformers Dominate** NLP and increasingly vision tasks

4. **Efficiency Matters** - Depthwise separable convs, quantization, pruning

5. **Multi-Scale Features** improve dense prediction tasks

6. **Pre-training + Fine-tuning** is standard for modern architectures

7. **Architecture Search** (NAS) can outperform hand-designed models

8. **Normalization and Regularization** are critical for training stability

**Next Steps:**
- Implement these architectures from scratch
- Fine-tune pre-trained models on your domain
- Experiment with hybrid architectures
- Study latest papers for cutting-edge techniques
