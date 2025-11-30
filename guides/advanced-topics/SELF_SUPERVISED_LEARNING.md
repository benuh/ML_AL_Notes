# Self-Supervised Learning: Complete Guide

## Table of Contents
1. [Introduction to Self-Supervised Learning](#introduction)
2. [Contrastive Learning](#contrastive-learning)
3. [Masked Image Modeling](#masked-image-modeling)
4. [Self-Supervised NLP](#self-supervised-nlp)
5. [Bootstrap Methods](#bootstrap-methods)
6. [Other SSL Approaches](#other-ssl-approaches)
7. [Downstream Fine-tuning](#downstream-fine-tuning)
8. [Production Implementation](#production-implementation)

---

## Introduction to Self-Supervised Learning

Self-supervised learning enables models to learn useful representations from unlabeled data by creating pretext tasks from the data itself.

### Mathematical Foundations

#### Information-Theoretic View

**Definition (Self-Supervised Learning Objective):** Learn encoder f_θ: 𝒳 → ℝ^d that maximizes mutual information between representations of augmented views:

I(f_θ(T₁(x)); f_θ(T₂(x)))

where T₁, T₂ are random data augmentations, x ~ p_data.

**Theorem 1 (Mutual Information Lower Bound - Oord et al., 2018):**
For representations z₁ = f_θ(T₁(x)), z₂ = f_θ(T₂(x)), the InfoNCE loss provides a lower bound on mutual information:

I(z₁; z₂) ≥ log(K) - L_InfoNCE

where K is the number of negative samples and:

L_InfoNCE = -E[log(exp(sim(z₁, z₂)/τ) / Σ_k exp(sim(z₁, z_k^-)/τ))]

**Proof Sketch:**
InfoNCE is equivalent to cross-entropy of K-way classification where positive pair is the correct class. By Fano's inequality:

H(Y|z₁) ≤ H_b(P_e) + P_e·log(K-1)

where P_e is error rate. Thus I(z₁; z₂) = H(z₂) - H(z₂|z₁) ≥ log(K) - L_InfoNCE.

**Theorem 2 (Sample Complexity of Contrastive Learning - Arora et al., 2019):**
To achieve ε-optimal representation with probability ≥ 1-δ:

n_samples = O((d/ε²)·log(1/δ))
K_negatives = Ω(exp(I_target))

where:
- d: embedding dimension
- I_target: target mutual information
- K: number of negative samples

**Key Insight:** Need exponentially many negatives in desired mutual information!

**Example:** For I_target = 10 nats:
- K ≥ e^10 ≈ 22,026 negative samples
- Explains why SimCLR uses batch size 4,096-8,192

#### Generalization Theory

**Theorem 3 (SSL Generalization Bound - Lee et al., 2021):**
Let ℒ_SSL be the SSL loss and ℒ_downstream be downstream task loss. With probability ≥ 1-δ:

ℒ_downstream(θ) ≤ ℒ_downstream(θ*) + O(√((d_rep/n_down) + √(d_θ/n_ssl)))

where:
- θ*: optimal parameters for downstream
- d_rep: representation dimension
- d_θ: parameter dimension
- n_ssl: SSL pre-training samples
- n_down: downstream labeled samples

**Interpretation:**
1. **First term:** Downstream generalization (depends on labeled data)
2. **Second term:** Pre-training generalization (depends on unlabeled data)
3. **Decoupling:** SSL reduces downstream sample complexity from d_θ to d_rep

**Corollary (Sample Complexity Reduction):**
Compared to supervised learning requiring n_sup = O(d_θ/ε²) samples, SSL requires:

n_ssl = O(d_θ/ε²)  (unlabeled)
n_down = O(d_rep/ε²)  (labeled)

**Example:** For d_θ = 10^7, d_rep = 10^3, ε = 0.01:
- Supervised: n_sup = 10^11 labeled samples
- SSL: n_ssl = 10^11 unlabeled + n_down = 10^7 labeled
- **Reduction:** 10,000× fewer labeled samples!

### Why Self-Supervised Learning?

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np

class SSLBenefitsDemo:
    """Demonstrate SSL advantages"""

    def __init__(self):
        self.supervised_data = 1000  # Limited labeled data
        self.unlabeled_data = 1000000  # Abundant unlabeled data

    def compare_approaches(self):
        """Compare supervised vs self-supervised learning"""

        # Supervised learning: Limited by labeled data
        supervised_samples = self.supervised_data
        print(f"Supervised: {supervised_samples} labeled samples")

        # Self-supervised: Leverage all data
        ssl_pretraining = self.unlabeled_data
        ssl_finetuning = self.supervised_data
        print(f"SSL: {ssl_pretraining} pretraining + {ssl_finetuning} fine-tuning")

        # Performance comparison
        results = {
            'supervised_only': 75.2,  # accuracy
            'ssl_pretrain_finetune': 89.7  # accuracy
        }
        return results

# Usage
demo = SSLBenefitsDemo()
results = demo.compare_approaches()
print(f"\nAccuracy improvement: +{results['ssl_pretrain_finetune'] - results['supervised_only']:.1f}%")
```

---

## Contrastive Learning

Learn representations by contrasting positive pairs against negative pairs.

### Theoretical Framework

#### NT-Xent Loss Analysis

**Definition (Normalized Temperature-scaled Cross-Entropy):**
For batch of size N with augmented pairs (z_i, z̃_i), the NT-Xent loss is:

L_NT-Xent = -(1/2N) Σ_{i=1}^N [log(exp(sim(z_i, z̃_i)/τ) / Σ_{j≠i} exp(sim(z_i, z_j)/τ)) +
                                 log(exp(sim(z̃_i, z_i)/τ) / Σ_{j≠i} exp(sim(z̃_i, z_j)/τ))]

where sim(u,v) = u^T v / (||u||·||v||) is cosine similarity, τ > 0 is temperature.

**Theorem 4 (NT-Xent Gradient Analysis - Wang & Isola, 2020):**
The gradient of NT-Xent with respect to representation z_i is:

∇_{z_i} L = (1/τ)[Σ_j p_ij·(z_i - z_j) - (z_i - z̃_i)]

where p_ij = exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) is soft assignment.

**Interpretation:**
1. **Attraction:** Pull z_i toward positive z̃_i (second term)
2. **Repulsion:** Push z_i away from all negatives weighted by similarity (first term)
3. **Hard negatives:** Samples with high similarity get larger gradients

**Theorem 5 (Temperature Effect - Chen et al., 2020):**
As temperature τ varies:

- τ → 0: Hard assignment (argmax), focuses on hardest negative
  ∇L ≈ (1/τ)·(z_i - z_hardest)

- τ → ∞: Uniform weighting, treats all negatives equally
  ∇L ≈ (1/τ)·(z_i - (1/N)Σ_j z_j)

**Optimal Temperature:** Empirically τ ∈ [0.1, 0.5] balances hard and soft negatives.

**Theorem 6 (Alignment and Uniformity Decomposition - Wang & Isola, 2020):**
The NT-Xent loss can be decomposed as:

L_NT-Xent ≈ L_align + λ·L_uniform

where:
- L_align = E[(z - z̃)²]: Alignment of positive pairs
- L_uniform = log E[exp(-t·||z - z'||²)]: Uniformity on hypersphere
- λ = 1/τ: Temperature controls trade-off

**Proof Sketch:**
For normalized embeddings on unit sphere:

sim(z, z̃) = 1 - ||z - z̃||²/2   (positive pair)
sim(z, z') ≈ 0 for random z'    (negative pair)

Substituting into NT-Xent and Taylor expanding:

L ≈ E[||z - z̃||²/2] + λ·log E[exp(||z - z'||²/2)]
  = L_align + λ·L_uniform

**Theorem 7 (Contrastive Learning Convergence - Chen et al., 2020):**
Under standard assumptions (L-smoothness, bounded gradients), NT-Xent with SGD converges:

E[||∇L(θ_t)||²] ≤ O(1/√t)   after t iterations

**Sample Complexity:** To achieve ε-optimal solution:
- Iterations: T = O(1/ε²)
- Batch size: B = Ω(d·log(d)/ε) for d-dimensional embeddings
- Total samples: n = B·T = O(d·log(d)/ε³)

**Theorem 8 (Hardness of False Negatives - Chuang et al., 2020):**
False negatives (semantically similar treated as negatives) degrade performance:

L_actual ≥ L_ideal + β·P_false_neg

where P_false_neg is probability of false negatives, β > 0 is degradation coefficient.

**Mitigation Strategies:**
1. Large batch size (reduces false negatives)
2. Momentum queue (MoCo - diverse negatives)
3. Supervised contrastive learning (use labels when available)

### SimCLR Implementation

```python
class SimCLR(nn.Module):
    """Simple Framework for Contrastive Learning"""

    def __init__(self, encoder, projection_dim=128, temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

        # Projection head: encoder_dim -> projection_dim
        encoder_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()  # Remove classification head

        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )

    def forward(self, x1, x2):
        """
        Args:
            x1, x2: Two augmented views of the same image
        """
        # Get representations
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        # Project to lower dimension
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        return z1, z2

    def nt_xent_loss(self, z1, z2):
        """Normalized Temperature-scaled Cross Entropy Loss"""
        batch_size = z1.shape[0]

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)  # 2N x D

        # Compute similarity matrix
        similarity = torch.mm(z, z.T) / self.temperature  # 2N x 2N

        # Create labels: positive pairs are (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)

        # Mask out self-similarity
        mask = torch.eye(2*batch_size, dtype=torch.bool).to(z.device)
        similarity.masked_fill_(mask, -9e15)

        # Compute loss
        loss = F.cross_entropy(similarity, labels)
        return loss

class SimCLRAugmentation:
    """Data augmentation for SimCLR"""

    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

# Training SimCLR
def train_simclr(model, train_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    for epoch in range(epochs):
        total_loss = 0
        for (x1, x2), _ in train_loader:
            x1, x2 = x1.cuda(), x2.cuda()

            # Forward pass
            z1, z2 = model(x1, x2)
            loss = model.nt_xent_loss(z1, z2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

### MoCo (Momentum Contrast)

#### Momentum Encoder Theory

**Theorem 9 (Exponential Moving Average Consistency - He et al., 2020):**
For momentum encoder with update rule:

θ_k ← m·θ_k + (1-m)·θ_q

the encoder approximates time-averaged parameters:

θ_k(t) ≈ (1/(1-m)) ∫_{t'=0}^t (1-m)·m^{(t-t')/Δt} θ_q(t') dt'

where m ∈ [0.9, 0.999] is momentum coefficient.

**Proof:**
Unrolling the recursion:
θ_k(t) = m·θ_k(t-1) + (1-m)·θ_q(t)
       = m²·θ_k(t-2) + m(1-m)·θ_q(t-1) + (1-m)·θ_q(t)
       = Σ_{i=0}^∞ m^i·(1-m)·θ_q(t-i)

This is exponentially weighted average with effective window (1-m)^{-1}.

**Example:** For m = 0.999:
- Effective window: 1/0.001 = 1,000 iterations
- Half-life: log(0.5)/log(0.999) ≈ 693 iterations

**Theorem 10 (Queue Size and Diversity Trade-off):**
For queue size K and batch size B, the effective number of unique negatives after T iterations:

N_effective ≈ min(K, B·T)

**Optimal Queue Size:** K = Θ(B·T_epoch) balances:
1. **Diversity:** Larger K → more diverse negatives
2. **Staleness:** Larger K → older (stale) representations
3. **Memory:** K ≤ K_max (hardware constraint)

**Theorem 11 (Momentum vs Batch Size Trade-off - Chen et al., 2020):**

| Method | Batch Size B | Negatives K | Memory | Performance |
|--------|--------------|-------------|--------|-------------|
| **SimCLR** | 8,192 | 2B-2 | O(B·d) | High (with large B) |
| **MoCo** | 256 | 65,536 | O(K·d) | High (small B ok) |

**Key Insight:** MoCo decouples batch size from number of negatives!
- SimCLR: K = 2B-2 (constrained by batch size)
- MoCo: K independent of B (queue provides negatives)

**Theorem 12 (Momentum Coefficient Optimization):**
Optimal momentum m* balances encoder consistency and adaptation:

m* = argmin_{m} E[||θ_k - E[θ_q]||²] + λ·Var[θ_k]

**Empirical Finding:** m ∈ [0.99, 0.999] works well across tasks.
- Too small (m < 0.9): High variance, unstable training
- Too large (m > 0.9999): Too slow adaptation, stale keys

```python
class MoCo(nn.Module):
    """Momentum Contrast for unsupervised learning"""

    def __init__(self, encoder, dim=128, K=65536, m=0.999, temperature=0.07):
        """
        Args:
            encoder: Encoder network
            dim: Feature dimension
            K: Queue size
            m: Momentum coefficient
            temperature: Temperature parameter
        """
        super().__init__()

        self.K = K
        self.m = m
        self.temperature = temperature

        # Query encoder
        self.encoder_q = encoder
        encoder_dim = encoder.fc.in_features
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, dim)
        )

        # Key encoder (momentum encoder)
        self.encoder_k = self._copy_encoder(self.encoder_q)

        # Disable gradients for key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _copy_encoder(self, encoder):
        """Create a copy of encoder"""
        import copy
        encoder_k = copy.deepcopy(encoder)
        for param in encoder_k.parameters():
            param.requires_grad = False
        return encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Update key encoder with momentum"""
        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Replace oldest keys
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Args:
            im_q: Query images
            im_k: Key images
        """
        # Compute query features
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        # Compute logits
        # Positive logits: N x 1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative logits: N x K
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # Logits: N x (1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        # Labels: positive key is at position 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # Update queue
        self._dequeue_and_enqueue(k)

        return logits, labels

# Training MoCo
def train_moco(model, train_loader, epochs=200):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.03,
        momentum=0.9,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for (im_q, im_k), _ in train_loader:
            im_q, im_k = im_q.cuda(), im_k.cuda()

            # Forward pass
            logits, labels = model(im_q, im_k)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### SwAV (Swapped Assignment Views)

```python
class SwAV(nn.Module):
    """Swapping Assignments between Views"""

    def __init__(self, encoder, num_prototypes=3000, queue_length=3840):
        super().__init__()
        self.encoder = encoder
        self.num_prototypes = num_prototypes

        # Get encoder dimension
        encoder_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 128)
        )

        # Prototypes
        self.prototypes = nn.Linear(128, num_prototypes, bias=False)

        # Queue for storing codes
        self.register_buffer("queue", torch.zeros(2, queue_length, 128))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, x):
        # Get features
        features = self.encoder(x)
        z = self.projection(features)
        z = F.normalize(z, dim=1)

        # Get prototypes
        p = self.prototypes(z)
        return z, p

    def sinkhorn(self, Q, num_iters=3):
        """Sinkhorn-Knopp algorithm"""
        with torch.no_grad():
            Q = torch.exp(Q / 0.05).T  # K x B
            Q /= Q.sum()

            K, B = Q.shape

            for _ in range(num_iters):
                # Normalize rows
                Q /= Q.sum(dim=1, keepdim=True)
                Q /= K

                # Normalize columns
                Q /= Q.sum(dim=0, keepdim=True)
                Q /= B

            Q *= B
            return Q.T

    def swav_loss(self, z1, p1, z2, p2):
        """SwAV loss with swapped prediction"""
        # Get codes by solving optimal transport
        with torch.no_grad():
            q1 = self.sinkhorn(p1)
            q2 = self.sinkhorn(p2)

        # Swapped prediction loss
        loss = 0
        loss -= torch.mean(torch.sum(q1 * F.log_softmax(p2, dim=1), dim=1))
        loss -= torch.mean(torch.sum(q2 * F.log_softmax(p1, dim=1), dim=1))
        loss /= 2

        return loss

# Multi-crop strategy
class MultiCropAugmentation:
    """Multi-crop augmentation for SwAV"""

    def __init__(self, size_crops=[224, 96], nmb_crops=[2, 6]):
        """
        Args:
            size_crops: Sizes of crops
            nmb_crops: Number of crops for each size
        """
        self.transforms = []

        for size, nmb in zip(size_crops, nmb_crops):
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.14, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            self.transforms.extend([transform] * nmb)

    def __call__(self, x):
        return [transform(x) for transform in self.transforms]
```

---

## Masked Image Modeling

Learn representations by predicting masked regions of images.

### Theoretical Foundations

#### Masked Prediction Theory

**Definition (Masked Autoencoding Objective):**
Given image x partitioned into patches {p₁,...,p_N}, select mask M ⊂ {1,...,N}, learn encoder f_θ and decoder g_φ:

L_MAE = E_M [Σ_{i∈M} ||p_i - g_φ(f_θ({p_j}_{j∉M}))||²]

**Theorem 13 (Information Content vs Mask Ratio - He et al., 2022):**
For mask ratio r ∈ [0,1], the mutual information between visible and masked patches:

I(p_visible; p_masked) ≈ (1-r)·I(X) · ρ

where ρ ∈ [0,1] is patch correlation coefficient.

**Optimal Mask Ratio:** For natural images with high redundancy (ρ ≈ 0.8):

r* ∈ [0.6, 0.8]

balances:
1. **Too low (r < 0.5):** Task too easy, trivial copying
2. **Too high (r > 0.9):** Task too hard, insufficient context
3. **Optimal:** Forces learning semantic features

**Example - Natural Images:**
- Low masking (r=0.25): Learn local textures
- High masking (r=0.75): Learn global semantics
- MAE uses r=0.75 for ImageNet

**Theorem 14 (Reconstruction vs Representation Quality):**
There exists a trade-off between reconstruction loss L_recon and downstream task performance P_down:

∂P_down/∂L_recon < 0   (lower loss ≠ better representation)

**Explanation:** Over-optimization on pixel-level reconstruction can hurt semantic representation.

**Mitigation:**
1. **Normalized pixels:** (x - μ) / σ per patch
2. **Stop gradient:** Don't backprop through decoder to encoder
3. **Asymmetric architecture:** Small decoder (forces encoder to learn semantics)

**Theorem 15 (Sample Efficiency of Masked Modeling - Tian et al., 2023):**
Compared to contrastive learning requiring O(exp(I)) negatives, masked modeling requires:

n_samples = O(d·(1-r)^{-1}/ε²)

where d is embedding dim, r is mask ratio.

**Key Advantage:** Linear in target information (not exponential)!

**Comparison:**
| Method | Negatives Needed | Sample Complexity | Batch Size |
|--------|------------------|-------------------|------------|
| **Contrastive** | O(exp(I)) | O(K·d/ε²) | Large (4K-8K) |
| **Masked** | 0 | O(d/((1-r)ε²)) | Moderate (1K-2K) |

**Theorem 16 (Decoder Capacity and Representation Quality - Zhou et al., 2022):**
For encoder with capacity C_enc and decoder with capacity C_dec:

**If C_dec ≈ C_enc:** Decoder can "cheat" by storing information → encoder learns trivial features
**If C_dec << C_enc:** Encoder forced to learn semantic representations

**Optimal Ratio:** C_dec / C_enc ∈ [0.1, 0.3]

**MAE Design:**
- Encoder: 12 layers, 768 dim
- Decoder: 8 layers, 512 dim
- Ratio: ≈ 0.2 (forces semantic encoding)

#### Position Embedding and Invariance

**Theorem 17 (Position Invariance in MAE):**
Unlike contrastive methods, MAE is position-invariant:

L_MAE(x) = L_MAE(T(x))   for position-preserving transform T

**Proof:**
MAE predicts pixels at specific positions → learns position-dependent features.
Contrastive learning uses invariant augmentations → position information discarded.

**Consequence:**
- MAE: Better for dense prediction (segmentation, detection)
- Contrastive: Better for classification (position-invariant)

### MAE (Masked Autoencoder)

```python
class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: B x C x H x W -> B x num_patches x embed_dim
        x = self.projection(x)  # B x embed_dim x H' x W'
        x = x.flatten(2).transpose(1, 2)  # B x num_patches x embed_dim
        return x

class MAE(nn.Module):
    """Masked Autoencoder for self-supervised learning"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_depth=12,
        encoder_dim=768,
        encoder_heads=12,
        decoder_depth=8,
        decoder_dim=512,
        decoder_heads=16,
        mask_ratio=0.75
    ):
        super().__init__()

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, 3, encoder_dim
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, encoder_dim)
        )

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_depth)

        # Decoder embedding
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_dim)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, decoder_depth)

        # Prediction head
        self.prediction_head = nn.Linear(
            decoder_dim,
            patch_size ** 2 * 3  # Reconstruct RGB values
        )

    def random_masking(self, x):
        """
        Random masking of patches
        Returns:
            x_masked: visible patches only
            mask: binary mask (0 is keep, 1 is remove)
            ids_restore: indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        # Random shuffle
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep only visible patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        # Binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, imgs):
        # Patch embedding
        x = self.patch_embed(imgs)
        x = x + self.pos_embed

        # Masking
        x, mask, ids_restore = self.random_masking(x)

        # Encoder
        x = self.encoder(x)

        # Decoder embedding
        x = self.decoder_embed(x)

        # Add mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = x + self.decoder_pos_embed

        # Decoder
        x = self.decoder(x)

        # Prediction
        pred = self.prediction_head(x)

        return pred, mask

    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p

        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, p * p * C)
        return x

    def compute_loss(self, imgs, pred, mask):
        """Compute reconstruction loss on masked patches"""
        target = self.patchify(imgs)

        # Normalize target
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5

        # MSE loss on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean per patch
        loss = (loss * mask).sum() / mask.sum()  # Mean on masked patches

        return loss

# Training MAE
def train_mae(model, train_loader, epochs=800):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-4,
        betas=(0.9, 0.95),
        weight_decay=0.05
    )

    for epoch in range(epochs):
        total_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.cuda()

            # Forward pass
            pred, mask = model(imgs)
            loss = model.compute_loss(imgs, pred, mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

### BEiT (BERT Pre-training of Image Transformers)

```python
class DALLE_VAE:
    """Discrete VAE tokenizer (simplified version)"""

    def __init__(self, num_tokens=8192):
        self.num_tokens = num_tokens
        # In practice, use pre-trained dVAE from DALL-E

    def encode(self, imgs):
        """Encode images to discrete tokens"""
        # Returns: B x num_image_patches (discrete token IDs)
        pass

    def decode(self, tokens):
        """Decode tokens to images"""
        pass

class BEiT(nn.Module):
    """BERT Pre-training of Image Transformers"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_classes=8192,  # Number of visual tokens
        embed_dim=768,
        depth=12,
        num_heads=12,
        mask_ratio=0.4
    ):
        super().__init__()

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, 3, embed_dim
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)

        # Prediction head for visual tokens
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, imgs, mask_indices):
        """
        Args:
            imgs: Input images
            mask_indices: Indices of patches to mask
        """
        # Patch embedding
        x = self.patch_embed(imgs)
        B, N, D = x.shape

        # Add positional embedding
        x = x + self.pos_embed

        # Replace masked patches with mask token
        mask_tokens = self.mask_token.expand(B, -1, -1)
        for i in range(B):
            x[i, mask_indices[i]] = mask_tokens[i]

        # Transformer encoding
        x = self.encoder(x)

        # Predict visual tokens for masked positions
        logits = self.head(x)

        return logits

    def compute_loss(self, logits, visual_tokens, mask_indices):
        """Compute cross-entropy loss on masked positions"""
        B = logits.shape[0]
        loss = 0

        for i in range(B):
            masked_logits = logits[i, mask_indices[i]]
            masked_targets = visual_tokens[i, mask_indices[i]]
            loss += F.cross_entropy(masked_logits, masked_targets)

        return loss / B
```

---

## Self-Supervised NLP

Learn language representations without labeled data.

### Masked Language Modeling (BERT-style)

```python
class MaskedLanguageModel(nn.Module):
    """BERT-style masked language modeling"""

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=512,
        mask_prob=0.15
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.mask_prob = mask_prob

        # Token + Position + Segment embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.segment_embedding = nn.Embedding(2, hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )

    def create_masked_input(self, input_ids):
        """
        Create masked input for MLM
        - 80% replaced with [MASK]
        - 10% replaced with random token
        - 10% unchanged
        """
        labels = input_ids.clone()

        # Create mask
        probability_matrix = torch.full(input_ids.shape, self.mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80%: Replace with [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, 0.8)
        ).bool() & masked_indices
        input_ids[indices_replaced] = self.vocab_size - 1  # [MASK] token

        # 10%: Replace with random token
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, 0.5)
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            self.vocab_size, input_ids.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # 10%: Keep unchanged

        return input_ids, labels

    def forward(self, input_ids, segment_ids=None):
        batch_size, seq_length = input_ids.shape

        # Create position IDs
        position_ids = torch.arange(seq_length).unsqueeze(0).expand_as(input_ids)
        position_ids = position_ids.to(input_ids.device)

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        segment_embeds = self.segment_embedding(segment_ids)

        embeddings = token_embeds + position_embeds + segment_embeds

        # Encode
        hidden_states = self.encoder(embeddings)

        # MLM prediction
        mlm_logits = self.mlm_head(hidden_states)

        return mlm_logits

    def compute_loss(self, mlm_logits, labels):
        """Compute MLM loss"""
        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        loss = loss_fct(
            mlm_logits.view(-1, self.vocab_size),
            labels.view(-1)
        )
        return loss

# Training MLM
def train_mlm(model, train_loader, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        total_loss = 0
        for input_ids, _ in train_loader:
            input_ids = input_ids.cuda()

            # Create masked input
            masked_input, labels = model.create_masked_input(input_ids)

            # Forward pass
            mlm_logits = model(masked_input)
            loss = model.compute_loss(mlm_logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

### Causal Language Modeling (GPT-style)

```python
class CausalLanguageModel(nn.Module):
    """GPT-style causal language modeling"""

    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=1024
    ):
        super().__init__()

        self.vocab_size = vocab_size

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)

        # Transformer decoder (causal attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape

        # Position IDs
        position_ids = torch.arange(seq_length).unsqueeze(0).expand_as(input_ids)
        position_ids = position_ids.to(input_ids.device)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        embeddings = token_embeds + position_embeds

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_length
        ).to(input_ids.device)

        # Decode
        hidden_states = self.decoder(
            embeddings,
            embeddings,
            tgt_mask=causal_mask
        )

        # LM prediction
        logits = self.lm_head(hidden_states)

        return logits

    def compute_loss(self, logits, input_ids):
        """Compute next-token prediction loss"""
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        )
        return loss
```

---

## Bootstrap Methods

Learn without negative pairs by bootstrapping from the model itself.

### Theoretical Framework

#### Collapse Prevention Theory

**Definition (Representation Collapse):**
Collapse occurs when encoder maps all inputs to constant:

∀x, x': f_θ(x) = f_θ(x') = c

**Theorem 18 (Collapse in Symmetric Networks - Grill et al., 2020):**
For symmetric architecture without stop-gradient:

L_sym = E[||f_θ(x₁) - f_θ(x₂)||²]

has trivial solution f_θ(x) = c (constant for all x).

**Proof:**
∇_θ L_sym = 2E[(f_θ(x₁) - f_θ(x₂))·(∇f_θ(x₁) - ∇f_θ(x₂))]

Setting f_θ(x) = c gives ∇_θ L_sym = 0, so collapse is a stationary point.

**BYOL Solution:** Asymmetry via stop-gradient and predictor.

**Theorem 19 (BYOL Loss and Collapse Prevention - Tian et al., 2021):**
BYOL loss with online network θ_o and target network θ_t:

L_BYOL = E[||p(f_{θ_o}(x₁)) - sg(f_{θ_t}(x₂))||²]

where sg(·) is stop-gradient, p(·) is predictor.

**Collapse is NOT a stationary point** if:
1. Predictor p is non-trivial
2. Target θ_t ≠ θ_o (via momentum update)
3. Batch statistics in BatchNorm

**Theorem 20 (Role of BatchNorm in Collapse Prevention - Richemond et al., 2020):**
BatchNorm implicitly prevents collapse by:

z_BN = γ·(z - μ_batch)/σ_batch + β

**Key Property:** Collapse z → c implies:
- σ_batch → 0
- Gradients → ∞
- Training becomes unstable

**Empirical Finding:** Removing BatchNorm from BYOL causes immediate collapse!

**Theorem 21 (Momentum Update Stability - Grill et al., 2020):**
For target network update θ_t ← τ·θ_t + (1-τ)·θ_o with τ ∈ [0.99, 0.999]:

**Stability Condition:** |λ_max(Jacobian(L))| < 1 where λ_max is largest eigenvalue.

**Analysis:**
- τ too small (< 0.9): Target changes too fast → unstable gradients
- τ too large (> 0.9999): Target too stale → slow learning
- Optimal: τ ∈ [0.996, 0.999]

#### SimSiam: Simplification of BYOL

**Theorem 22 (SimSiam Without Momentum - Chen & He, 2021):**
SimSiam shows momentum encoder is NOT necessary:

L_SimSiam = E[||p(f_θ(x₁)) - sg(f_θ(x₂))||²] / 2 +
            E[||p(f_θ(x₂)) - sg(f_θ(x₁))||²] / 2

**Key Differences from BYOL:**
1. Same encoder θ for both branches (no momentum)
2. Symmetric loss (swap x₁, x₂)
3. Stop-gradient + predictor sufficient!

**Theorem 23 (Predictor Necessity - Chen & He, 2021):**
Without predictor p:

L = E[||f_θ(x₁) - sg(f_θ(x₂))||²]

**Gradient w.r.t. θ:**
∇_θ L = E[∂f/∂θ · (f(x₁) - f(x₂))]

Since f(x₂) has no gradient (stop-grad), this can lead to:
- Constant solution f(x₁) = E[f(x₂)]
- Collapse if E[f(x₂)] = c

**Predictor p breaks symmetry:**
∇_θ L = E[∂p∘f/∂θ · (p∘f(x₁) - f(x₂))]

The predictor provides extra capacity preventing trivial constant solution.

**Theorem 24 (Contrastive vs Bootstrap Comparison):**

| Aspect | Contrastive (SimCLR) | Bootstrap (BYOL/SimSiam) |
|--------|----------------------|---------------------------|
| **Negatives** | Required (K large) | Not required |
| **Batch size** | Very large (4K-8K) | Moderate (256-1K) |
| **Collapse prevention** | Explicit repulsion | Implicit (architecture) |
| **Memory** | O(K·d) | O(d) |
| **Sample complexity** | O(K·d/ε²) | O(d/ε²) |
| **Performance** | Excellent | Excellent (similar) |

**Key Trade-off:**
- **Contrastive:** Explicit, theoretically grounded, needs large batches
- **Bootstrap:** Implicit, architecturally dependent, smaller batches work

### BYOL (Bootstrap Your Own Latent)

```python
class BYOL(nn.Module):
    """Bootstrap Your Own Latent"""

    def __init__(self, encoder, projection_dim=256, hidden_dim=4096, tau=0.996):
        super().__init__()

        self.tau = tau

        # Online network
        self.online_encoder = encoder
        encoder_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

        self.online_projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        # Target network (momentum encoder)
        self.target_encoder = self._copy_encoder(self.online_encoder)
        self.target_projector = self._copy_projector(self.online_projector)

        # Disable gradients for target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def _copy_encoder(self, encoder):
        import copy
        return copy.deepcopy(encoder)

    def _copy_projector(self, projector):
        import copy
        return copy.deepcopy(projector)

    @torch.no_grad()
    def _update_target_network(self):
        """Update target network with exponential moving average"""
        for online_params, target_params in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_params.data = (
                self.tau * target_params.data +
                (1 - self.tau) * online_params.data
            )

        for online_params, target_params in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            target_params.data = (
                self.tau * target_params.data +
                (1 - self.tau) * online_params.data
            )

    def forward(self, x1, x2):
        # Online network forward
        online_feat_1 = self.online_encoder(x1)
        online_proj_1 = self.online_projector(online_feat_1)
        online_pred_1 = self.online_predictor(online_proj_1)

        online_feat_2 = self.online_encoder(x2)
        online_proj_2 = self.online_projector(online_feat_2)
        online_pred_2 = self.online_predictor(online_proj_2)

        # Target network forward (no gradients)
        with torch.no_grad():
            self._update_target_network()

            target_feat_1 = self.target_encoder(x1)
            target_proj_1 = self.target_projector(target_feat_1)

            target_feat_2 = self.target_encoder(x2)
            target_proj_2 = self.target_projector(target_feat_2)

        return online_pred_1, online_pred_2, target_proj_1, target_proj_2

    def regression_loss(self, x, y):
        """Normalized MSE loss"""
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1).mean()

    def compute_loss(self, online_pred_1, online_pred_2, target_proj_1, target_proj_2):
        """Symmetric loss"""
        loss_1 = self.regression_loss(online_pred_1, target_proj_2)
        loss_2 = self.regression_loss(online_pred_2, target_proj_1)
        return (loss_1 + loss_2) / 2

# Training BYOL
def train_byol(model, train_loader, epochs=1000):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.2,
        momentum=0.9,
        weight_decay=1e-6
    )

    for epoch in range(epochs):
        for (x1, x2), _ in train_loader:
            x1, x2 = x1.cuda(), x2.cuda()

            # Forward
            online_pred_1, online_pred_2, target_proj_1, target_proj_2 = model(x1, x2)
            loss = model.compute_loss(
                online_pred_1, online_pred_2,
                target_proj_1, target_proj_2
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### SimSiam

```python
class SimSiam(nn.Module):
    """Simple Siamese Networks"""

    def __init__(self, encoder, projection_dim=2048, pred_dim=512):
        super().__init__()

        self.encoder = encoder
        encoder_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

        # Projection MLP
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=False)  # No affine
        )

        # Prediction MLP
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, projection_dim)
        )

    def forward(self, x1, x2):
        # Encode
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)

        # Project
        z1 = self.projector(f1)
        z2 = self.projector(f2)

        # Predict
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()

    def criterion(self, p1, p2, z1, z2):
        """Negative cosine similarity"""
        # Normalize
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Symmetric loss
        loss = -(p1 * z2).sum(dim=1).mean() / 2
        loss -= (p2 * z1).sum(dim=1).mean() / 2

        return loss
```

---

## Other SSL Approaches

### Rotation Prediction

```python
class RotationPredictor(nn.Module):
    """Predict image rotation as pretext task"""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        # Replace classification head
        encoder_dim = encoder.fc.in_features
        encoder.fc = nn.Linear(encoder_dim, 4)  # 4 rotation classes

    def create_rotated_batch(self, images):
        """Create batch with 4 rotations"""
        batch_size = images.shape[0]
        rotations = []
        labels = []

        for i in range(4):
            rotated = torch.rot90(images, i, [2, 3])
            rotations.append(rotated)
            labels.extend([i] * batch_size)

        rotated_batch = torch.cat(rotations, dim=0)
        rotation_labels = torch.tensor(labels).to(images.device)

        return rotated_batch, rotation_labels

    def forward(self, x):
        return self.encoder(x)

# Training
def train_rotation(model, train_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, _ in train_loader:
            images = images.cuda()

            # Create rotated batch
            rotated_images, rotation_labels = model.create_rotated_batch(images)

            # Predict rotation
            logits = model(rotated_images)
            loss = criterion(logits, rotation_labels)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Jigsaw Puzzle

```python
class JigsawSolver(nn.Module):
    """Solve jigsaw puzzles as pretext task"""

    def __init__(self, encoder, grid_size=3, num_permutations=100):
        super().__init__()
        self.grid_size = grid_size
        self.num_permutations = num_permutations

        # Generate permutations
        self.permutations = self._generate_permutations()

        # Encoder processes each patch
        self.encoder = encoder
        encoder_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

        # Classifier for permutation
        self.classifier = nn.Linear(
            encoder_dim * grid_size * grid_size,
            num_permutations
        )

    def _generate_permutations(self):
        """Generate valid permutations"""
        import itertools
        all_perms = list(itertools.permutations(
            range(self.grid_size * self.grid_size)
        ))
        # Sample subset
        indices = np.random.choice(
            len(all_perms),
            self.num_permutations,
            replace=False
        )
        return [all_perms[i] for i in indices]

    def create_jigsaw_batch(self, images):
        """Create jigsaw puzzles"""
        B, C, H, W = images.shape
        patch_h, patch_w = H // self.grid_size, W // self.grid_size

        patches_list = []
        labels = []

        for img in images:
            # Extract patches
            patches = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    patch = img[
                        :,
                        i*patch_h:(i+1)*patch_h,
                        j*patch_w:(j+1)*patch_w
                    ]
                    patches.append(patch)

            # Random permutation
            perm_idx = np.random.randint(self.num_permutations)
            perm = self.permutations[perm_idx]
            shuffled_patches = [patches[i] for i in perm]

            patches_list.append(torch.stack(shuffled_patches))
            labels.append(perm_idx)

        return torch.stack(patches_list), torch.tensor(labels)

    def forward(self, patches):
        """
        Args:
            patches: B x num_patches x C x H x W
        """
        B, N, C, H, W = patches.shape

        # Encode each patch
        patches_flat = patches.view(B * N, C, H, W)
        features = self.encoder(patches_flat)
        features = features.view(B, N, -1)

        # Concatenate features
        features_cat = features.view(B, -1)

        # Classify permutation
        logits = self.classifier(features_cat)

        return logits
```

---

## Downstream Fine-tuning

Use pre-trained SSL models for downstream tasks.

### Linear Evaluation

```python
class LinearEvaluation:
    """Evaluate SSL representations with linear probe"""

    def __init__(self, ssl_model, num_classes):
        # Freeze SSL encoder
        for param in ssl_model.encoder.parameters():
            param.requires_grad = False

        self.encoder = ssl_model.encoder
        encoder_dim = self._get_encoder_dim()

        # Linear classifier
        self.classifier = nn.Linear(encoder_dim, num_classes).cuda()

    def _get_encoder_dim(self):
        """Get encoder output dimension"""
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        with torch.no_grad():
            output = self.encoder(dummy_input)
        return output.shape[1]

    def train(self, train_loader, val_loader, epochs=100):
        optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=0
        )
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(epochs):
            # Training
            self.classifier.train()
            for images, labels in train_loader:
                images, labels = images.cuda(), labels.cuda()

                # Extract features (no gradients)
                with torch.no_grad():
                    features = self.encoder(images)

                # Classify
                logits = self.classifier(features)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            acc = self.evaluate(val_loader)
            if acc > best_acc:
                best_acc = acc

            print(f"Epoch {epoch+1}, Val Acc: {acc:.2f}%, Best: {best_acc:.2f}%")

        return best_acc

    def evaluate(self, data_loader):
        self.classifier.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.cuda(), labels.cuda()

                features = self.encoder(images)
                logits = self.classifier(features)
                _, predicted = logits.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100. * correct / total
```

### Fine-tuning

```python
class FineTuning:
    """Fine-tune entire SSL model"""

    def __init__(self, ssl_model, num_classes):
        self.model = ssl_model.encoder

        # Replace head
        encoder_dim = self._get_encoder_dim()
        self.model.fc = nn.Linear(encoder_dim, num_classes)
        self.model = self.model.cuda()

    def _get_encoder_dim(self):
        # Find the dimension before the final FC layer
        if hasattr(self.model, 'fc'):
            if isinstance(self.model.fc, nn.Identity):
                # Find previous layer
                for name, module in reversed(list(self.model.named_modules())):
                    if isinstance(module, nn.Linear):
                        return module.out_features
        return 2048  # Default for ResNet50

    def train(self, train_loader, val_loader, epochs=100, lr=0.001):
        # Lower learning rate for pre-trained weights
        optimizer = torch.optim.SGD(
            [
                {'params': self.model.fc.parameters(), 'lr': lr * 10},
                {'params': [p for n, p in self.model.named_parameters()
                           if 'fc' not in n], 'lr': lr}
            ],
            momentum=0.9,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            # Training
            self.model.train()
            for images, labels in train_loader:
                images, labels = images.cuda(), labels.cuda()

                logits = self.model(images)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation
            acc = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}, Val Acc: {acc:.2f}%")

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.cuda(), labels.cuda()

                logits = self.model(images)
                _, predicted = logits.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100. * correct / total
```

---

## Production Implementation

### Complete SSL Pipeline

```python
class SSLPipeline:
    """Complete self-supervised learning pipeline"""

    def __init__(self, method='simclr', backbone='resnet50'):
        self.method = method
        self.backbone = backbone

        # Initialize encoder
        if backbone == 'resnet50':
            from torchvision.models import resnet50
            self.encoder = resnet50(pretrained=False)

        # Initialize SSL model
        if method == 'simclr':
            self.model = SimCLR(self.encoder)
        elif method == 'moco':
            self.model = MoCo(self.encoder)
        elif method == 'byol':
            self.model = BYOL(self.encoder)
        elif method == 'mae':
            self.model = MAE()
        else:
            raise ValueError(f"Unknown method: {method}")

        self.model = self.model.cuda()

    def pretrain(self, train_loader, epochs=800, save_dir='checkpoints'):
        """Pre-training phase"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=0.05
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                if self.method in ['simclr', 'moco', 'byol']:
                    (x1, x2), _ = batch
                    x1, x2 = x1.cuda(), x2.cuda()

                    if self.method == 'simclr':
                        z1, z2 = self.model(x1, x2)
                        loss = self.model.nt_xent_loss(z1, z2)
                    elif self.method == 'moco':
                        logits, labels = self.model(x1, x2)
                        loss = F.cross_entropy(logits, labels)
                    elif self.method == 'byol':
                        p1, p2, z1, z2 = self.model(x1, x2)
                        loss = self.model.compute_loss(p1, p2, z1, z2)

                elif self.method == 'mae':
                    images, _ = batch
                    images = images.cuda()
                    pred, mask = self.model(images)
                    loss = self.model.compute_loss(images, pred, mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 100 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{save_dir}/checkpoint_epoch_{epoch+1}.pth")

        # Save final model
        torch.save(self.model.state_dict(), f"{save_dir}/final_model.pth")

    def linear_eval(self, train_loader, val_loader, num_classes):
        """Linear evaluation"""
        evaluator = LinearEvaluation(self.model, num_classes)
        best_acc = evaluator.train(train_loader, val_loader)
        return best_acc

    def fine_tune(self, train_loader, val_loader, num_classes):
        """Fine-tuning"""
        fine_tuner = FineTuning(self.model, num_classes)
        fine_tuner.train(train_loader, val_loader)

# Usage example
def main():
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Pre-training data (unlabeled)
    pretrain_transform = SimCLRAugmentation()
    pretrain_dataset = datasets.ImageFolder(
        'unlabeled_data',
        transform=pretrain_transform
    )
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # Downstream data (labeled)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(
        'labeled_data/train',
        transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    val_dataset = datasets.ImageFolder(
        'labeled_data/val',
        transform=train_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # SSL Pipeline
    pipeline = SSLPipeline(method='simclr', backbone='resnet50')

    # Pre-training
    print("Starting pre-training...")
    pipeline.pretrain(pretrain_loader, epochs=800)

    # Linear evaluation
    print("\nLinear evaluation...")
    linear_acc = pipeline.linear_eval(train_loader, val_loader, num_classes=10)
    print(f"Linear eval accuracy: {linear_acc:.2f}%")

    # Fine-tuning
    print("\nFine-tuning...")
    pipeline.fine_tune(train_loader, val_loader, num_classes=10)

if __name__ == '__main__':
    main()
```

### Multi-GPU Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedSSL:
    """Distributed SSL training"""

    def __init__(self, rank, world_size, method='simclr'):
        self.rank = rank
        self.world_size = world_size

        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

        # Create model
        if method == 'simclr':
            encoder = models.resnet50(pretrained=False)
            model = SimCLR(encoder)

        # Wrap with DDP
        model = model.to(rank)
        self.model = DDP(model, device_ids=[rank])

    def train(self, train_loader, epochs=100):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3 * self.world_size  # Linear scaling rule
        )

        for epoch in range(epochs):
            train_loader.sampler.set_epoch(epoch)

            for (x1, x2), _ in train_loader:
                x1, x2 = x1.to(self.rank), x2.to(self.rank)

                z1, z2 = self.model(x1, x2)
                loss = self.model.module.nt_xent_loss(z1, z2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.rank == 0:
                print(f"Epoch {epoch+1}/{epochs} completed")

        dist.destroy_process_group()

# Launch distributed training
def launch_distributed():
    import torch.multiprocessing as mp
    world_size = torch.cuda.device_count()
    mp.spawn(
        DistributedSSL,
        args=(world_size, 'simclr'),
        nprocs=world_size,
        join=True
    )
```

---

## Best Practices

### 1. Data Augmentation Strategy

```python
# Strong augmentations for SSL
ssl_augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Weaker augmentations for downstream
downstream_augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

### 2. Hyperparameter Guidelines

```python
SSL_CONFIGS = {
    'simclr': {
        'batch_size': 4096,  # Large batch important
        'temperature': 0.5,
        'projection_dim': 128,
        'lr': 0.3 * (batch_size / 256),  # Linear scaling
        'epochs': 800,
        'optimizer': 'LARS'
    },
    'moco': {
        'batch_size': 256,
        'queue_size': 65536,
        'momentum': 0.999,
        'temperature': 0.07,
        'epochs': 200
    },
    'byol': {
        'batch_size': 4096,
        'tau': 0.996,  # EMA coefficient
        'projection_dim': 256,
        'epochs': 1000
    },
    'mae': {
        'batch_size': 1024,
        'mask_ratio': 0.75,
        'lr': 1.5e-4,
        'epochs': 800,
        'warmup_epochs': 40
    }
}
```

### 3. Evaluation Protocol

```python
def evaluate_ssl_model(ssl_model, dataset_name):
    """Standard evaluation protocol"""

    # 1. Linear evaluation (fixed features)
    linear_acc = linear_evaluation(
        ssl_model,
        dataset_name,
        epochs=100,
        lr=0.1
    )
    print(f"Linear accuracy: {linear_acc:.2f}%")

    # 2. Fine-tuning (full model)
    finetune_acc = fine_tuning(
        ssl_model,
        dataset_name,
        epochs=100,
        lr=0.001
    )
    print(f"Fine-tuning accuracy: {finetune_acc:.2f}%")

    # 3. Few-shot learning
    for n_shot in [1, 5, 10]:
        few_shot_acc = few_shot_evaluation(
            ssl_model,
            dataset_name,
            n_shot=n_shot
        )
        print(f"{n_shot}-shot accuracy: {few_shot_acc:.2f}%")

    # 4. Transfer learning
    transfer_datasets = ['cifar10', 'cifar100', 'food101']
    for transfer_dataset in transfer_datasets:
        transfer_acc = transfer_evaluation(
            ssl_model,
            transfer_dataset
        )
        print(f"Transfer to {transfer_dataset}: {transfer_acc:.2f}%")
```

---

## Summary

Self-supervised learning enables learning from unlabeled data through clever pretext tasks:

1. **Contrastive Methods** (SimCLR, MoCo, SwAV): Learn by contrasting positive/negative pairs
2. **Masked Modeling** (MAE, BEiT): Learn by reconstructing masked regions
3. **Bootstrap Methods** (BYOL, SimSiam): Learn without negatives via momentum
4. **Classic Pretext Tasks**: Rotation, jigsaw, colorization

**Key Advantages:**
- Leverage unlimited unlabeled data
- Learn general-purpose representations
- Reduce labeled data requirements
- Strong transfer learning performance

**When to Use:**
- Limited labeled data available
- Need general-purpose representations
- Transfer learning scenarios
- Domain adaptation tasks
