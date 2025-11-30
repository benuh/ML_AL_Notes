# Multimodal AI: Complete Guide

## Table of Contents
1. [Introduction to Multimodal Learning](#introduction)
2. [Vision-Language Models](#vision-language-models)
3. [Audio-Visual Learning](#audio-visual-learning)
4. [Multimodal Transformers](#multimodal-transformers)
5. [Cross-Modal Retrieval](#cross-modal-retrieval)
6. [Multimodal Generation](#multimodal-generation)
7. [Video Understanding](#video-understanding)
8. [Production Implementation](#production-implementation)

---

## Introduction to Multimodal Learning

Multimodal AI combines information from multiple sources (vision, language, audio) to create richer representations and solve complex tasks.

### Mathematical Foundations

#### Information Theory for Multimodal Learning

**Definition (Multimodal Data):** Let X ∈ 𝒳 and Y ∈ 𝒴 be random variables representing two modalities (e.g., image and text).

**Goal:** Learn joint representation Z that captures information from both modalities.

**Theorem 1 (Multimodal Information Gain - Baltrušaitis et al., 2019):**
The information about target T contained in multimodal representation is bounded:

I(T; X, Y) ≤ I(T; X) + I(T; Y)

with equality when X and Y are conditionally independent given T:

X ⊥ Y | T

**Proof:**
By chain rule of mutual information:
I(T; X, Y) = I(T; X) + I(T; Y | X)

Since I(T; Y | X) ≤ I(T; Y) with equality iff X ⊥ Y | T:
I(T; X, Y) ≤ I(T; X) + I(T; Y) ∎

**Corollary (Complementarity):**
If modalities are complementary (X ⊥ Y | T), multimodal information is sum of unimodal information.

**Example:**
- Image: I(T; X) = 0.8 nats
- Text: I(T; Y) = 0.7 nats
- Multimodal (independent): I(T; X,Y) = 1.5 nats
- Multimodal (dependent): I(T; X,Y) < 1.5 nats (redundancy)

**Theorem 2 (Redundancy vs Synergy Decomposition - Williams & Beer, 2010):**
Multimodal mutual information decomposes as:

I(T; X, Y) = R(X,Y) + U_X + U_Y + S(X,Y)

where:
- R(X,Y): Redundancy (information in both X and Y)
- U_X: Unique information in X
- U_Y: Unique information in Y
- S(X,Y): Synergy (information only in joint)

**Key Insight:** Synergy S(X,Y) > 0 justifies multimodal fusion!

**Theorem 3 (Sample Complexity of Multimodal Learning - Arora & Livni, 2017):**
To learn ε-optimal multimodal classifier with probability ≥ 1-δ:

n_samples = O((d_X + d_Y + d_Z)·log(1/δ) / ε²)

where d_X, d_Y are modality dimensions, d_Z is joint representation dimension.

**Comparison with Unimodal:**
- Unimodal X: n_X = O(d_X / ε²)
- Multimodal X,Y: n_{X,Y} = O((d_X + d_Y + d_Z) / ε²)
- Trade-off: More parameters vs more information

#### Canonical Correlation Analysis (CCA) Theory

**Definition (CCA Objective):** Find linear projections that maximize correlation:

max_{w_X, w_Y} Corr(w_X^T X, w_Y^T Y)

subject to: Var(w_X^T X) = Var(w_Y^T Y) = 1

**Theorem 4 (CCA Solution - Hotelling, 1936):**
The optimal projections (w_X, w_Y) satisfy generalized eigenvalue problem:

Σ_{XY} Σ_{YY}^{-1} Σ_{YX} w_X = ρ² Σ_{XX} w_X
Σ_{YX} Σ_{XX}^{-1} Σ_{XY} w_Y = ρ² Σ_{YY} w_Y

where:
- Σ_{XX} = Cov(X, X), Σ_{YY} = Cov(Y, Y)
- Σ_{XY} = Cov(X, Y) = Σ_{YX}^T
- ρ = √λ is canonical correlation (eigenvalue)

**Multiple Components:** Top-k eigenvectors give k canonical correlations ρ₁ ≥ ρ₂ ≥ ... ≥ ρ_k.

**Theorem 5 (Deep CCA - Andrew et al., 2013):**
For nonlinear encoders f_X: 𝒳 → ℝ^d and f_Y: 𝒴 → ℝ^d, Deep CCA objective:

max_{f_X, f_Y} Corr(f_X(X), f_Y(Y))

**Gradient:** Uses SVD backpropagation through correlation matrix.

**Sample Complexity:** n = O((d² + d_X + d_Y)/ε²) for ε-optimal correlation.

**Theorem 6 (CCA Generalization Bound - Gretton et al., 2005):**
With probability ≥ 1-δ over n samples:

|ρ_empirical - ρ_population| ≤ O(√(d·log(d/δ) / n))

where d = min(d_X, d_Y).

**Interpretation:** Need n = Ω(d/ε²) samples for ε-accurate correlation estimation.

### Why Multimodal AI?

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer
import numpy as np

class MultimodalBenefits:
    """Demonstrate advantages of multimodal learning"""

    def __init__(self):
        self.modalities = ['vision', 'language', 'audio']

    def compare_approaches(self):
        """Compare unimodal vs multimodal performance"""

        results = {
            'vision_only': 75.3,  # Image classification
            'text_only': 78.1,    # Text classification
            'vision_text': 89.4,  # Multimodal classification
        }

        print("Performance Comparison:")
        for approach, acc in results.items():
            print(f"{approach:15s}: {acc:.1f}%")

        improvement = results['vision_text'] - max(
            results['vision_only'],
            results['text_only']
        )
        print(f"\nMultimodal improvement: +{improvement:.1f}%")

        return results

# Usage
demo = MultimodalBenefits()
demo.compare_approaches()
```

### Multimodal Fusion Strategies

#### Theoretical Analysis of Fusion Methods

**Theorem 13 (Early vs Late Fusion Trade-off - Ngiam et al., 2011):**

**Early Fusion:** Concatenate raw features X ⊕ Y, then process jointly.
- **Advantage:** Captures cross-modal interactions at all levels
- **Disadvantage:** High dimensionality d_X + d_Y, needs more data
- **Sample complexity:** O((d_X + d_Y)² / ε²)

**Late Fusion:** Process modalities separately, combine predictions.
- **Advantage:** Lower dimensional, modality-specific learning
- **Disadvantage:** Misses low-level interactions
- **Sample complexity:** O((d_X² + d_Y²) / ε²)

**Optimal Choice:**
- Small datasets: Late fusion (lower complexity)
- Large datasets with interaction: Early fusion
- Hybrid: Intermediate fusion at multiple levels

**Theorem 14 (Attention-Based Fusion - Bahdanau et al., 2015):**
Cross-modal attention weight for query Q from modality X and key-value from Y:

α_ij = exp(score(q_i, k_j)) / Σ_j' exp(score(q_i, k_j'))

where score(q, k) = q^T W k (bilinear) or q^T k (dot product).

**Attended output:**
c_i = Σ_j α_ij v_j

**Theorem 15 (Multi-Head Attention for Multimodal - Vaswani et al., 2017):**
With H attention heads:

MultiHead(Q, K, V) = Concat(head₁, ..., head_H) W^O

where head_h = Attention(Q W_h^Q, K W_h^K, V W_h^V)

**Capacity:** H heads capture H different interaction patterns.

**Sample Complexity:** O(H·d² / ε²) where d is head dimension.

**Optimal H:** Empirically H ∈ [8, 16] for vision-language tasks.

**Theorem 16 (Fusion Methods Comparison):**

| Method | Complexity | Sample Efficiency | Cross-Modal Interaction | Use Case |
|--------|------------|-------------------|------------------------|----------|
| **Concatenation** | O(d_X + d_Y) | Low (needs large data) | Full (all levels) | Large datasets |
| **Element-wise** | O(max(d_X, d_Y)) | High (simple) | Weak (same position) | Aligned features |
| **Bilinear** | O(d_X · d_Y) | Medium | Strong (pairwise) | Medium datasets |
| **Attention** | O(n·d) | High (selective) | Adaptive | Variable importance |
| **Tensor Fusion** | O(d_X · d_Y · d_Z) | Low (many params) | Very strong | Small feature dims |

**Theorem 17 (Tensor Fusion Network - Zadeh et al., 2017):**
Outer product of all modality representations:

Z = [1; z_X; z_Y; z_Z] ⊗ [1; z_X; z_Y; z_Z] ⊗ [1; z_X; z_Y; z_Z]

**Dimension:** (1 + d_X) × (1 + d_Y) × (1 + d_Z) = O(d³)

**Challenge:** Exponential growth with modalities → use low-rank approximation.

**Low-Rank Tensor Fusion:**
Z ≈ Σ_{r=1}^R u_r ⊗ v_r ⊗ w_r

**Parameters:** R(d_X + d_Y + d_Z) instead of d_X · d_Y · d_Z

**Sample Complexity Reduction:** O(R·d / ε²) vs O(d³ / ε²)

```python
class FusionStrategies(nn.Module):
    """Different ways to combine modalities"""

    def __init__(self, vision_dim=2048, text_dim=768, hidden_dim=512):
        super().__init__()

        # Early fusion layers
        self.early_fusion = nn.Linear(vision_dim + text_dim, hidden_dim)

        # Late fusion layers
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Hybrid fusion
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)

    def early_fusion(self, vision_feat, text_feat):
        """Concatenate features early, then process together"""
        combined = torch.cat([vision_feat, text_feat], dim=-1)
        fused = self.early_fusion(combined)
        return F.relu(fused)

    def late_fusion(self, vision_feat, text_feat):
        """Process modalities separately, combine at end"""
        vision_proj = F.relu(self.vision_proj(vision_feat))
        text_proj = F.relu(self.text_proj(text_feat))

        # Element-wise combination
        fused = vision_proj + text_proj  # or * for multiplication
        return fused

    def attention_fusion(self, vision_feat, text_feat):
        """Use attention to dynamically weight modalities"""
        vision_proj = self.vision_proj(vision_feat).unsqueeze(0)
        text_proj = self.text_proj(text_feat).unsqueeze(0)

        # Cross-attention: query=vision, key/value=text
        attended, weights = self.attention(vision_proj, text_proj, text_proj)
        return attended.squeeze(0), weights

# Test fusion strategies
fusion = FusionStrategies()
vision = torch.randn(8, 2048)
text = torch.randn(8, 768)

early = fusion.early_fusion(vision, text)
late = fusion.late_fusion(vision, text)
attn, weights = fusion.attention_fusion(vision, text)

print(f"Early fusion: {early.shape}")
print(f"Late fusion: {late.shape}")
print(f"Attention fusion: {attn.shape}")
```

---

## Vision-Language Models

Learn joint representations of images and text.

### Theoretical Foundations of CLIP

#### Contrastive Learning for Vision-Language Alignment

**Definition (CLIP Objective - Radford et al., 2021):**
Given N (image, text) pairs {(x_i, y_i)}_{i=1}^N, learn encoders f: 𝒳 → ℝ^d and g: 𝒴 → ℝ^d to maximize:

L_CLIP = (1/N) Σ_{i=1}^N [log(exp(sim(z_i^v, z_i^t)/τ) / Σ_j exp(sim(z_i^v, z_j^t)/τ)) +
                            log(exp(sim(z_i^t, z_i^v)/τ) / Σ_j exp(sim(z_j^v, z_i^t)/τ))]

where:
- z_i^v = f(x_i) / ||f(x_i)||: normalized image embedding
- z_i^t = g(y_i) / ||g(y_i)||: normalized text embedding
- sim(u, v) = u^T v: cosine similarity
- τ > 0: temperature parameter

**Theorem 7 (CLIP Information-Theoretic Justification):**
The CLIP loss approximates maximizing mutual information:

I(f(X); g(Y))

between visual and textual representations.

**Proof Sketch:**
CLIP loss is symmetric InfoNCE with N-1 negatives.
By Theorem 1 (InfoNCE bound from Self-Supervised Learning):

I(f(X); g(Y)) ≥ log(N) - L_CLIP

Thus minimizing L_CLIP maximizes lower bound on I(f(X); g(Y)). ∎

**Theorem 8 (Zero-Shot Transfer via Text Prompts):**
For classification with C classes, CLIP constructs text prompts:

t_c = "a photo of a [class_c]"

and classifies via:

ŷ = argmax_c sim(f(x), g(t_c))

**Generalization Bound:** With probability ≥ 1-δ:

E[ℓ_0-1(ŷ, y)] ≤ E[ℓ_CLIP] + O(√(d·log(C/δ) / N))

where ℓ_CLIP is contrastive loss, ℓ_0-1 is 0-1 loss.

**Interpretation:**
- Good CLIP training (low contrastive loss) → good zero-shot performance
- Sample complexity: N = O(d·log C / ε²) for ε-accurate zero-shot

**Theorem 9 (CLIP Batch Size and Temperature Effects - Chen et al., 2020):**

**Batch Size N:**
- Effective negatives: N-1 per sample
- Optimal: N ∈ [32K, 64K] for 400M pairs
- Too small (N < 1K): Insufficient negatives, poor alignment
- Too large (N > 100K): Diminishing returns, hardware constraints

**Temperature τ:**
- Controls sharpness of similarity distribution
- Optimal τ ∈ [0.01, 0.07]
- Too small (τ → 0): Overconfident, gradient vanishing
- Too large (τ → ∞): Uniform distribution, no learning

**Theorem 10 (CLIP Sample Efficiency - Radford et al., 2021):**
CLIP trained on 400M (image, text) pairs achieves:
- Zero-shot ImageNet: 76.2% (vs 88.4% supervised ResNet-50)
- Sample efficiency: ~10× fewer task-specific labeled samples for fine-tuning

**Data Scaling Law:**
Accuracy(N) ≈ α - β·N^{-γ}

where N is number of training pairs, γ ≈ 0.35 (power law).

**Example:** To improve from 76% to 80% zero-shot accuracy:
N_required ≈ 400M × (0.04/0.08)^{1/0.35} ≈ 2.8B pairs

**Theorem 11 (Cross-Modal Retrieval with CLIP):**
For image-to-text retrieval, rank texts by similarity:

rank(y_j | x_i) = -sim(f(x_i), g(y_j))

**Recall@K:** Probability that correct text is in top-K:

Recall@K = P(rank(y_i | x_i) ≤ K)

**Performance Bound:** With N training pairs and temperature τ:

Recall@K ≥ 1 - (1 - e^{1/τ}/N)^K

**Example:** For N=1M, τ=0.07, K=10:
Recall@10 ≥ 1 - (1 - 1.01/10^6)^10 ≈ 1 - e^{-10.1/10^6} ≈ 10^{-5} (very low)

**Practical Solution:** Use hard negatives mining to improve recall.

#### Prompt Engineering Theory

**Definition (Prompt Template):**
For class c, create text embedding via template T:

t_c = T(c) = f("a [descriptor] of a [class_c]")

where [descriptor] ∈ {"photo", "painting", "image", ...}

**Theorem 12 (Ensemble over Prompt Templates - Radford et al., 2021):**
Using M different templates and averaging:

z_c^t = (1/M) Σ_{m=1}^M g(T_m(c))

improves zero-shot accuracy by 3-5% on ImageNet.

**Proof of Improvement:**
Variance reduction via ensemble:

Var[z̄] = Var[z] / M   (if templates independent)

Lower variance → more robust class representations → better accuracy. ∎

**Optimal M:** M ∈ [7, 80] depending on dataset (diminishing returns beyond 80).

### CLIP Implementation

```python
class CLIP(nn.Module):
    """Contrastive Language-Image Pre-training"""

    def __init__(
        self,
        vision_encoder,
        text_encoder,
        embed_dim=512,
        temperature=0.07
    ):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature

        # Projection heads
        vision_dim = vision_encoder.output_dim
        text_dim = text_encoder.config.hidden_size

        self.vision_projection = nn.Linear(vision_dim, embed_dim)
        self.text_projection = nn.Linear(text_dim, embed_dim)

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def encode_image(self, images):
        """Encode images to embeddings"""
        image_features = self.vision_encoder(images)
        image_embeds = self.vision_projection(image_features)
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds

    def encode_text(self, input_ids, attention_mask):
        """Encode text to embeddings"""
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_embeds = self.text_projection(text_features)
        text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds

    def forward(self, images, input_ids, attention_mask):
        """
        Compute contrastive loss between images and text
        """
        # Encode both modalities
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids, attention_mask)

        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text

    def contrastive_loss(self, logits_per_image, logits_per_text):
        """Symmetric cross-entropy loss"""
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size).to(logits_per_image.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)

        loss = (loss_i + loss_t) / 2
        return loss

# Create CLIP model
from torchvision.models import resnet50
from transformers import BertModel, BertTokenizer

# Vision encoder
vision_encoder = resnet50(pretrained=True)
vision_encoder.output_dim = vision_encoder.fc.in_features
vision_encoder.fc = nn.Identity()

# Text encoder
text_encoder = BertModel.from_pretrained('bert-base-uncased')

# CLIP model
clip_model = CLIP(vision_encoder, text_encoder, embed_dim=512)

# Training
def train_clip(model, dataloader, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        total_loss = 0
        for images, input_ids, attention_mask in dataloader:
            images = images.cuda()
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

            # Forward pass
            logits_per_image, logits_per_text = model(
                images, input_ids, attention_mask
            )
            loss = model.contrastive_loss(logits_per_image, logits_per_text)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

### Zero-Shot Classification with CLIP

```python
class ZeroShotCLIP:
    """Zero-shot image classification using CLIP"""

    def __init__(self, clip_model, tokenizer):
        self.model = clip_model
        self.tokenizer = tokenizer
        self.model.eval()

    def create_text_prompts(self, class_names):
        """Create text prompts for each class"""
        prompts = [f"a photo of a {name}" for name in class_names]
        return prompts

    def encode_text_prompts(self, prompts):
        """Encode text prompts to embeddings"""
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            text_embeds = self.model.encode_text(
                encoded['input_ids'].cuda(),
                encoded['attention_mask'].cuda()
            )

        return text_embeds

    def classify(self, image, class_names):
        """Zero-shot classification"""
        # Prepare text prompts
        prompts = self.create_text_prompts(class_names)
        text_embeds = self.encode_text_prompts(prompts)

        # Encode image
        image = image.unsqueeze(0).cuda()
        with torch.no_grad():
            image_embed = self.model.encode_image(image)

        # Compute similarities
        similarities = (100.0 * image_embed @ text_embeds.T).softmax(dim=-1)
        similarities = similarities.cpu().numpy()[0]

        # Get top predictions
        results = [
            (class_names[i], similarities[i])
            for i in np.argsort(similarities)[::-1]
        ]

        return results

# Usage
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
zero_shot = ZeroShotCLIP(clip_model, tokenizer)

# Classify image
class_names = ['dog', 'cat', 'bird', 'fish', 'horse']
# image = load_image('test.jpg')
# results = zero_shot.classify(image, class_names)
# for name, prob in results:
#     print(f"{name}: {prob:.2%}")
```

### Visual Question Answering (VQA)

```python
class VQAModel(nn.Module):
    """Visual Question Answering using vision-language fusion"""

    def __init__(
        self,
        vision_encoder,
        text_encoder,
        num_answers=3129,  # VQA dataset has 3129 common answers
        hidden_dim=512
    ):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        vision_dim = vision_encoder.output_dim
        text_dim = text_encoder.config.hidden_size

        # Multimodal fusion
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Attention-based fusion
        self.multimodal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

        # Answer classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_answers)
        )

    def forward(self, images, input_ids, attention_mask):
        """
        Args:
            images: Image tensors (B, C, H, W)
            input_ids: Question token IDs (B, L)
            attention_mask: Question attention mask (B, L)
        """
        # Encode image
        image_features = self.vision_encoder(images)
        image_proj = self.vision_proj(image_features).unsqueeze(1)  # (B, 1, D)

        # Encode question
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state  # (B, L, D)
        text_proj = self.text_proj(text_features)

        # Cross-modal attention (image attends to question)
        fused_features, _ = self.multimodal_attention(
            image_proj, text_proj, text_proj
        )
        fused_features = fused_features.squeeze(1)  # (B, D)

        # Classify answer
        logits = self.classifier(fused_features)

        return logits

# Training VQA
def train_vqa(model, train_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, questions, attention_mask, answers in train_loader:
            images = images.cuda()
            questions = questions.cuda()
            attention_mask = attention_mask.cuda()
            answers = answers.cuda()

            # Forward pass
            logits = model(images, questions, attention_mask)
            loss = criterion(logits, answers)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

---

## Audio-Visual Learning

Combine audio and visual information for richer understanding.

### Audio-Visual Synchronization

```python
class AudioVisualSync(nn.Module):
    """Learn audio-visual correspondences"""

    def __init__(self, visual_dim=512, audio_dim=128, embed_dim=256):
        super().__init__()

        # Visual encoder (CNN for video frames)
        self.visual_encoder = models.resnet18(pretrained=True)
        self.visual_encoder.fc = nn.Linear(
            self.visual_encoder.fc.in_features, visual_dim
        )

        # Audio encoder (CNN for spectrograms)
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, audio_dim)
        )

        # Projection to common space
        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)

    def encode_visual(self, frames):
        """Encode video frames"""
        visual_features = self.visual_encoder(frames)
        visual_embed = self.visual_proj(visual_features)
        return F.normalize(visual_embed, dim=-1)

    def encode_audio(self, spectrograms):
        """Encode audio spectrograms"""
        audio_features = self.audio_encoder(spectrograms)
        audio_embed = self.audio_proj(audio_features)
        return F.normalize(audio_embed, dim=-1)

    def forward(self, frames, spectrograms):
        """Compute audio-visual similarity"""
        visual_embed = self.encode_visual(frames)
        audio_embed = self.encode_audio(spectrograms)

        # Compute similarity
        similarity = torch.mm(visual_embed, audio_embed.T)
        return similarity

    def contrastive_loss(self, similarity):
        """Contrastive loss for synchronized pairs"""
        batch_size = similarity.shape[0]
        labels = torch.arange(batch_size).to(similarity.device)

        # Cross-entropy on both directions
        loss_v = F.cross_entropy(similarity, labels)
        loss_a = F.cross_entropy(similarity.T, labels)

        return (loss_v + loss_a) / 2

# Training
def train_audio_visual(model, dataloader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for frames, spectrograms in dataloader:
            frames = frames.cuda()
            spectrograms = spectrograms.cuda()

            # Forward pass
            similarity = model(frames, spectrograms)
            loss = model.contrastive_loss(similarity)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Sound Source Localization

```python
class SoundSourceLocalizer(nn.Module):
    """Locate sound sources in video frames"""

    def __init__(self):
        super().__init__()

        # Visual stream
        self.visual_cnn = models.resnet18(pretrained=True)
        self.visual_cnn.fc = nn.Identity()

        # Audio stream
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Conv2d(512 + 128, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, frames, spectrograms):
        """
        Args:
            frames: (B, C, H, W) video frames
            spectrograms: (B, 1, F, T) audio spectrograms
        Returns:
            heatmap: (B, 1, H, W) sound localization heatmap
        """
        # Extract visual features
        visual_feat = self.visual_cnn(frames)  # (B, 512)
        B, C = visual_feat.shape
        H, W = frames.shape[2] // 32, frames.shape[3] // 32
        visual_feat = visual_feat.view(B, C, 1, 1).expand(B, C, H, W)

        # Extract audio features
        audio_feat = self.audio_cnn(spectrograms)  # (B, 128, H, W)

        # Fuse and localize
        combined = torch.cat([visual_feat, audio_feat], dim=1)
        heatmap = self.fusion(combined)

        # Upsample to original size
        heatmap = F.interpolate(
            heatmap,
            size=frames.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return heatmap
```

---

## Multimodal Transformers

Unified transformer architectures for multiple modalities.

### ViLT (Vision-and-Language Transformer)

```python
class ViLT(nn.Module):
    """Vision-and-Language Transformer without CNN"""

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_patches=196,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        vocab_size=30522,
        max_seq_length=40
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = num_patches

        # Patch embedding (like ViT)
        self.patch_embed = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        )

        # Text embedding
        self.text_embed = nn.Embedding(vocab_size, hidden_size)

        # Positional embeddings
        self.image_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size)
        )
        self.text_pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_length, hidden_size)
        )

        # Modality type embeddings
        self.image_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Unified transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Task-specific heads
        self.itm_head = nn.Linear(hidden_size, 2)  # Image-text matching
        self.mlm_head = nn.Linear(hidden_size, vocab_size)  # Masked LM

    def forward_image(self, images):
        """Process images to patches"""
        B = images.shape[0]

        # Patch embedding
        patches = self.patch_embed(images)  # (B, D, H, W)
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add positional and type embeddings
        patches = patches + self.image_pos_embed + self.image_type_embed

        return patches

    def forward_text(self, input_ids):
        """Process text"""
        B, L = input_ids.shape

        # Text embedding
        text_embeds = self.text_embed(input_ids)

        # Add positional and type embeddings
        text_embeds = text_embeds + self.text_pos_embed[:, :L, :] + self.text_type_embed

        return text_embeds

    def forward(self, images, input_ids, task='itm'):
        """
        Args:
            images: (B, 3, H, W)
            input_ids: (B, L)
            task: 'itm' (image-text matching) or 'mlm' (masked LM)
        """
        # Process both modalities
        image_embeds = self.forward_image(images)
        text_embeds = self.forward_text(input_ids)

        # Concatenate modalities
        multimodal_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        # Unified transformer encoding
        encoded = self.transformer(multimodal_embeds)

        # Task-specific output
        if task == 'itm':
            # Use [CLS] token (first text token)
            cls_output = encoded[:, self.num_patches, :]
            logits = self.itm_head(cls_output)
            return logits
        elif task == 'mlm':
            # Predict masked text tokens
            text_output = encoded[:, self.num_patches:, :]
            logits = self.mlm_head(text_output)
            return logits
        else:
            return encoded

# Training ViLT
def train_vilt(model, dataloader, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for batch in dataloader:
            images = batch['images'].cuda()
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            task = batch['task']

            # Forward pass
            logits = model(images, input_ids, task=task)

            if task == 'itm':
                loss = F.cross_entropy(logits, labels)
            elif task == 'mlm':
                loss = F.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    labels.view(-1)
                )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## Cross-Modal Retrieval

Retrieve relevant items across modalities.

### Image-Text Retrieval

```python
class CrossModalRetrieval:
    """Retrieve images from text queries and vice versa"""

    def __init__(self, model):
        """
        Args:
            model: Trained vision-language model (e.g., CLIP)
        """
        self.model = model
        self.model.eval()

        self.image_embeddings = None
        self.text_embeddings = None
        self.image_ids = None
        self.text_ids = None

    def index_images(self, image_loader):
        """Pre-compute embeddings for image database"""
        embeddings = []
        ids = []

        with torch.no_grad():
            for images, image_ids_batch in image_loader:
                images = images.cuda()
                embeds = self.model.encode_image(images)
                embeddings.append(embeds.cpu())
                ids.extend(image_ids_batch)

        self.image_embeddings = torch.cat(embeddings, dim=0)
        self.image_ids = ids

    def index_texts(self, text_loader):
        """Pre-compute embeddings for text database"""
        embeddings = []
        ids = []

        with torch.no_grad():
            for input_ids, attention_mask, text_ids_batch in text_loader:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()

                embeds = self.model.encode_text(input_ids, attention_mask)
                embeddings.append(embeds.cpu())
                ids.extend(text_ids_batch)

        self.text_embeddings = torch.cat(embeddings, dim=0)
        self.text_ids = ids

    def retrieve_images(self, text_query, top_k=5):
        """Retrieve top-k images for text query"""
        # Encode query
        with torch.no_grad():
            query_embed = self.model.encode_text(
                text_query['input_ids'].cuda(),
                text_query['attention_mask'].cuda()
            )

        # Compute similarities
        similarities = query_embed @ self.image_embeddings.T
        similarities = similarities.cpu().numpy()[0]

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [
            (self.image_ids[i], similarities[i])
            for i in top_indices
        ]

        return results

    def retrieve_texts(self, image, top_k=5):
        """Retrieve top-k texts for image query"""
        # Encode query
        with torch.no_grad():
            query_embed = self.model.encode_image(image.unsqueeze(0).cuda())

        # Compute similarities
        similarities = query_embed @ self.text_embeddings.T
        similarities = similarities.cpu().numpy()[0]

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [
            (self.text_ids[i], similarities[i])
            for i in top_indices
        ]

        return results

# Usage
retrieval = CrossModalRetrieval(clip_model)

# Index databases
# retrieval.index_images(image_loader)
# retrieval.index_texts(text_loader)

# Retrieve images
# text_query = tokenizer("a dog playing in the park", return_tensors='pt')
# results = retrieval.retrieve_images(text_query, top_k=5)
```

---

## Multimodal Generation

Generate content in one modality conditioned on another.

### Text-to-Image Generation

```python
class TextToImageGenerator(nn.Module):
    """Generate images from text descriptions"""

    def __init__(self, text_encoder, diffusion_model):
        super().__init__()
        self.text_encoder = text_encoder
        self.diffusion_model = diffusion_model

    def encode_text(self, prompts, tokenizer):
        """Encode text prompts to conditioning vectors"""
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            text_embeds = self.text_encoder(
                input_ids=encoded['input_ids'].cuda(),
                attention_mask=encoded['attention_mask'].cuda()
            ).last_hidden_state

        return text_embeds

    @torch.no_grad()
    def generate(
        self,
        prompts,
        tokenizer,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=512,
        width=512
    ):
        """Generate images from text prompts"""

        # Encode text
        text_embeds = self.encode_text(prompts, tokenizer)

        # Unconditional embeddings for classifier-free guidance
        uncond_embeds = self.encode_text(
            [""] * len(prompts), tokenizer
        )

        # Initialize latents
        latents = torch.randn(
            len(prompts), 4, height // 8, width // 8
        ).cuda()

        # Diffusion sampling
        self.diffusion_model.set_timesteps(num_inference_steps)

        for t in self.diffusion_model.timesteps:
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.diffusion_model.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=torch.cat([uncond_embeds, text_embeds])
                ).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # Denoise step
            latents = self.diffusion_model.scheduler.step(
                noise_pred, t, latents
            ).prev_sample

        # Decode latents to images
        images = self.diffusion_model.vae.decode(latents / 0.18215).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        return images
```

### Image Captioning

```python
class ImageCaptioner(nn.Module):
    """Generate text captions for images"""

    def __init__(self, vision_encoder, text_decoder, vocab_size, max_length=20):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.max_length = max_length

        # Project vision features to text decoder dimension
        vision_dim = vision_encoder.output_dim
        text_dim = text_decoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_dim, text_dim)

    def forward(self, images, captions, attention_mask):
        """
        Training forward pass

        Args:
            images: (B, C, H, W)
            captions: (B, L) target caption token IDs
            attention_mask: (B, L)
        """
        # Encode image
        image_features = self.vision_encoder(images)
        image_embeds = self.vision_proj(image_features).unsqueeze(1)  # (B, 1, D)

        # Decoder forward pass with image as context
        outputs = self.text_decoder(
            input_ids=captions[:, :-1],  # Shift right
            attention_mask=attention_mask[:, :-1],
            encoder_hidden_states=image_embeds,
            labels=captions[:, 1:]  # Shift left for target
        )

        return outputs.loss

    @torch.no_grad()
    def generate_caption(self, image, tokenizer, num_beams=5):
        """Generate caption for image using beam search"""

        # Encode image
        image_features = self.vision_encoder(image.unsqueeze(0))
        image_embeds = self.vision_proj(image_features).unsqueeze(1)

        # Start with [CLS] token
        input_ids = torch.tensor([[tokenizer.cls_token_id]]).cuda()

        # Generate with beam search
        outputs = self.text_decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=image_embeds,
            max_length=self.max_length,
            num_beams=num_beams,
            early_stopping=True
        )

        # Decode to text
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return caption

# Training
def train_captioner(model, dataloader, epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        for images, captions, attention_mask in dataloader:
            images = images.cuda()
            captions = captions.cuda()
            attention_mask = attention_mask.cuda()

            # Forward pass
            loss = model(images, captions, attention_mask)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## Video Understanding

Process temporal sequences with multiple modalities.

### Video Classification

```python
class VideoClassifier(nn.Module):
    """Classify videos using spatio-temporal features"""

    def __init__(self, num_classes, num_frames=16):
        super().__init__()

        self.num_frames = num_frames

        # Spatial feature extractor (per-frame)
        self.spatial_cnn = models.resnet50(pretrained=True)
        self.spatial_cnn.fc = nn.Identity()

        # Temporal aggregation (LSTM)
        self.temporal_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        # Classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, videos):
        """
        Args:
            videos: (B, T, C, H, W) where T is number of frames
        """
        B, T, C, H, W = videos.shape

        # Extract spatial features for each frame
        frames = videos.view(B * T, C, H, W)
        spatial_features = self.spatial_cnn(frames)  # (B*T, 2048)
        spatial_features = spatial_features.view(B, T, -1)  # (B, T, 2048)

        # Temporal aggregation
        temporal_features, _ = self.temporal_lstm(spatial_features)
        video_features = temporal_features[:, -1, :]  # Use last output

        # Classify
        logits = self.classifier(video_features)

        return logits

# 3D CNN alternative
class Video3DCNN(nn.Module):
    """3D CNN for video classification"""

    def __init__(self, num_classes):
        super().__init__()

        self.conv3d = nn.Sequential(
            # Conv1
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),

            # Conv2
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)),

            # Conv3
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)),

            # Global average pooling
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, videos):
        """
        Args:
            videos: (B, C, T, H, W)
        """
        features = self.conv3d(videos)
        logits = self.classifier(features)
        return logits
```

---

## Production Implementation

### Complete Multimodal Pipeline

```python
class MultimodalPipeline:
    """Production-ready multimodal AI system"""

    def __init__(self):
        # Load pre-trained models
        from transformers import CLIPModel, CLIPProcessor

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.clip_model.eval()
        self.clip_model.cuda()

    def process_image(self, image_path):
        """Process image file"""
        from PIL import Image

        image = Image.open(image_path).convert('RGB')
        inputs = self.clip_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(
                inputs['pixel_values'].cuda()
            )

        return image_features.cpu().numpy()

    def process_text(self, text):
        """Process text"""
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(
                inputs['input_ids'].cuda(),
                inputs['attention_mask'].cuda()
            )

        return text_features.cpu().numpy()

    def compute_similarity(self, image_path, texts):
        """Compute image-text similarity"""
        # Process inputs
        image_features = self.process_image(image_path)
        text_features = [self.process_text(text) for text in texts]
        text_features = np.vstack(text_features)

        # Compute similarities
        image_features = image_features / np.linalg.norm(image_features)
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

        similarities = np.dot(text_features, image_features.T).flatten()

        # Return ranked results
        results = [
            (texts[i], float(similarities[i]))
            for i in np.argsort(similarities)[::-1]
        ]

        return results

# FastAPI deployment
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import io
from PIL import Image

app = FastAPI()
pipeline = MultimodalPipeline()

@app.post("/classify-image")
async def classify_image(
    image: UploadFile = File(...),
    classes: str = Form(...)
):
    """Zero-shot image classification"""
    # Load image
    image_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(image_bytes))

    # Save temporarily
    temp_path = "/tmp/temp_image.jpg"
    image_pil.save(temp_path)

    # Process
    class_list = [c.strip() for c in classes.split(',')]
    results = pipeline.compute_similarity(temp_path, class_list)

    return JSONResponse(content={"predictions": results})

@app.post("/retrieve-images")
async def retrieve_images(query: str = Form(...)):
    """Retrieve images matching text query"""
    # Process query
    query_features = pipeline.process_text(query)

    # Search in database (simplified)
    # In production: use vector database like FAISS or Pinecone

    return JSONResponse(content={"message": "Image retrieval endpoint"})
```

---

## Best Practices

### 1. Data Preprocessing

```python
class MultimodalDataset(torch.utils.data.Dataset):
    """Dataset for multimodal learning"""

    def __init__(self, image_paths, texts, labels, tokenizer):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and transform image
        from PIL import Image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.image_transform(image)

        # Tokenize text
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=77,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': self.labels[idx]
        }
```

### 2. Model Evaluation

```python
def evaluate_multimodal(model, dataloader):
    """Comprehensive evaluation"""
    model.eval()

    all_image_embeds = []
    all_text_embeds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].cuda()
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['label']

            # Get embeddings
            image_embeds = model.encode_image(images)
            text_embeds = model.encode_text(input_ids, attention_mask)

            all_image_embeds.append(image_embeds.cpu())
            all_text_embeds.append(text_embeds.cpu())
            all_labels.append(labels)

    # Concatenate
    image_embeds = torch.cat(all_image_embeds)
    text_embeds = torch.cat(all_text_embeds)
    labels = torch.cat(all_labels)

    # Compute metrics
    # 1. Image-to-text retrieval
    i2t_recall = compute_retrieval_recall(image_embeds, text_embeds, labels)

    # 2. Text-to-image retrieval
    t2i_recall = compute_retrieval_recall(text_embeds, image_embeds, labels)

    print(f"Image-to-Text Recall@1: {i2t_recall[0]:.2%}")
    print(f"Text-to-Image Recall@1: {t2i_recall[0]:.2%}")

def compute_retrieval_recall(query_embeds, target_embeds, labels, k_values=[1, 5, 10]):
    """Compute Recall@K for retrieval"""
    similarities = query_embeds @ target_embeds.T

    recalls = []
    for k in k_values:
        correct = 0
        for i in range(len(query_embeds)):
            top_k_indices = similarities[i].topk(k).indices
            if labels[i] in labels[top_k_indices]:
                correct += 1
        recalls.append(correct / len(query_embeds))

    return recalls
```

---

## Summary

Multimodal AI combines information from vision, language, and audio:

1. **Vision-Language Models** (CLIP, ViLT): Joint understanding
2. **Audio-Visual Learning**: Synchronization and localization
3. **Cross-Modal Retrieval**: Search across modalities
4. **Multimodal Generation**: Text-to-image, image captioning
5. **Video Understanding**: Spatio-temporal modeling

**Key Applications:**
- Visual question answering
- Image/video captioning
- Cross-modal search
- Content generation
- Video analysis

**Best Practices:**
- Use appropriate fusion strategies
- Pre-train on large-scale data
- Evaluate on multiple downstream tasks
- Handle modality imbalance
