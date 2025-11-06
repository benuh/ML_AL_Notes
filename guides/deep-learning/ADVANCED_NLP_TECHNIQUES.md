# Advanced NLP Techniques: Complete Implementation Guide

## Table of Contents
1. [Advanced Transformer Architectures](#advanced-transformer-architectures)
2. [Efficient Transformers & Long-Context Models](#efficient-transformers)
3. [Instruction Tuning & Alignment](#instruction-tuning)
4. [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation)
5. [Multi-Task Learning & Mixture of Experts](#multi-task-learning)
6. [Text Generation Strategies](#text-generation-strategies)
7. [Named Entity Recognition & Information Extraction](#ner-and-ie)
8. [Dialogue Systems & Conversational AI](#dialogue-systems)

---

## Advanced Transformer Architectures

State-of-the-art transformer variants and optimizations.

### Attention Mechanism Mathematics

**Mathematical Foundation:**

**1. Scaled Dot-Product Attention**
```
Given:
- Query: Q ∈ ℝ^(n×d_k)
- Key: K ∈ ℝ^(m×d_k)
- Value: V ∈ ℝ^(m×d_v)

Attention(Q, K, V) = softmax(QK^T / √d_k) V

Components:
1. QK^T: Similarity scores ∈ ℝ^(n×m)
   - Each entry (i,j): similarity between query i and key j

2. 1/√d_k: Scaling factor
   - Without scaling: QK^T has variance d_k (problematic for large d_k)
   - Softmax becomes saturated → small gradients
   - √d_k normalizes variance to 1

3. softmax: Attention weights ∈ ℝ^(n×m)
   - α_ij = exp(q_i·k_j/√d_k) / Σ_j exp(q_i·k_j/√d_k)
   - Properties: Σ_j α_ij = 1, α_ij ≥ 0

4. Final output: Weighted sum of values ∈ ℝ^(n×d_v)
   - out_i = Σ_j α_ij · v_j

Complexity:
- Time: O(n·m·d_k + n·m·d_v) = O(n·m·d)
  - QK^T: O(n·m·d_k)
  - softmax(·)V: O(n·m·d_v)
  - For self-attention (n=m): O(n²·d)
- Space: O(n·m) for attention matrix
```

**2. Multi-Head Attention**
```
Intuition: Different representation subspaces learn different patterns
- Heads attend to different positions/relationships
- Some heads: syntactic relationships (subject-verb)
- Some heads: semantic relationships (coreference)
- Some heads: positional patterns (local vs global)

Mathematical Formulation:

Single Head:
head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

where:
- W^Q_i ∈ ℝ^(d_model × d_k): Query projection for head i
- W^K_i ∈ ℝ^(d_model × d_k): Key projection for head i
- W^V_i ∈ ℝ^(d_model × d_v): Value projection for head i
- d_k = d_v = d_model / h (typically)
- h: number of heads

Multi-Head Attention:
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where W^O ∈ ℝ^(h·d_v × d_model): Output projection

Parameter Count:
- Query weights: h · d_model · d_k
- Key weights: h · d_model · d_k
- Value weights: h · d_model · d_v
- Output weights: h·d_v · d_model
- Total: 4·d_model² (when d_k = d_v = d_model/h)

Complexity:
- Time: h · O(n²·d_k)
- Space: h · O(n²)

Benefits:
1. Parallel computation across heads
2. Richer representations (ensemble of attention patterns)
3. Better gradient flow (multiple paths)
```

**3. Self-Attention vs Cross-Attention**
```
Self-Attention:
Q = K = V = X  (same sequence)

Example: "The cat sat on the mat"
- Each word attends to all words (including itself)
- Learns contextual relationships within sequence
- Used in: Encoder layers, Decoder masked self-attention

Cross-Attention:
Q ≠ K = V  (different sequences)

Example: Translation (English → French)
- Q: Decoder hidden states (French being generated)
- K, V: Encoder hidden states (English source)
- Decoder attends to relevant source words
- Used in: Encoder-Decoder models (seq2seq, T5)

Masked Self-Attention (Causal):
Attention(Q, K, V) = softmax(mask(QK^T / √d_k)) V

where mask sets future positions to -∞:
mask_ij = {0 if j ≤ i, -∞ if j > i}

Result: Position i can only attend to positions ≤ i
Used in: Autoregressive generation (GPT, decoder)
```

**4. Attention Variants**

**Additive Attention (Bahdanau):**
```
score(q, k) = v^T tanh(W_q q + W_k k)

where:
- W_q ∈ ℝ^(d_a × d_k)
- W_k ∈ ℝ^(d_a × d_k)
- v ∈ ℝ^d_a
- d_a: attention dimension

Complexity: O(n·m·d_a)
Advantage: Can learn non-linear interactions
Disadvantage: More parameters, slower
```

**Multiplicative Attention (Luong):**
```
score(q, k) = q^T W k  or  q^T k  (dot product)

Complexity: O(n·m·d_k)
Advantage: Faster, fewer parameters
Used in: Modern transformers (scaled dot-product)
```

**5. Attention Interpretations**

**Information Retrieval Perspective:**
```
Query: "What are you looking for?"
Key: "What do I have?"
Value: "What do I actually return?"

Soft dictionary lookup:
- Keys: Dictionary entries
- Values: Associated content
- Query: Search term
- Attention weights: Soft match scores
- Output: Weighted combination of relevant values
```

**Kernel Perspective:**
```
Attention as kernel smoothing:

out_i = Σ_j k(q_i, k_j) v_j / Σ_j k(q_i, k_j)

where k(q, k) = exp(q^T k / √d_k) is RBF-like kernel

Properties:
- Localization: High weights for similar q, k
- Smoothness: Differentiable
- Normalization: Softmax ensures Σ weights = 1
```

**6. Theoretical Properties**

**Universal Approximation:**
```
Theorem (Yun et al., 2020):
Transformers with sufficient depth can approximate any
continuous sequence-to-sequence function arbitrarily well.

Depth requirement: O(n) layers for length-n sequences
(compared to O(n²) for CNNs/RNNs)
```

**Expressiveness:**
```
Single-head attention:
- Rank ≤ min(d_k, n)
- Limited expressiveness

Multi-head attention:
- Effective rank ≤ h·min(d_k, n)
- Each head learns different subspace
- Higher capacity
```

**7. Computational Complexity Summary**

```
Operation              | Time           | Space
-----------------------|----------------|--------
Standard Attention     | O(n² · d)      | O(n²)
Multi-Head (h heads)   | O(h·n²·d)      | O(h·n²)
Linear Attention       | O(n·d²)        | O(d²)
Sparse Attention       | O(n·√n·d)      | O(n·√n)
Flash Attention        | O(n²·d)        | O(n)    ← Memory efficient!

where:
- n: sequence length
- d: model dimension
- h: number of heads
```

**8. Gradient Flow in Attention**

```
Backpropagation through attention:

∂L/∂V = A^T · ∂L/∂O
∂L/∂Q = (∂L/∂A · V^T) · (1/√d_k) · K
∂L/∂K = (∂L/∂A^T · V^T) · (1/√d_k) · Q

where:
- A = softmax(QK^T/√d_k): Attention weights
- O: Output

Key properties:
1. Direct path from output to all inputs (V, Q, K)
2. No gradient vanishing (unlike RNNs)
3. Gradient magnitude controlled by softmax
4. Enables training very deep models (GPT-3: 96 layers)
```

### GPT Architecture with Flash Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FlashAttention(nn.Module):
    """
    Flash Attention - Memory-efficient attention computation

    Mathematical Foundation:
    Standard Attention: Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Standard Complexity:
    - Time: O(n² · d) where n = seq_len, d = model_dim
    - Space: O(n²) for attention matrix (bottleneck!)

    Flash Attention Innovation:
    - Tiling: Divide Q, K, V into blocks that fit in SRAM
    - Recomputation: Don't store full attention matrix
    - Online softmax: Compute softmax incrementally

    Optimized Complexity:
    - Time: O(n² · d) (same, but faster wall-clock time)
    - Space: O(n) (down from O(n²)!) - enables 10-100x longer sequences

    Speed Improvement:
    - 2-4x faster on A100 GPU due to memory hierarchy optimization
    - Reduces HBM (slow) accesses, increases SRAM (fast) usage

    Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
    Attention with IO-Awareness", NeurIPS 2022
    """

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/√d_k for scaled dot-product attention
        self.dropout = dropout

    def forward(self, q, k, v, attention_mask=None):
        """
        Args:
            q, k, v: (B, num_heads, seq_len, head_dim)
            attention_mask: (B, seq_len) or (B, 1, seq_len, seq_len)
        """
        B, num_heads, seq_len, head_dim = q.shape

        # Standard attention computation (Flash Attention optimized in CUDA)
        # This is a simplified Python version - actual implementation uses custom CUDA kernels

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        # Apply causal mask (for autoregressive generation)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand mask
                attention_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores + attention_mask

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)

        if self.dropout > 0:
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Apply attention to values
        output = torch.matmul(attn_probs, v)  # (B, H, L, head_dim)

        return output

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from RoFormer

    Mathematical Foundation:

    Traditional absolute position encoding:
    x_i = x_i + PE(i)  (additive, position i)

    RoPE (multiplicative in complex space):
    For query q and key k at positions m and n:

    f_q(x_m, m) = R_θ,m · W_q · x_m
    f_k(x_n, n) = R_θ,n · W_k · x_n

    where R_θ,m is 2D rotation matrix:
    R_θ,m = [cos(mθ)  -sin(mθ)]
            [sin(mθ)   cos(mθ)]

    For dimension d, θ_i = 10000^(-2i/d) for i = 0, 1, ..., d/2-1

    Key Properties:
    1. Relative position encoding:
       <f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m-n)
       Inner product only depends on relative distance m-n!

    2. Decays with distance:
       Attention between tokens naturally decreases with distance

    3. Efficient extrapolation:
       Can handle sequences longer than training length

    Advantages over learned position embeddings:
    - ✓ No learned parameters for positions
    - ✓ Better relative position modeling
    - ✓ Extends to arbitrary sequence lengths
    - ✓ Used in GPT-Neo, GPT-J, LLaMA, PaLM

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
    Position Embedding", arXiv:2104.09864, 2021
    """

    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        # Compute inverse frequencies: θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

        # Build cache
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        """
        Args:
            x: (batch, num_heads, seq_len, head_dim)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings to q and k"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadAttentionWithRoPE(nn.Module):
    """Multi-head attention with Rotary Position Embeddings"""

    def __init__(self, dim, num_heads, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, max_seq_len=max_seq_len
        )
        self.flash_attn = FlashAttention(dim, num_heads, dropout)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (batch, seq_len, dim)
            attention_mask: (batch, seq_len)
        """
        B, L, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        cos, sin = self.rotary_emb(q, seq_len=L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Apply attention
        attn_output = self.flash_attn(q, k, v, attention_mask)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        output = self.out_proj(attn_output)

        return output

class SwiGLU(nn.Module):
    """
    SwiGLU activation from "GLU Variants Improve Transformer"
    Used in PaLM, LLaMA, and other modern LLMs
    """

    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * 8/3)  # Common ratio

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """SwiGLU(x) = Swish(W1·x) ⊙ W3·x"""
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class GPTBlock(nn.Module):
    """GPT Transformer block with modern improvements"""

    def __init__(
        self,
        dim,
        num_heads,
        dropout=0.1,
        max_seq_len=2048,
        use_swiglu=True
    ):
        super().__init__()

        # Pre-normalization (as in GPT-3 and beyond)
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionWithRoPE(
            dim, num_heads, dropout, max_seq_len
        )

        self.ln2 = nn.LayerNorm(dim)

        if use_swiglu:
            self.mlp = SwiGLU(dim, dropout=dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout)
            )

    def forward(self, x, attention_mask=None):
        # Attention with residual
        x = x + self.attn(self.ln1(x), attention_mask)

        # MLP with residual
        x = x + self.mlp(self.ln2(x))

        return x

class ModernGPT(nn.Module):
    """
    Modern GPT architecture with:
    - Rotary Position Embeddings
    - Pre-normalization
    - SwiGLU activation
    - Flash Attention
    """

    def __init__(
        self,
        vocab_size=50257,
        max_seq_len=2048,
        num_layers=24,
        dim=1024,
        num_heads=16,
        dropout=0.1
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(dim, num_heads, dropout, max_seq_len)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(dim)

        # Output projection
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.token_emb.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) where 0 = attend, 1 = mask
        """
        # Token embeddings
        x = self.token_emb(input_ids)

        # Create causal attention mask
        if attention_mask is None:
            seq_len = input_ids.size(1)
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len), diagonal=1
            ).bool().to(input_ids.device)
            attention_mask = attention_mask * -1e9  # Large negative value

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=None
    ):
        """
        Generate text autoregressively

        Args:
            input_ids: (batch, seq_len) prompt tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus (top-p) sampling
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits = self(input_ids_cond)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

# Training with gradient accumulation and mixed precision
class GPTTrainer:
    """Trainer for modern GPT models"""

    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        learning_rate=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        warmup_steps=2000,
        max_steps=100000,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        use_amp=True
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Optimizer with weight decay only on certain parameters
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'ln', 'norm']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=learning_rate, betas=betas)

        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        self.step = 0

    def train_step(self, batch):
        """Single training step"""
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()

        # Forward pass with automatic mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logits = self.model(input_ids)

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.gradient_accumulation_steps

    def train(self, max_steps=100000):
        """Training loop"""
        self.model.train()

        running_loss = 0.0
        step_in_epoch = 0

        while self.step < max_steps:
            for batch in self.train_dataloader:
                loss = self.train_step(batch)
                running_loss += loss
                step_in_epoch += 1

                # Update weights after accumulation
                if step_in_epoch % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                    # Optimizer step
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    self.step += 1

                    # Logging
                    if self.step % 100 == 0:
                        avg_loss = running_loss / 100
                        lr = self.scheduler.get_last_lr()[0]
                        print(f"Step {self.step}, Loss: {avg_loss:.4f}, LR: {lr:.6f}")
                        running_loss = 0.0

                    # Validation
                    if self.step % 1000 == 0:
                        val_loss = self.validate()
                        print(f"Validation Loss: {val_loss:.4f}")
                        self.model.train()

                    if self.step >= max_steps:
                        break

            if self.step >= max_steps:
                break

    @torch.no_grad()
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()

            logits = self.model(input_ids)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches
```

### BERT with Whole Word Masking and Dynamic Masking

```python
class DynamicMaskingBERT(nn.Module):
    """
    BERT with dynamic masking - generates new masks each epoch
    Includes whole word masking and span masking
    """

    def __init__(
        self,
        vocab_size=30522,
        max_seq_len=512,
        num_layers=12,
        dim=768,
        num_heads=12,
        dropout=0.1
    ):
        super().__init__()

        # Token + Position + Segment embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.segment_emb = nn.Embedding(2, dim)

        self.emb_norm = nn.LayerNorm(dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Prediction heads
        self.mlm_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, vocab_size)
        )

        self.nsp_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 2)  # Is next / Not next
        )

        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size - 1  # Assuming [MASK] is last token

    def forward(self, input_ids, segment_ids=None, masked_positions=None):
        """
        Args:
            input_ids: (batch, seq_len)
            segment_ids: (batch, seq_len)
            masked_positions: (batch, seq_len) binary mask for MLM
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_embeds = self.token_emb(input_ids)

        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(input_ids.device)
        pos_embeds = self.pos_emb(positions)

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        segment_embeds = self.segment_emb(segment_ids)

        embeddings = token_embeds + pos_embeds + segment_embeds
        embeddings = self.emb_dropout(self.emb_norm(embeddings))

        # Encode
        encoded = self.encoder(embeddings)

        # MLM predictions
        mlm_logits = None
        if masked_positions is not None:
            masked_encoded = encoded[masked_positions]
            mlm_logits = self.mlm_head(masked_encoded)

        # NSP prediction (using [CLS] token)
        cls_output = encoded[:, 0, :]
        nsp_logits = self.nsp_head(cls_output)

        return mlm_logits, nsp_logits, encoded

class WholeWordMasking:
    """
    Whole word masking for BERT pretraining
    Masks entire words instead of subword tokens
    """

    def __init__(self, tokenizer, mlm_probability=0.15, span_masking=False):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.span_masking = span_masking

    def __call__(self, examples):
        """
        Apply whole word masking to batch of examples

        Args:
            examples: List of tokenized text sequences
        """
        masked_inputs = []
        labels = []

        for tokens in examples:
            # Group subword tokens by whole words
            word_groups = self._group_subwords(tokens)

            # Sample words to mask
            masked_tokens, masked_labels = self._mask_words(tokens, word_groups)

            masked_inputs.append(masked_tokens)
            labels.append(masked_labels)

        return torch.tensor(masked_inputs), torch.tensor(labels)

    def _group_subwords(self, tokens):
        """Group subword tokens into whole words"""
        word_groups = []
        current_word = []

        for i, token in enumerate(tokens):
            # Check if token starts a new word (doesn't start with ##)
            token_str = self.tokenizer.convert_ids_to_tokens([token])[0]

            if not token_str.startswith('##'):
                if current_word:
                    word_groups.append(current_word)
                current_word = [i]
            else:
                current_word.append(i)

        if current_word:
            word_groups.append(current_word)

        return word_groups

    def _mask_words(self, tokens, word_groups):
        """Mask entire words"""
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)  # -100 = ignore in loss

        # Calculate number of words to mask
        num_words_to_mask = max(1, int(len(word_groups) * self.mlm_probability))

        # Randomly select words to mask
        import random
        words_to_mask = random.sample(word_groups, num_words_to_mask)

        for word_indices in words_to_mask:
            for idx in word_indices:
                labels[idx] = tokens[idx]

                # 80%: replace with [MASK]
                # 10%: replace with random token
                # 10%: keep original
                rand = random.random()
                if rand < 0.8:
                    masked_tokens[idx] = self.tokenizer.mask_token_id
                elif rand < 0.9:
                    masked_tokens[idx] = random.randint(0, self.tokenizer.vocab_size - 1)

        return masked_tokens, labels

    def _span_masking(self, tokens, max_span_length=3):
        """Mask spans of tokens (used in SpanBERT)"""
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)

        # Geometric distribution for span lengths
        import random
        i = 0
        while i < len(tokens):
            # Sample span length
            span_length = min(
                random.choices(
                    range(1, max_span_length + 1),
                    weights=[0.5 ** (l - 1) for l in range(1, max_span_length + 1)]
                )[0],
                len(tokens) - i
            )

            # Decide whether to mask this span
            if random.random() < self.mlm_probability:
                for j in range(i, i + span_length):
                    labels[j] = tokens[j]
                    masked_tokens[j] = self.tokenizer.mask_token_id

            i += span_length

        return masked_tokens, labels
```

This provides a solid foundation for Advanced NLP. Would you like me to continue with:
1. Efficient Transformers (Longformer, BigBird)
2. Instruction Tuning & RLHF
3. RAG systems
4. Multi-task Learning
5. Advanced Text Generation

And then create the other guides (Probabilistic Deep Learning, Production ML Systems, Graph Learning, Case Studies)?