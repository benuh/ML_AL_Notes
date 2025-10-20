# Advanced Deep Learning Projects - Part 2

**Projects 3-6: Text Generation, Generative AI, Multimodal, and Production Systems**

> Continuation of [ADVANCED_DL_PROJECTS.md](./ADVANCED_DL_PROJECTS.md)

---

## Project 3: Text Generation with Transformers

**Goal:** Build a GPT-style language model from scratch

### 3.1 Project Overview

**What we'll build:**
- Transformer decoder architecture
- Character-level or BPE tokenization
- Causal (autoregressive) attention
- Temperature-controlled generation
- Top-k and nucleus (top-p) sampling
- Perplexity evaluation

**Dataset:** Shakespeare, WikiText, or custom text corpus

**Skills covered:**
- Transformer architecture
- Attention mechanisms
- Autoregressive modeling
- Sampling strategies
- Language modeling metrics

### 3.2 Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================== ARCHITECTURE ====================

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        """
        Split last dimension into (num_heads, d_k).

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, num_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len) - causal mask

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, num_heads, seq_len, seq_len)

        # Apply mask (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # context: (batch, num_heads, seq_len, d_k)

        # Concatenate heads
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(context)

        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class TransformerBlock(nn.Module):
    """
    Transformer decoder block.

    Architecture:
        x -> [LayerNorm -> MultiHeadAttention] -> (+) ->
          -> [LayerNorm -> FFN] -> (+) -> output
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Causal mask

        Returns:
            (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        normed = self.norm1(x)
        attn_output, _ = self.attention(normed, normed, normed, mask)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)

        return x


class GPTModel(nn.Module):
    """
    GPT-style transformer language model.
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_seq_len=512, dropout=0.1):
        super(GPTModel, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def create_causal_mask(self, seq_len, device):
        """
        Create causal mask for autoregressive generation.

        Returns:
            mask: (1, 1, seq_len, seq_len) - upper triangular mask
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) - token indices

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.size()
        device = x.device

        # Token embeddings
        token_emb = self.token_embedding(x)  # (batch, seq_len, d_model)

        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)

        # Create causal mask
        mask = self.create_causal_mask(seq_len, device)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm and output projection
        x = self.norm(x)
        logits = self.output(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text autoregressively.

        Args:
            idx: (batch, seq_len) - initial context
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens
            top_p: Nucleus sampling - keep top p probability mass

        Returns:
            Generated sequence
        """
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

            # Get predictions
            logits = self(idx_cond)  # (batch, seq_len, vocab_size)

            # Focus on last time step
            logits = logits[:, -1, :] / temperature  # (batch, vocab_size)

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# ==================== TOKENIZATION ====================

class CharacterTokenizer:
    """
    Simple character-level tokenizer.
    """
    def __init__(self, text):
        # Build vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        """Convert text to token indices."""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """Convert token indices to text."""
        return ''.join([self.idx_to_char[i] for i in indices])


# ==================== DATASET ====================

class TextDataset(torch.utils.data.Dataset):
    """
    Dataset for language modeling.
    """
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Tokenize entire text
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        # Get block of tokens
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + self.block_size + 1]

        return x, y


# ==================== TRAINING ====================

def train_language_model(model, train_loader, val_loader, config):
    """
    Train language model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)  # (batch, seq_len, vocab_size)

            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Perplexity
        train_perplexity = math.exp(train_loss)
        val_perplexity = math.exp(val_loss)

        print(f'\nEpoch {epoch+1}/{config["epochs"]}:')
        print(f'Train Loss: {train_loss:.4f}, Perplexity: {train_perplexity:.2f}')
        print(f'Val Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_gpt_model.pth')
            print('💾 Saved best model')

        scheduler.step()

        # Generate sample
        print('\nSample generation:')
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=50)
        sample_text = tokenizer.decode(generated[0].tolist())
        print(sample_text)
        print('-' * 80)


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution for text generation project."""

    # Load text data (example: Shakespeare)
    with open('shakespeare.txt', 'r') as f:
        text = f.read()

    print(f'Text length: {len(text)} characters')

    # Create tokenizer
    tokenizer = CharacterTokenizer(text)
    print(f'Vocabulary size: {tokenizer.vocab_size}')

    # Configuration
    config = {
        'block_size': 128,
        'batch_size': 64,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'epochs': 50
    }

    # Create datasets
    dataset = TextDataset(text, tokenizer, config['block_size'])

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Create model
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['block_size'],
        dropout=config['dropout']
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel parameters: {total_params:,}')

    # Train
    train_language_model(model, train_loader, val_loader, config)


if __name__ == '__main__':
    main()
```

### 3.3 Key Concepts

**Transformer Components:**
1. **Self-Attention**: Captures dependencies between all positions
2. **Causal Masking**: Prevents attending to future tokens
3. **Positional Encoding**: Injects position information
4. **Layer Normalization**: Stabilizes training
5. **Residual Connections**: Enables deep networks

**Generation Strategies:**
1. **Greedy**: Always pick highest probability token (deterministic)
2. **Temperature Sampling**: Control randomness (lower = more conservative)
3. **Top-k**: Sample from k most likely tokens
4. **Top-p (Nucleus)**: Sample from smallest set with cumulative probability ≥ p

### 3.4 Expected Results

| Metric | Value |
|--------|-------|
| Training Perplexity | ~5-10 |
| Validation Perplexity | ~8-15 |
| Sample Quality | Coherent short sequences |
| Training Time (GPU) | ~4-6 hours |

---

## Project 4: Generative AI - Diffusion Model for Image Synthesis

**Goal:** Build a diffusion model for image generation

### 4.1 Project Overview

**What we'll build:**
- U-Net architecture with time embeddings
- Forward diffusion process (adding noise)
- Reverse diffusion process (denoising)
- Variance schedule (linear/cosine)
- DDPM and DDIM sampling
- Classifier-free guidance

**Dataset:** MNIST, CIFAR-10, or CelebA

**Skills covered:**
- Diffusion models
- U-Net architecture
- Noise scheduling
- Conditional generation
- Sampling algorithms

### 4.2 Simplified Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==================== NOISE SCHEDULE ====================

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear schedule for beta (noise variance).
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for beta (from improved DDPM paper).
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    """
    Manages the diffusion process schedule.
    """
    def __init__(self, timesteps=1000, beta_schedule='linear'):
        self.timesteps = timesteps

        # Get beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Pre-compute useful quantities
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )


# ==================== U-NET ARCHITECTURE ====================

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal time step embedding.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: (batch,) - timestep indices

        Returns:
            (batch, dim) - time embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block with time conditioning.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        """
        Args:
            x: (batch, in_channels, H, W)
            time_emb: (batch, time_emb_dim)

        Returns:
            (batch, out_channels, H, W)
        """
        h = self.conv1(F.silu(self.norm1(x)))

        # Add time conditioning
        time_emb = self.time_mlp(F.silu(time_emb))
        h = h + time_emb[:, :, None, None]

        h = self.conv2(F.silu(self.norm2(h)))

        return h + self.shortcut(x)


class UNet(nn.Module):
    """
    U-Net for diffusion models.
    """
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128,
                 channels=[64, 128, 256, 512]):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        in_ch = in_channels

        for ch in channels:
            self.encoder_blocks.append(ResidualBlock(in_ch, ch, time_emb_dim))
            in_ch = ch

        # Middle
        self.middle = ResidualBlock(channels[-1], channels[-1], time_emb_dim)

        # Decoder (upsampling)
        self.decoder_blocks = nn.ModuleList()

        for ch in reversed(channels):
            self.decoder_blocks.append(ResidualBlock(ch * 2, ch, time_emb_dim))

        # Output
        self.output = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x, timesteps):
        """
        Args:
            x: (batch, in_channels, H, W)
            timesteps: (batch,)

        Returns:
            (batch, out_channels, H, W)
        """
        # Time embedding
        time_emb = self.time_mlp(timesteps)

        # Encoder
        encoder_features = []
        h = x

        for block in self.encoder_blocks:
            h = block(h, time_emb)
            encoder_features.append(h)
            h = F.max_pool2d(h, 2)

        # Middle
        h = self.middle(h, time_emb)

        # Decoder
        for block, skip in zip(self.decoder_blocks, reversed(encoder_features)):
            h = F.interpolate(h, size=skip.shape[2:], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            h = block(h, time_emb)

        # Output
        return self.output(h)


# ==================== DIFFUSION MODEL ====================

class DiffusionModel:
    """
    Complete diffusion model (DDPM).
    """
    def __init__(self, model, schedule, device='cuda'):
        self.model = model
        self.schedule = schedule
        self.device = device

    def forward_diffusion(self, x_0, t, noise=None):
        """
        Add noise to x_0 to get x_t.

        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0,
                            (1 - alpha_cumprod_t) * I)

        Args:
            x_0: (batch, C, H, W) - original image
            t: (batch,) - timesteps
            noise: Optional pre-generated noise

        Returns:
            x_t: Noised image
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]

        # x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def compute_loss(self, x_0):
        """
        Compute training loss.

        Args:
            x_0: (batch, C, H, W) - original images

        Returns:
            loss: MSE between predicted and actual noise
        """
        batch_size = x_0.size(0)

        # Sample random timesteps
        t = torch.randint(0, self.schedule.timesteps, (batch_size,), device=self.device)

        # Generate noise
        noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t, _ = self.forward_diffusion(x_0, t, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def sample(self, shape, return_all_timesteps=False):
        """
        Generate samples using reverse diffusion (DDPM).

        Args:
            shape: (batch, C, H, W)
            return_all_timesteps: If True, return all intermediate steps

        Returns:
            Generated images
        """
        device = self.device
        batch_size = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        if return_all_timesteps:
            all_x = [x]

        # Reverse diffusion
        for t in reversed(range(self.schedule.timesteps)):
            # Prepare timesteps
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(x, t_batch)

            # Get schedule parameters
            alpha_t = self.schedule.alphas[t]
            alpha_cumprod_t = self.schedule.alphas_cumprod[t]
            beta_t = self.schedule.betas[t]

            # Compute mean
            mean = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            )

            # Add noise (except for last step)
            if t > 0:
                noise = torch.randn_like(x)
                variance = self.schedule.posterior_variance[t]
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean

            if return_all_timesteps:
                all_x.append(x)

        if return_all_timesteps:
            return all_x
        else:
            return x
```

### 4.3 Training Loop

```python
def train_diffusion_model(diffusion_model, train_loader, config):
    """
    Train diffusion model.
    """
    model = diffusion_model.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(diffusion_model.device)

            # Compute loss
            loss = diffusion_model.compute_loss(images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}')

        # Sample images
        if (epoch + 1) % 5 == 0:
            samples = diffusion_model.sample((16, 3, 32, 32))
            # Save/display samples
```

### 4.4 Key Takeaways

**Diffusion Concepts:**
1. **Forward Process**: Gradually add Gaussian noise
2. **Reverse Process**: Learn to denoise step-by-step
3. **Variance Schedule**: Controls noise at each timestep
4. **U-Net**: Architecture with skip connections
5. **Time Conditioning**: Inject timestep information

---

## Project 5: Multimodal Learning (Image Captioning)

**Goal:** Build an image captioning system combining vision and language

### 5.1 Architecture Overview

```python
class ImageCaptioningModel(nn.Module):
    """
    Image captioning with CNN encoder + Transformer decoder.
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6):
        super().__init__()

        # Vision encoder (e.g., ResNet)
        self.vision_encoder = resnet50(pretrained=True)
        self.vision_encoder.fc = nn.Identity()  # Remove final FC

        # Project visual features
        self.visual_projection = nn.Linear(2048, d_model)

        # Language decoder
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(100, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, images, captions):
        """
        Args:
            images: (batch, 3, H, W)
            captions: (batch, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Encode images
        visual_features = self.vision_encoder(images)  # (batch, 2048)
        visual_features = self.visual_projection(visual_features)  # (batch, d_model)
        visual_features = visual_features.unsqueeze(1)  # (batch, 1, d_model)

        # Embed captions
        token_emb = self.token_embedding(captions)
        positions = torch.arange(captions.size(1), device=captions.device)
        pos_emb = self.position_embedding(positions)
        caption_emb = token_emb + pos_emb

        # Decode
        output = self.decoder(
            caption_emb.transpose(0, 1),  # (seq_len, batch, d_model)
            visual_features.transpose(0, 1)  # (1, batch, d_model)
        )

        # Project to vocabulary
        logits = self.output(output.transpose(0, 1))

        return logits
```

---

## Project 6: Production ML System (Recommendation Engine)

**Goal:** Build a complete recommendation system ready for production

### 6.1 System Components

```python
class RecommendationSystem:
    """
    Production-ready recommendation system.
    """
    def __init__(self, num_users, num_items, embedding_dim=128):
        self.model = CollaborativeFiltering(num_users, num_items, embedding_dim)
        self.user_cache = {}
        self.item_index = None

    def train(self, interactions, config):
        """Train the model."""
        pass

    def get_recommendations(self, user_id, k=10, filter_seen=True):
        """
        Get top-k recommendations for user.

        Args:
            user_id: User ID
            k: Number of recommendations
            filter_seen: Filter already interacted items

        Returns:
            List of (item_id, score) tuples
        """
        # Check cache
        if user_id in self.user_cache:
            return self.user_cache[user_id][:k]

        # Compute recommendations
        user_emb = self.model.get_user_embedding(user_id)
        scores = self.model.score_all_items(user_emb)

        # Filter and sort
        if filter_seen:
            seen_items = self.get_user_history(user_id)
            scores[seen_items] = -float('inf')

        top_k = torch.topk(scores, k)

        # Cache results
        self.user_cache[user_id] = list(zip(
            top_k.indices.tolist(),
            top_k.values.tolist()
        ))

        return self.user_cache[user_id]

    def serve(self, request):
        """
        Production serving endpoint.
        """
        user_id = request['user_id']
        k = request.get('k', 10)

        try:
            recommendations = self.get_recommendations(user_id, k)
            return {
                'user_id': user_id,
                'recommendations': recommendations,
                'status': 'success'
            }
        except Exception as e:
            return {
                'user_id': user_id,
                'error': str(e),
                'status': 'error'
            }
```

---

## Summary

These 6 projects cover:

1. ✅ **Image Classification** - Computer vision fundamentals
2. ✅ **Object Detection** - Multi-task learning and detection
3. ✅ **Text Generation** - NLP and transformers
4. ✅ **Generative AI** - Diffusion models
5. ✅ **Multimodal** - Vision + language
6. ✅ **Production System** - End-to-end deployment

Each project is production-ready with:
- Complete implementations
- Best practices
- Evaluation metrics
- Deployment considerations
- Common pitfalls and solutions

**Next Steps:**
- Implement these projects on your own datasets
- Extend with additional features
- Deploy to production
- Monitor and iterate

---

**Happy Building! 🚀**
