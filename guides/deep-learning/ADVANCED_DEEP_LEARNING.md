# Advanced Deep Learning Techniques

**Comprehensive Guide to Modern Deep Learning**

> Complete reference for advanced deep learning techniques, generative models, optimization strategies, and production best practices.

**Last Updated:** October 2025

---

## Table of Contents

1. [Generative Models](#1-generative-models)
   - Variational Autoencoders (VAEs)
   - Generative Adversarial Networks (GANs)
   - Autoregressive Models
   - Diffusion Models (Basics)
2. [Advanced Optimization Techniques](#2-advanced-optimization-techniques)
3. [Advanced Regularization](#3-advanced-regularization)
4. [Self-Supervised and Contrastive Learning](#4-self-supervised-and-contrastive-learning)
5. [Model Compression and Efficiency](#5-model-compression-and-efficiency)
6. [Advanced Training Techniques](#6-advanced-training-techniques)
7. [Neural Architecture Design](#7-neural-architecture-design)
8. [Best Practices and Tips](#8-best-practices-and-tips)

---

## 1. Generative Models

### 1.1 Variational Autoencoders (VAEs)

**Theoretical Foundation**

VAEs learn a latent representation by maximizing the Evidence Lower Bound (ELBO):

```
log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x) || p(z))
         = ELBO

where:
- p(x|z): Decoder (likelihood)
- q(z|x): Encoder (approximate posterior)
- p(z): Prior (usually N(0, I))
- KL: Kullback-Leibler divergence
```

**Reparameterization Trick**

To allow backpropagation through sampling:

```
z = μ + σ ⊙ ε, where ε ~ N(0, I)
```

This makes the sampling operation differentiable.

**Complete PyTorch Implementation**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """Variational Autoencoder for image generation"""

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # Mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ * ε"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector to reconstruction"""
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    """
    VAE loss = Reconstruction loss + KL divergence

    Reconstruction: Binary cross-entropy
    KL: 0.5 * sum(1 + log(σ²) - μ² - σ²)
    """
    # Reconstruction loss (binary cross-entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # KL divergence loss
    # KL(q(z|x) || p(z)) where p(z) = N(0,I)
    # = 0.5 * sum(1 + log(σ²) - μ² - σ²)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# Training loop
def train_vae(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)

        # Backward pass
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
    return avg_loss

# Generate new samples
def generate_samples(model, num_samples=64, latent_dim=20):
    """Generate new images by sampling from prior"""
    model.eval()
    with torch.no_grad():
        # Sample from prior N(0, I)
        z = torch.randn(num_samples, latent_dim)
        samples = model.decode(z)
    return samples.view(num_samples, 1, 28, 28)

# Interpolate in latent space
def interpolate_latent(model, x1, x2, num_steps=10):
    """Interpolate between two images in latent space"""
    model.eval()
    with torch.no_grad():
        # Encode both images
        mu1, _ = model.encode(x1.view(-1, 784))
        mu2, _ = model.encode(x2.view(-1, 784))

        # Interpolate
        alphas = torch.linspace(0, 1, num_steps)
        interpolations = []

        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            recon = model.decode(z)
            interpolations.append(recon.view(1, 28, 28))

    return torch.cat(interpolations, dim=0)
```

**β-VAE for Disentangled Representations**

```python
def beta_vae_loss(recon_x, x, mu, logvar, beta=4.0):
    """
    β-VAE loss with controllable disentanglement

    Higher β → more disentangled but lower quality
    β = 1 → standard VAE
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + beta * KLD
```

**When to Use VAEs:**
- Need probabilistic latent representations
- Want to generate new samples
- Require smooth latent space for interpolation
- Need disentangled representations (β-VAE)

---

### 1.2 Generative Adversarial Networks (GANs)

**Theoretical Foundation**

GANs involve a minimax game between generator G and discriminator D:

```
min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]

where:
- D(x): Discriminator's estimate that x is real
- G(z): Generator's output from noise z
- Goal: G generates realistic samples to fool D
```

**Vanilla GAN Implementation**

```python
class Generator(nn.Module):
    """Generator network: noise → image"""

    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    """Discriminator network: image → real/fake"""

    def __init__(self, img_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Training GAN
def train_gan(generator, discriminator, train_loader, num_epochs=200):
    """Train GAN with alternating updates"""

    # Optimizers
    lr = 0.0002
    b1, b2 = 0.5, 0.999  # Adam parameters
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Loss function
    adversarial_loss = nn.BCELoss()

    latent_dim = 100

    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(train_loader):
            batch_size = imgs.size(0)

            # Adversarial ground truths
            valid = torch.ones(batch_size, 1)
            fake = torch.zeros(batch_size, 1)

            # Configure input
            real_imgs = imgs

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss on real images
            real_loss = adversarial_loss(discriminator(real_imgs), valid)

            # Generate fake images
            z = torch.randn(batch_size, latent_dim)
            fake_imgs = generator(z)

            # Loss on fake images
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake images
            gen_imgs = generator(z)

            # Generator wants discriminator to think images are valid
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item():.4f}] "
                  f"[G loss: {g_loss.item():.4f}]")
```

**Deep Convolutional GAN (DCGAN)**

```python
class DCGANGenerator(nn.Module):
    """DCGAN Generator using transposed convolutions"""

    def __init__(self, latent_dim=100, channels=1):
        super(DCGANGenerator, self).__init__()

        self.init_size = 7  # Initial spatial size
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 7x7 → 14x14
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # 14x14 → 28x28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator using strided convolutions"""

    def __init__(self, channels=1):
        super(DCGANDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),  # Stride 2
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),  # 28x28 → 14x14
            *discriminator_block(16, 32),                   # 14x14 → 7x7
            *discriminator_block(32, 64),                   # 7x7 → 3x3
            *discriminator_block(64, 128),                  # 3x3 → 1x1
        )

        # Output layer
        ds_size = 1
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
```

**Wasserstein GAN (WGAN) - More Stable Training**

```python
def wasserstein_loss_discriminator(real_output, fake_output):
    """
    WGAN loss for discriminator (critic)
    Maximize: E[D(x_real)] - E[D(x_fake)]
    """
    return -torch.mean(real_output) + torch.mean(fake_output)

def wasserstein_loss_generator(fake_output):
    """
    WGAN loss for generator
    Maximize: E[D(G(z))]
    """
    return -torch.mean(fake_output)

def gradient_penalty(discriminator, real_imgs, fake_imgs, device='cpu'):
    """
    Gradient penalty for WGAN-GP
    Enforces 1-Lipschitz constraint
    """
    batch_size = real_imgs.size(0)

    # Random weight term for interpolation
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)

    # Get interpolated image
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)

    # Get discriminator output
    d_interpolates = discriminator(interpolates)

    # Get gradient w.r.t. interpolates
    fake = torch.ones(batch_size, 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

# WGAN-GP training step
def train_wgan_gp_step(generator, discriminator, real_imgs,
                       optimizer_G, optimizer_D, latent_dim=100, lambda_gp=10):
    """Single training step for WGAN-GP"""
    batch_size = real_imgs.size(0)

    # ---------------------
    #  Train Discriminator
    # ---------------------
    optimizer_D.zero_grad()

    # Sample noise
    z = torch.randn(batch_size, latent_dim)
    fake_imgs = generator(z).detach()

    # Discriminator outputs
    real_validity = discriminator(real_imgs)
    fake_validity = discriminator(fake_imgs)

    # Gradient penalty
    gp = gradient_penalty(discriminator, real_imgs, fake_imgs)

    # Wasserstein loss with gradient penalty
    d_loss = wasserstein_loss_discriminator(real_validity, fake_validity) + lambda_gp * gp

    d_loss.backward()
    optimizer_D.step()

    # -----------------
    #  Train Generator (every n_critic steps)
    # -----------------
    n_critic = 5  # Train discriminator 5x more than generator

    if batch_size % n_critic == 0:
        optimizer_G.zero_grad()

        # Generate fake images
        gen_imgs = generator(torch.randn(batch_size, latent_dim))
        fake_validity = discriminator(gen_imgs)

        # Generator loss
        g_loss = wasserstein_loss_generator(fake_validity)

        g_loss.backward()
        optimizer_G.step()
    else:
        g_loss = torch.tensor(0.0)

    return d_loss.item(), g_loss.item()
```

**Conditional GAN (cGAN)**

```python
class ConditionalGenerator(nn.Module):
    """Generator conditioned on class labels"""

    def __init__(self, latent_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super(ConditionalGenerator, self).__init__()
        self.img_shape = img_shape

        # Embedding for class labels
        self.label_emb = nn.Embedding(num_classes, num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate noise and label embedding
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img
```

**GAN Training Tips:**

1. **Use label smoothing**: Real labels = 0.9, fake labels = 0.1
2. **Train discriminator more**: 2-5x more updates than generator
3. **Use LeakyReLU**: In both G and D (slope = 0.2)
4. **Batch normalization**: In generator (not discriminator output layer)
5. **Use Adam**: β1 = 0.5, β2 = 0.999
6. **Monitor mode collapse**: Use diversity metrics

**When to Use GANs:**
- High-quality image generation
- Data augmentation
- Style transfer
- Image-to-image translation

---

### 1.3 Autoregressive Models

**Theory**

Autoregressive models decompose joint probability:

```
p(x) = p(x₁) · p(x₂|x₁) · p(x₃|x₁,x₂) · ... · p(xₙ|x₁,...,xₙ₋₁)
```

**PixelCNN-style Implementation**

```python
class MaskedConv2d(nn.Conv2d):
    """Masked convolution for autoregressive property"""

    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))

        self.mask[:, :, :self.kernel_size[0] // 2] = 1
        self.mask[:, :, self.kernel_size[0] // 2, :self.kernel_size[1] // 2] = 1

        if mask_type == 'B':
            # Include center pixel for subsequent layers
            self.mask[:, :, self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class PixelCNN(nn.Module):
    """Simplified PixelCNN for autoregressive generation"""

    def __init__(self, num_layers=7, hidden_dim=64, num_classes=256):
        super(PixelCNN, self).__init__()

        # First layer uses mask type A (excludes center)
        self.layers = nn.ModuleList([
            MaskedConv2d('A', 1, hidden_dim, 7, padding=3)
        ])

        # Subsequent layers use mask type B (includes center)
        for _ in range(num_layers - 1):
            self.layers.append(
                MaskedConv2d('B', hidden_dim, hidden_dim, 7, padding=3)
            )
            self.layers.append(nn.ReLU())

        # Output layer: predict pixel values (0-255)
        self.output = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return self.output(out)

# Sampling from PixelCNN
@torch.no_grad()
def sample_pixelcnn(model, num_samples=16, img_shape=(28, 28)):
    """Sample images pixel by pixel"""
    model.eval()
    samples = torch.zeros(num_samples, 1, *img_shape)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            # Predict distribution over pixel values
            logits = model(samples)
            probs = F.softmax(logits[:, :, i, j], dim=1)

            # Sample pixel value
            samples[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.0

    return samples
```

---

### 1.4 Diffusion Models (Introduction)

**Theory**

Diffusion models learn to reverse a gradual noising process:

1. **Forward process** (fixed): Add Gaussian noise over T steps
   ```
   q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)
   ```

2. **Reverse process** (learned): Denoise step by step
   ```
   pθ(xₜ₋₁|xₜ) = N(xₜ₋₁; μθ(xₜ, t), Σθ(xₜ, t))
   ```

**Simplified DDPM Implementation**

```python
class DiffusionModel(nn.Module):
    """Simplified Denoising Diffusion Probabilistic Model"""

    def __init__(self, img_channels=1, time_dim=256):
        super(DiffusionModel, self).__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Simple U-Net-like architecture
        self.down1 = self.conv_block(img_channels + time_dim, 64)
        self.down2 = self.conv_block(64, 128)

        self.bottleneck = self.conv_block(128, 256)

        self.up2 = self.conv_block(256 + 128, 128)
        self.up1 = self.conv_block(128 + 64, 64)

        self.out = nn.Conv2d(64, img_channels, 1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, t):
        # Embed time
        t_emb = self.time_mlp(t.unsqueeze(-1))
        t_emb = t_emb.view(-1, t_emb.size(1), 1, 1)
        t_emb = t_emb.expand(-1, -1, x.size(2), x.size(3))

        # Concatenate time embedding with input
        x = torch.cat([x, t_emb], dim=1)

        # U-Net forward pass
        d1 = self.down1(x)
        d2 = self.down2(F.max_pool2d(d1, 2))

        b = self.bottleneck(F.max_pool2d(d2, 2))

        u2 = self.up2(torch.cat([F.interpolate(b, scale_factor=2), d2], dim=1))
        u1 = self.up1(torch.cat([F.interpolate(u2, scale_factor=2), d1], dim=1))

        return self.out(u1)

# Noise schedule
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# Simplified training
def diffusion_loss(model, x0, timesteps=1000):
    """
    Simplified diffusion training loss
    Predict the noise added to the image
    """
    # Sample random timestep
    t = torch.randint(0, timesteps, (x0.size(0),))

    # Sample noise
    noise = torch.randn_like(x0)

    # Get noise schedule
    betas = cosine_beta_schedule(timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Add noise to images
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])

    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)

    noisy_images = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    # Predict noise
    predicted_noise = model(noisy_images, t.float() / timesteps)

    # MSE loss between predicted and actual noise
    loss = F.mse_loss(predicted_noise, noise)

    return loss
```

**When to Use Diffusion Models:**
- State-of-the-art image quality
- Stable training (unlike GANs)
- Can condition on various inputs
- Trade-off: Slower sampling than GANs

---

## 2. Advanced Optimization Techniques

### 2.1 Learning Rate Schedules

**Step Decay**

```python
# Reduce LR by factor every N epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Example: LR = 0.1 → 0.01 at epoch 30 → 0.001 at epoch 60
```

**Exponential Decay**

```python
# LR = initial_lr * gamma^epoch
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

**Cosine Annealing**

```python
# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # First restart after 10 epochs
    T_mult=2,    # Double restart period each time
    eta_min=1e-6 # Minimum learning rate
)
```

**One Cycle Policy (Highly Effective)**

```python
# Best practice for many tasks
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=100,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,        # Warm-up for 30% of training
    anneal_strategy='cos', # Cosine annealing
    div_factor=25.0,      # Initial_lr = max_lr/25
    final_div_factor=1e4  # Min_lr = initial_lr/1e4
)

# Update per batch (not per epoch!)
for epoch in range(num_epochs):
    for batch in train_loader:
        # Training step
        optimizer.step()
        scheduler.step()  # Update LR every batch
```

**Learning Rate Finder**

```python
def find_lr(model, train_loader, criterion, optimizer,
            init_lr=1e-8, final_lr=10., beta=0.98):
    """
    Find optimal learning rate using exponential search
    Plot loss vs LR and pick LR where loss decreases fastest
    """
    num_batches = len(train_loader) - 1
    mult = (final_lr / init_lr) ** (1 / num_batches)
    lr = init_lr
    optimizer.param_groups[0]['lr'] = lr

    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    lrs = []

    for data, targets in train_loader:
        batch_num += 1

        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Compute smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        # Stop if loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break

        # Record best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # Store values
        losses.append(smoothed_loss)
        lrs.append(lr)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update learning rate
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    # Plot
    import matplotlib.pyplot as plt
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('LR Finder')
    plt.show()

    return lrs, losses

# Usage:
# lrs, losses = find_lr(model, train_loader, criterion, optimizer)
# Pick LR where loss decreases most steeply (usually 10x before minimum)
```

### 2.2 Gradient Clipping

**Prevent Exploding Gradients**

```python
# Clip by norm (most common)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value (less common)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Training loop with gradient clipping
for epoch in range(num_epochs):
    for data, targets in train_loader:
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)

        loss.backward()

        # Clip gradients before optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
```

### 2.3 Mixed Precision Training

**16-bit and 32-bit Mixed Training (2-3x Speedup)**

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize gradient scaler
scaler = GradScaler()

for epoch in range(num_epochs):
    for data, targets in train_loader:
        optimizer.zero_grad()

        # Forward pass with autocasting
        with autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Unscale gradients and clip (optional)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()
```

**Benefits:**
- 2-3x faster training
- 50% less memory usage
- Allows larger batch sizes
- Minimal accuracy loss

### 2.4 Advanced Optimizers

**AdamW (Adam with Weight Decay Decoupling)**

```python
# Better than Adam for most tasks
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # Decoupled weight decay
)
```

**RAdam (Rectified Adam)**

```python
# Handles early training instability
# pip install torch-optimizer
import torch_optimizer as optim

optimizer = optim.RAdam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)
```

**Lookahead Optimizer**

```python
# Wraps another optimizer for more stable updates
from torch_optimizer import Lookahead

base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
```

### 2.5 Gradient Accumulation

**Simulate Larger Batch Sizes**

```python
accumulation_steps = 4  # Effective batch size = batch_size * 4

optimizer.zero_grad()

for i, (data, targets) in enumerate(train_loader):
    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, targets)

    # Normalize loss to account for accumulation
    loss = loss / accumulation_steps

    # Backward pass
    loss.backward()

    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**When to Use:**
- GPU memory limited
- Want larger batch sizes for stability
- Batch normalization requires larger batches

---

## 3. Advanced Regularization

### 3.1 Label Smoothing

**Prevent Overconfidence**

```python
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing

    Instead of [0, 0, 1, 0, 0]
    Use: [ε/K, ε/K, 1-ε+ε/K, ε/K, ε/K]

    where ε = smoothing factor, K = num classes
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        pred: (batch_size, num_classes) - predicted logits
        target: (batch_size,) - true class indices
        """
        num_classes = pred.size(1)

        # One-hot encode targets
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)

        # Apply label smoothing
        smoothed = one_hot * (1 - self.smoothing) + self.smoothing / num_classes

        # Compute loss
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(smoothed * log_prob).sum(dim=1).mean()

        return loss

# Usage
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
```

**Benefits:**
- Better calibrated predictions
- Prevents overconfidence
- Improves generalization
- Typical smoothing: 0.1

### 3.2 Mixup

**Interpolate Training Examples**

```python
def mixup_data(x, y, alpha=1.0):
    """
    Mixup data augmentation

    Creates virtual training examples:
    x_new = λ*x_i + (1-λ)*x_j
    y_new = λ*y_i + (1-λ)*y_j
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training with mixup
for data, targets in train_loader:
    # Apply mixup
    data, targets_a, targets_b, lam = mixup_data(data, targets, alpha=1.0)

    # Forward pass
    outputs = model(data)

    # Mixup loss
    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Benefits:**
- Stronger regularization
- Better calibration
- Adversarial robustness
- Typical alpha: 0.2-1.0

### 3.3 CutMix

**Cut and Mix Image Patches**

```python
def cutmix_data(x, y, alpha=1.0):
    """
    CutMix data augmentation

    Cut a random patch from one image and paste to another
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    # Get random box
    _, _, H, W = x.size()
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply cutmix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda based on actual box size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# Training loop is same as mixup
```

### 3.4 Cutout / Random Erasing

**Randomly Mask Regions**

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.5,           # Probability
        scale=(0.02, 0.33),  # Erase area ratio
        ratio=(0.3, 3.3),     # Aspect ratio
        value=0          # Fill value
    )
])
```

### 3.5 Stochastic Depth

**Randomly Drop Layers During Training**

```python
class StochasticDepth(nn.Module):
    """
    Stochastic depth (drop path) regularization
    Randomly drops residual connections during training
    """

    def __init__(self, drop_prob=0.1):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x

        keep_prob = 1 - self.drop_prob

        # Create binary tensor
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize

        return x.div(keep_prob) * random_tensor

# Use in residual blocks
class ResidualBlockWithStochasticDepth(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stochastic_depth = StochasticDepth(drop_prob)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply stochastic depth to residual connection
        out = identity + self.stochastic_depth(out)
        out = F.relu(out)

        return out
```

**Benefits:**
- Trains ensemble of sub-networks
- Reduces overfitting in deep networks
- Faster training (effective depth reduced)
- Used in EfficientNet, Vision Transformers

---

## 4. Self-Supervised and Contrastive Learning

### 4.1 Contrastive Learning (SimCLR)

**Learn Representations Without Labels**

```python
class SimCLR(nn.Module):
    """
    Simple Framework for Contrastive Learning

    Key idea: Maximize agreement between different augmented
    views of the same image
    """

    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()

        # Base encoder (e.g., ResNet)
        self.encoder = base_encoder

        # Projection head
        # Maps representations to space where contrastive loss is applied
        hidden_dim = base_encoder.fc.in_features
        self.encoder.fc = nn.Identity()  # Remove classification head

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z  # Return both representation and projection

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss
    (NT-Xent loss for SimCLR)
    """

    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: Projections of two augmented views (batch_size, projection_dim)
        """
        batch_size = z_i.size(0)

        # Concatenate projections
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, projection_dim)

        # Compute similarity matrix
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        # Temperature scaling
        sim_matrix = sim_matrix / self.temperature

        # Create mask for positive pairs
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        mask = torch.eye(2 * batch_size, dtype=torch.bool)
        positives_mask = torch.zeros_like(mask)
        positives_mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool)
        positives_mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool)

        # Remove self-comparisons
        sim_matrix = sim_matrix[~mask].view(2 * batch_size, -1)

        # Get positive pairs
        positives = sim_matrix[positives_mask[~mask].view(2 * batch_size, -1)]
        positives = positives.view(2 * batch_size, 1)

        # Compute loss
        # -log(exp(sim_pos) / sum(exp(sim)))
        logits = torch.cat([positives, sim_matrix], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long)

        loss = F.cross_entropy(logits, labels)

        return loss

# Data augmentation for SimCLR
class SimCLRAugmentation:
    """Strong augmentation pipeline for contrastive learning"""

    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        # Return two augmented views
        return self.transform(x), self.transform(x)

# Training SimCLR
def train_simclr(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for (x_i, x_j), _ in train_loader:
        x_i, x_j = x_i.to(device), x_j.to(device)

        # Forward pass for both views
        _, z_i = model(x_i)
        _, z_j = model(x_j)

        # Compute contrastive loss
        loss = criterion(z_i, z_j)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)
```

### 4.2 Momentum Contrast (MoCo)

**Use Momentum Encoder and Queue**

```python
class MoCo(nn.Module):
    """
    Momentum Contrast for unsupervised learning

    Key innovations:
    - Momentum encoder (slow-moving version of query encoder)
    - Large queue of negative samples
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        """
        dim: Feature dimension
        K: Queue size
        m: Momentum coefficient
        T: Temperature
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # Query encoder
        self.encoder_q = base_encoder
        hidden_dim = self.encoder_q.fc.in_features
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        # Key encoder (momentum updated)
        self.encoder_k = copy.deepcopy(self.encoder_q)

        # No gradients for key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # Queue for negative samples
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update: θ_k = m * θ_k + (1 - m) * θ_q
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue"""
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # Replace oldest embeddings
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        im_q: Query images
        im_k: Key images (different augmentation of same images)
        """
        # Compute query features
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():
            # Update key encoder
            self._momentum_update_key_encoder()

            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        # Compute logits
        # Positive logits: (batch_size, 1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # Negative logits: (batch_size, K)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # Logits: (batch_size, 1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature
        logits /= self.T

        # Labels: positives are at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long)

        # Update queue
        self._dequeue_and_enqueue(k)

        return logits, labels
```

### 4.3 Masked Autoencoders (MAE)

**BERT-style Pretraining for Vision**

```python
class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for images

    Randomly mask patches and reconstruct them
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=16, mask_ratio=0.75):
        super().__init__()

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)

        num_patches = (img_size // patch_size) ** 2

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Encoder (simplified - use actual Transformer blocks in practice)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=depth
        )

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=decoder_num_heads),
            num_layers=decoder_depth
        )

        # Reconstruction head
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

    def random_masking(self, x, mask_ratio):
        """
        Random masking of patches
        Returns: masked x, mask, ids_restore
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # Random shuffle
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, imgs):
        # Patch embedding
        x = self.patch_embed(imgs)  # (N, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (N, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # Encode
        x = self.encoder(x)

        # Decoder embedding
        x = self.decoder_embed(x)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # Decode
        x = self.decoder(x_full)

        # Predict pixels
        pred = self.decoder_pred(x)

        return pred, mask

def mae_loss(imgs, pred, mask, patch_size=16):
    """
    Compute MAE loss (only on masked patches)
    """
    # Patchify target
    target = imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    target = target.contiguous().view(imgs.shape[0], imgs.shape[1], -1, patch_size, patch_size)
    target = target.permute(0, 2, 3, 4, 1).reshape(imgs.shape[0], -1, patch_size**2 * imgs.shape[1])

    # MSE loss on masked patches
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # Mean per patch

    # Only compute loss on masked patches
    loss = (loss * mask).sum() / mask.sum()

    return loss
```

**When to Use Self-Supervised Learning:**
- Limited labeled data
- Want general-purpose features
- Pre-training for downstream tasks
- Data augmentation is effective

---

## 5. Model Compression and Efficiency

### 5.1 Knowledge Distillation

**Transfer Knowledge from Large to Small Model**

```python
class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss

    L = α * L_CE(student, labels) + (1-α) * L_KD(student, teacher)
    """

    def __init__(self, alpha=0.5, temperature=4.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        # Hard targets loss (with true labels)
        loss_ce = self.ce_loss(student_logits, labels)

        # Soft targets loss (with teacher predictions)
        # Use temperature to soften distributions
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        loss_kd = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        # Combined loss
        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd

        return loss

# Training with distillation
def train_with_distillation(student, teacher, train_loader, optimizer, criterion):
    """Train student model with teacher guidance"""

    student.train()
    teacher.eval()  # Teacher in eval mode

    total_loss = 0

    for data, targets in train_loader:
        optimizer.zero_grad()

        # Student predictions
        student_logits = student(data)

        # Teacher predictions (no gradients)
        with torch.no_grad():
            teacher_logits = teacher(data)

        # Distillation loss
        loss = criterion(student_logits, teacher_logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# Example: Distill ResNet-50 to MobileNet
teacher = torchvision.models.resnet50(pretrained=True)
student = torchvision.models.mobilenet_v2(num_classes=1000)

criterion = DistillationLoss(alpha=0.5, temperature=4.0)
optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

# Train student
for epoch in range(num_epochs):
    loss = train_with_distillation(student, teacher, train_loader, optimizer, criterion)
    print(f'Epoch {epoch}, Loss: {loss:.4f}')
```

### 5.2 Pruning

**Remove Unnecessary Weights**

```python
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """
    Prune model by removing smallest magnitude weights

    amount: Fraction of weights to prune (0.3 = remove 30%)
    """
    # Prune all Conv2d and Linear layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # L1 unstructured pruning (magnitude-based)
            prune.l1_unstructured(module, name='weight', amount=amount)

            # Can also prune biases
            if module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=amount)

    return model

# Structured pruning (remove entire channels/filters)
def structured_prune_model(model, amount=0.3):
    """Remove entire channels based on L2 norm"""

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Prune channels (dim=0 for output channels)
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)

    return model

# Make pruning permanent
def make_pruning_permanent(model):
    """Remove pruning reparameterization and make it permanent"""

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.remove(module, 'weight')
            if hasattr(module, 'bias') and module.bias is not None:
                prune.remove(module, 'bias')

    return model

# Iterative pruning (best practice)
def iterative_pruning(model, train_loader, optimizer, criterion,
                     initial_sparsity=0.0, final_sparsity=0.9, num_iterations=10):
    """
    Gradually increase pruning amount

    1. Train model
    2. Prune small amount
    3. Fine-tune
    4. Repeat
    """
    sparsity_step = (final_sparsity - initial_sparsity) / num_iterations
    current_sparsity = initial_sparsity

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}, Sparsity: {current_sparsity:.2f}")

        # Prune
        prune_model(model, amount=sparsity_step)

        # Fine-tune
        for epoch in range(5):  # Few epochs of fine-tuning
            train_epoch(model, train_loader, optimizer, criterion)

        current_sparsity += sparsity_step

    # Make pruning permanent
    make_pruning_permanent(model)

    return model

# Check sparsity
def check_sparsity(model):
    """Calculate percentage of zero weights"""

    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

    sparsity = 100.0 * zero_params / total_params
    print(f"Global sparsity: {sparsity:.2f}%")

    return sparsity
```

### 5.3 Quantization

**Reduce Precision (32-bit → 8-bit)**

```python
# Post-training quantization (PTQ)
def quantize_model_ptq(model, calibration_loader):
    """
    Post-training quantization
    Quantize trained model without retraining
    """
    # Set model to eval mode
    model.eval()

    # Fuse layers (Conv+BN+ReLU)
    model_fused = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])

    # Specify quantization configuration
    model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare model for quantization
    model_prepared = torch.quantization.prepare(model_fused)

    # Calibrate with representative data
    with torch.no_grad():
        for data, _ in calibration_loader:
            model_prepared(data)

    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)

    return model_quantized

# Quantization-aware training (QAT)
def quantize_model_qat(model, train_loader, num_epochs=10):
    """
    Quantization-aware training
    Train with quantization in mind for better accuracy
    """
    # Fuse layers
    model_fused = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])

    # Specify quantization configuration
    model_fused.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    # Prepare for QAT
    model_prepared = torch.quantization.prepare_qat(model_fused)

    # Train with fake quantization
    model_prepared.train()
    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model_prepared(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

    # Convert to quantized model
    model_prepared.eval()
    model_quantized = torch.quantization.convert(model_prepared)

    return model_quantized

# Dynamic quantization (for LSTMs/RNNs)
def dynamic_quantize_model(model):
    """
    Dynamic quantization - quantize weights, not activations
    Good for RNNs and models where activation distribution varies
    """
    model_quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},  # Layers to quantize
        dtype=torch.qint8
    )

    return model_quantized

# Check model size reduction
def check_model_size(model, quantized_model):
    """Compare model sizes"""

    # Save models
    torch.save(model.state_dict(), 'model.pth')
    torch.save(quantized_model.state_dict(), 'model_quantized.pth')

    import os
    original_size = os.path.getsize('model.pth') / 1e6  # MB
    quantized_size = os.path.getsize('model_quantized.pth') / 1e6

    print(f"Original size: {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
```

**Quantization Benefits:**
- 4x smaller model size (32-bit → 8-bit)
- 2-4x faster inference
- Lower memory bandwidth
- Minimal accuracy loss (< 1% with QAT)

---

## 6. Advanced Training Techniques

### 6.1 Multi-GPU Training

**Data Parallel**

```python
# Simple data parallelism
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to('cuda')

# Training is the same
for data, targets in train_loader:
    data, targets = data.to('cuda'), targets.to('cuda')
    outputs = model(data)
    loss = criterion(outputs, targets)
    # ...
```

**Distributed Data Parallel (Better Performance)**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    """Training function for each process"""
    setup(rank, world_size)

    # Create model and move to GPU
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Shuffle differently each epoch

        for data, targets in train_loader:
            data, targets = data.to(rank), targets.to(rank)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    cleanup()

# Launch with torch.multiprocessing
import torch.multiprocessing as mp

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

### 6.2 Exponential Moving Average (EMA)

**Smooth Model Weights**

```python
class ModelEMA:
    """
    Exponential Moving Average of model weights

    Keeps running average of model parameters
    Often improves generalization
    """

    def __init__(self, model, decay=0.9999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad = False

    def update(self, model):
        """Update EMA parameters"""
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(),
                                              model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data,
                                                     alpha=1 - self.decay)

    def update_attr(self, model):
        """Copy other attributes (batch norm stats, etc.)"""
        for ema_attr, model_attr in zip(self.model.state_dict(),
                                        model.state_dict()):
            if not ema_attr.startswith('num_batches_tracked'):
                self.model.state_dict()[ema_attr].copy_(model.state_dict()[model_attr])

# Usage
model = MyModel()
ema = ModelEMA(model, decay=0.9999)

for epoch in range(num_epochs):
    for data, targets in train_loader:
        # Train step
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Update EMA
        ema.update(model)

    # Evaluate with EMA model
    eval_accuracy = evaluate(ema.model, val_loader)
```

### 6.3 Curriculum Learning

**Train on Easy Examples First**

```python
class CurriculumSampler(torch.utils.data.Sampler):
    """
    Sample easier examples first, gradually include harder ones
    """

    def __init__(self, dataset, difficulty_scores, initial_fraction=0.2):
        """
        difficulty_scores: Array of difficulty for each sample (higher = harder)
        initial_fraction: Start with easiest X% of data
        """
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores
        self.initial_fraction = initial_fraction
        self.current_fraction = initial_fraction

        # Sort indices by difficulty
        self.sorted_indices = np.argsort(difficulty_scores)

    def set_epoch(self, epoch, total_epochs):
        """Gradually include more difficult samples"""
        self.current_fraction = self.initial_fraction + \
            (1.0 - self.initial_fraction) * (epoch / total_epochs)

    def __iter__(self):
        # Number of samples to include
        num_samples = int(len(self.dataset) * self.current_fraction)

        # Get indices of easiest samples
        indices = self.sorted_indices[:num_samples]

        # Shuffle
        np.random.shuffle(indices)

        return iter(indices.tolist())

    def __len__(self):
        return int(len(self.dataset) * self.current_fraction)

# Define difficulty (example: prediction entropy, loss, etc.)
def compute_difficulty(model, dataset):
    """Compute difficulty score for each sample"""
    model.eval()
    difficulties = []

    with torch.no_grad():
        for data, _ in dataset:
            output = model(data.unsqueeze(0))
            probs = F.softmax(output, dim=1)

            # Use entropy as difficulty measure
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            difficulties.append(entropy.item())

    return np.array(difficulties)

# Training with curriculum learning
sampler = CurriculumSampler(train_dataset, difficulty_scores)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch, num_epochs)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)

    # Train epoch
    train_epoch(model, train_loader, optimizer, criterion)
```

---

## 7. Neural Architecture Design

### 7.1 Squeeze-and-Excitation (SE) Blocks

**Channel Attention Mechanism**

```python
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block

    Recalibrates channel-wise features by:
    1. Squeeze: Global average pooling
    2. Excitation: FC layers to get channel weights
    3. Scale: Multiply features by weights
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        # Squeeze: Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation: Two FC layers with bottleneck
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()

        # Squeeze: (B, C, H, W) → (B, C, 1, 1) → (B, C)
        y = self.avg_pool(x).view(batch, channels)

        # Excitation: (B, C) → (B, C)
        y = self.fc(y).view(batch, channels, 1, 1)

        # Scale: Element-wise multiplication
        return x * y.expand_as(x)

# Use in residual block
class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SEResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE block
        self.se = SEBlock(out_channels)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply SE
        out = self.se(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out
```

### 7.2 Inverted Residuals (MobileNet Style)

**Efficient Mobile Architecture**

```python
class InvertedResidual(nn.Module):
    """
    Inverted residual block from MobileNetV2

    Standard residual: Wide → Narrow → Wide
    Inverted residual: Narrow → Wide → Narrow

    Uses depthwise separable convolutions for efficiency
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []

        # Expand (1x1 convolution)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # Depthwise (3x3 convolution)
        layers.extend([
            # Depthwise: each channel processed separately
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # Project (1x1 convolution, linear activation)
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
```

### 7.3 Efficient Channel Attention (ECA)

**Parameter-Free Channel Attention**

```python
class ECABlock(nn.Module):
    """
    Efficient Channel Attention

    More efficient than SE block
    Uses 1D convolution for local cross-channel interaction
    """

    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()

        # Adaptive kernel size
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze
        y = self.avg_pool(x)

        # 1D convolution along channel dimension
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Excitation
        y = self.sigmoid(y)

        return x * y.expand_as(x)
```

---

## 8. Best Practices and Tips

### 8.1 Debugging Checklist

**When Training Fails:**

1. **Overfit single batch first**
   ```python
   # Take one batch and overfit it
   data, targets = next(iter(train_loader))
   for i in range(1000):
       optimizer.zero_grad()
       outputs = model(data)
       loss = criterion(outputs, targets)
       loss.backward()
       optimizer.step()
       if i % 100 == 0:
           print(f'Step {i}, Loss: {loss.item():.4f}')

   # If this doesn't work, problem is in model/loss, not data
   ```

2. **Check data**
   ```python
   # Visualize samples
   import matplotlib.pyplot as plt
   data, targets = next(iter(train_loader))
   plt.imshow(data[0].permute(1, 2, 0))
   plt.title(f'Label: {targets[0]}')
   plt.show()

   # Check labels distribution
   from collections import Counter
   labels = [label for _, label in train_dataset]
   print(Counter(labels))
   ```

3. **Check gradients**
   ```python
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f'{name}: {param.grad.abs().mean():.6f}')
       else:
           print(f'{name}: No gradient!')
   ```

4. **Check learning rate**
   ```python
   # Try multiple learning rates
   lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
   for lr in lrs:
       model = MyModel()
       optimizer = torch.optim.Adam(model.parameters(), lr=lr)
       # Train for few steps and check loss
   ```

### 8.2 Training Tips

**Architecture Design:**
- Start simple, add complexity gradually
- Use batch normalization after convolutions
- Use residual connections for deep networks
- Use global average pooling instead of FC layers

**Hyperparameters:**
- Batch size: As large as GPU memory allows (32-512)
- Learning rate: Use LR finder, typically 1e-3 to 1e-4
- Optimizer: AdamW for most tasks
- Weight decay: 1e-4 to 1e-2

**Regularization:**
- Always use data augmentation
- Dropout: 0.1-0.5 depending on model size
- Weight decay: Prevent overfitting
- Early stopping: Monitor validation loss

**Monitoring:**
- Log training and validation metrics
- Plot loss curves
- Save checkpoints regularly
- Use TensorBoard or Weights & Biases

### 8.3 Common Mistakes to Avoid

❌ **Don't:**
1. Train without validation set
2. Use batch norm with batch size < 16
3. Forget to call model.eval() during evaluation
4. Use different preprocessing for train/val/test
5. Ignore data imbalance
6. Use too high learning rate
7. Train on unshuffled data
8. Forget to zero gradients
9. Use softmax before cross-entropy loss (PyTorch includes it)
10. Deploy model without testing edge cases

✅ **Do:**
1. Always shuffle training data
2. Use proper train/val/test splits
3. Monitor both training and validation metrics
4. Use checkpointing to save best model
5. Test on out-of-distribution data
6. Profile your code to find bottlenecks
7. Use mixed precision training
8. Document hyperparameters
9. Version your data and code
10. Start with simple baselines

---

## Interview Questions

### Generative Models

**Q: Explain the reparameterization trick in VAEs and why it's necessary.**

A: The reparameterization trick makes sampling operations differentiable. Instead of sampling z ~ N(μ, σ²) directly (which is non-differentiable), we sample ε ~ N(0, 1) and compute z = μ + σ * ε. This separates the randomness (ε) from the parameters (μ, σ), allowing gradients to flow through μ and σ during backpropagation.

**Q: What is mode collapse in GANs and how can you detect/prevent it?**

A: Mode collapse occurs when the generator produces limited variety of samples, even though real data is diverse.

Detection:
- Visual inspection of generated samples
- Measure diversity metrics (Inception Score, FID)
- Monitor discriminator loss (stays high if collapsed)

Prevention:
- Use Wasserstein GAN with gradient penalty
- Mini-batch discrimination
- Feature matching
- Use multiple GANs (ensemble)

**Q: Compare VAEs and GANs. When would you use each?**

A:
- **VAEs**: Stable training, probabilistic framework, good for interpolation and disentangled representations. Lower sample quality.
  - Use for: Anomaly detection, representation learning, controlled generation

- **GANs**: High-quality samples, unstable training, no explicit density model.
  - Use for: High-resolution image generation, style transfer, data augmentation

**Q: Explain how diffusion models work at a high level.**

A: Diffusion models learn to reverse a gradual noising process:
1. **Forward process** (fixed): Gradually add Gaussian noise to data over T steps until it becomes pure noise
2. **Reverse process** (learned): Train a neural network to denoise, starting from noise and gradually recovering the original data
3. **Sampling**: Start with noise and iteratively denoise using the learned model

Benefits: State-of-the-art quality, stable training. Downside: Slow sampling (need many steps).

### Optimization

**Q: Explain the difference between Adam and AdamW.**

A: AdamW decouples weight decay from the adaptive learning rate mechanism:
- **Adam**: Weight decay is applied through L2 regularization in the loss, which interacts with adaptive learning rates
- **AdamW**: Weight decay is applied directly to parameters, independent of gradients

AdamW is generally superior because proper weight decay regularization helps generalization.

**Q: What is the One Cycle learning rate policy and why is it effective?**

A: One Cycle gradually increases LR from low to high (warm-up), then decreases to very low:
1. **Warm-up** (30%): LR increases from initial_lr to max_lr
2. **Annealing** (70%): LR decreases from max_lr to min_lr

Benefits:
- Faster convergence
- Better generalization
- Regularization effect (high LR acts as noise)
- Avoids local minima

**Q: When and why would you use gradient clipping?**

A: Gradient clipping prevents exploding gradients by capping gradient magnitudes.

Use when:
- Training RNNs/LSTMs (prone to exploding gradients)
- Very deep networks
- Unstable training with loss spikes

Methods:
- Clip by norm: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`
- Clip by value: Less common

### Regularization

**Q: Explain mixup and cutmix. How do they help?**

A: Both are data augmentation techniques that create virtual training examples:

**Mixup**: Linearly interpolate images and labels
- x_new = λ*x_i + (1-λ)*x_j
- y_new = λ*y_i + (1-λ)*y_j

**CutMix**: Cut a patch from one image and paste to another, mix labels proportionally

Benefits:
- Stronger regularization than standard augmentation
- Better calibrated predictions (less overconfident)
- Improved adversarial robustness
- Force model to learn from entire image, not just discriminative patches

**Q: What is label smoothing and why does it help?**

A: Label smoothing prevents overconfident predictions by softening hard targets:
- Instead of [0, 0, 1, 0, 0]
- Use [ε/K, ε/K, 1-ε+ε/K, ε/K, ε/K]

Benefits:
- Better calibrated probabilities
- Prevents overfitting to noisy labels
- Improves generalization
- Acts as regularizer

Typical ε: 0.1

### Self-Supervised Learning

**Q: Explain contrastive learning (SimCLR). How does it learn without labels?**

A: SimCLR learns by maximizing agreement between different augmented views of the same image:

1. Take an image, create two augmented versions (different crops, colors, etc.)
2. Pass both through encoder → get embeddings z_i, z_j
3. Treat (z_i, z_j) as positive pair (should be similar)
4. Treat other images in batch as negative pairs (should be different)
5. Use contrastive loss (NT-Xent) to pull positives together, push negatives apart

Key insights:
- Strong augmentation is crucial
- Large batch sizes help (more negatives)
- Temperature scaling controls hardness
- Momentum encoder (MoCo) uses queue for more negatives

**Q: What is knowledge distillation? When would you use it?**

A: Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model:

Loss = α * CE(student, labels) + (1-α) * KL(student, teacher)

Temperature parameter softens distributions to reveal dark knowledge (relationships between classes).

Use when:
- Need smaller model for deployment
- Want faster inference
- Limited labeled data (teacher can be pre-trained)
- Ensemble distillation (multiple teachers → single student)

### Model Compression

**Q: Compare pruning and quantization for model compression.**

A:
**Pruning**: Remove unnecessary weights/neurons
- Unstructured: Remove individual weights (needs sparse matrix support)
- Structured: Remove entire channels/filters (faster, easier deployment)
- Benefits: Smaller model, faster inference
- Challenges: Needs fine-tuning, irregular sparsity hard to accelerate

**Quantization**: Reduce precision (32-bit → 8-bit)
- Post-training quantization (PTQ): Quantize after training
- Quantization-aware training (QAT): Train with quantization in mind
- Benefits: 4x smaller, 2-4x faster, hardware support
- Minimal accuracy loss with QAT

Can combine both for maximum compression.

**Q: What is the difference between knowledge distillation and transfer learning?**

A:
**Transfer learning**: Use pre-trained model weights as initialization
- Student architecture often same as teacher
- Fine-tune on new task/dataset
- Uses hard labels

**Knowledge distillation**: Student learns to mimic teacher's outputs
- Student typically smaller than teacher
- Uses soft labels (probabilities)
- Can transfer to same task or related task

---

## References

1. **VAEs**: Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)
2. **GANs**: Goodfellow et al., "Generative Adversarial Networks" (2014)
3. **WGAN**: Arjovsky et al., "Wasserstein GAN" (2017)
4. **SimCLR**: Chen et al., "A Simple Framework for Contrastive Learning" (2020)
5. **MoCo**: He et al., "Momentum Contrast for Unsupervised Visual Representation Learning" (2020)
6. **MAE**: He et al., "Masked Autoencoders Are Scalable Vision Learners" (2021)
7. **Diffusion**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
8. **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
9. **Mixup**: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2018)
10. **Label Smoothing**: Szegedy et al., "Rethinking the Inception Architecture" (2016)
11. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling" (2019)
12. **One Cycle**: Smith & Topin, "Super-Convergence" (2018)

---

**Last Updated:** October 2025

*For questions or suggestions, please open an issue on the repository.*
