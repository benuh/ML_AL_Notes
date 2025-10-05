# Generative AI Deep Dive

Master the art of generating new data: images, text, audio, and more.

## Table of Contents
1. [Generative Models Overview](#generative-models-overview)
2. [Variational Autoencoders (VAEs)](#variational-autoencoders)
3. [Generative Adversarial Networks (GANs)](#generative-adversarial-networks)
4. [Diffusion Models](#diffusion-models)
5. [Autoregressive Models](#autoregressive-models)
6. [Advanced Generative Techniques](#advanced-generative-techniques)
7. [Applications](#applications)

---

## Generative Models Overview

### What are Generative Models?

**Goal:** Learn the underlying data distribution P(x) to generate new samples.

**Types:**
1. **Explicit Density Models** - Directly model P(x)
   - VAEs, Autoregressive models, Normalizing Flows

2. **Implicit Density Models** - Generate samples without explicit P(x)
   - GANs

3. **Energy-Based Models** - Model energy function
   - Diffusion Models

**Evaluation Metrics:**
- **Inception Score (IS)** - Higher is better
- **Fréchet Inception Distance (FID)** - Lower is better
- **Perceptual Path Length (PPL)** - For image quality
- **Human Evaluation** - Gold standard

---

## Variational Autoencoders

### VAE Fundamentals

**Key Idea:** Learn a latent representation z that can generate data x.

**Objective:**
```
Maximize: ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
         = Reconstruction Loss - KL Divergence
```

### Basic VAE Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """Variational Autoencoder"""

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ * ε"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent variable to reconstruction"""
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss: reconstruction + KL divergence"""
        # Reconstruction loss (BCE)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss

    def sample(self, num_samples=64):
        """Generate new samples"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.fc_mu.out_features)
            samples = self.decode(z)
        return samples

# Training
def train_vae(model, train_loader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(data.size(0), -1)  # Flatten

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

# Usage
vae = VAE(input_dim=784, latent_dim=20)
train_vae(vae, train_loader, epochs=10)

# Generate new samples
samples = vae.sample(num_samples=64)
```

---

### Conditional VAE (CVAE)

```python
class ConditionalVAE(nn.Module):
    """Conditional VAE - generate conditioned on labels"""

    def __init__(self, input_dim=784, num_classes=10, hidden_dim=400, latent_dim=20):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.num_classes = num_classes

    def encode(self, x, c):
        """Encode with condition"""
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, c):
        """Decode with condition"""
        inputs = torch.cat([z, c], dim=1)
        return self.decoder(inputs)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def sample(self, num_samples, class_label):
        """Generate samples of specific class"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.fc_mu.out_features)
            c = F.one_hot(torch.tensor([class_label] * num_samples),
                         num_classes=self.num_classes).float()
            samples = self.decode(z, c)
        return samples
```

---

### β-VAE (Disentangled Representations)

```python
class BetaVAE(VAE):
    """Beta-VAE for disentangled representations"""

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, beta=4.0):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.beta = beta

    def loss_function(self, recon_x, x, mu, logvar):
        """Beta-VAE loss with weighted KL divergence"""
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Beta weight on KL divergence
        return recon_loss + self.beta * kl_loss
```

---

## Generative Adversarial Networks

### Basic GAN

**Key Idea:** Train Generator and Discriminator in adversarial game.

**Objective:**
```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

```python
class Generator(nn.Module):
    """GAN Generator"""

    def __init__(self, latent_dim=100, output_dim=784, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    """GAN Discriminator"""

    def __init__(self, input_dim=784, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(generator, discriminator, train_loader, epochs=50,
              lr=2e-4, latent_dim=100):
    """Train GAN"""

    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for batch_idx, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1)

            # Labels
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # =================== Train Discriminator ===================
            d_optimizer.zero_grad()

            # Real images
            d_real = discriminator(real_images)
            d_loss_real = criterion(d_real, real_labels)

            # Fake images
            z = torch.randn(batch_size, latent_dim)
            fake_images = generator(z)
            d_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(d_fake, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # =================== Train Generator ===================
            g_optimizer.zero_grad()

            z = torch.randn(batch_size, latent_dim)
            fake_images = generator(z)
            d_fake = discriminator(fake_images)

            # Generator loss: fool discriminator
            g_loss = criterion(d_fake, real_labels)
            g_loss.backward()
            g_optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

# Usage
generator = Generator(latent_dim=100, output_dim=784)
discriminator = Discriminator(input_dim=784)
train_gan(generator, discriminator, train_loader, epochs=50)

# Generate samples
with torch.no_grad():
    z = torch.randn(64, 100)
    fake_images = generator(z)
```

---

### DCGAN (Deep Convolutional GAN)

```python
class DCGANGenerator(nn.Module):
    """DCGAN Generator"""

    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State: 512 x 4 x 4

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: 256 x 8 x 8

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: 128 x 16 x 16

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State: 64 x 32 x 32

            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: channels x 64 x 64
        )

    def forward(self, z):
        return self.model(z)

class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator"""

    def __init__(self, channels=1):
        super().__init__()
        self.model = nn.Sequential(
            # Input: channels x 64 x 64
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)
```

---

### Wasserstein GAN (WGAN)

**Key Innovation:** Use Wasserstein distance instead of JS divergence.

```python
class WGANCritic(nn.Module):
    """WGAN Critic (not a discriminator)"""

    def __init__(self, input_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
            # No sigmoid! Output real numbers
        )

    def forward(self, x):
        return self.model(x)

def train_wgan(generator, critic, train_loader, epochs=50,
               lr=5e-5, latent_dim=100, n_critic=5, clip_value=0.01):
    """Train WGAN with weight clipping"""

    g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=lr)
    c_optimizer = torch.optim.RMSprop(critic.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch_idx, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1)

            # =================== Train Critic ===================
            for _ in range(n_critic):
                c_optimizer.zero_grad()

                # Real images
                c_real = critic(real_images)

                # Fake images
                z = torch.randn(batch_size, latent_dim)
                fake_images = generator(z).detach()
                c_fake = critic(fake_images)

                # Wasserstein loss
                c_loss = -(torch.mean(c_real) - torch.mean(c_fake))
                c_loss.backward()
                c_optimizer.step()

                # Clip weights
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # =================== Train Generator ===================
            g_optimizer.zero_grad()

            z = torch.randn(batch_size, latent_dim)
            fake_images = generator(z)
            g_loss = -torch.mean(critic(fake_images))
            g_loss.backward()
            g_optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], c_loss: {c_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
```

---

### StyleGAN2 Architecture

```python
class MappingNetwork(nn.Module):
    """StyleGAN Mapping Network: Z → W"""

    def __init__(self, latent_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2)
            ])
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""

    def __init__(self, num_features, w_dim=512):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style = nn.Linear(w_dim, num_features * 2)

    def forward(self, x, w):
        style = self.style(w).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, dim=1)
        out = self.norm(x)
        return gamma * out + beta

class StyleGAN2Block(nn.Module):
    """StyleGAN2 synthesis block"""

    def __init__(self, in_channels, out_channels, w_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.conv1(x)
        x = self.adain1(x, w)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.adain2(x, w)
        x = self.activation(x)

        return x
```

---

## Diffusion Models

### Denoising Diffusion Probabilistic Models (DDPM)

**Key Idea:** Learn to reverse a gradual noising process.

**Forward Process (add noise):**
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

**Reverse Process (denoise):**
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

```python
class DiffusionModel(nn.Module):
    """Denoising Diffusion Model"""

    def __init__(self, channels=1, time_steps=1000):
        super().__init__()
        self.time_steps = time_steps

        # Noise schedule
        self.beta = torch.linspace(1e-4, 0.02, time_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # U-Net for denoising
        self.unet = UNet(channels=channels, time_emb_dim=128)

    def forward_diffusion(self, x_0, t):
        """Add noise to x_0 at timestep t"""
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        # x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

        return x_t, noise

    def predict_noise(self, x_t, t):
        """Predict noise at timestep t"""
        return self.unet(x_t, t)

    def sample(self, shape, device='cpu'):
        """Generate samples by reversing diffusion"""
        x_t = torch.randn(shape).to(device)

        for t in reversed(range(self.time_steps)):
            # Predict noise
            t_tensor = torch.tensor([t] * shape[0]).to(device)
            predicted_noise = self.predict_noise(x_t, t_tensor)

            # Compute mean
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]

            # x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t, t))
            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)

            x_t = (
                1 / torch.sqrt(alpha_t) * (
                    x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise
                ) + torch.sqrt(beta_t) * noise
            )

        return x_t

class UNet(nn.Module):
    """U-Net for diffusion model"""

    def __init__(self, channels=1, time_emb_dim=128):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder
        self.enc1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1)

        # Decoder
        self.dec3 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec2 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec1 = nn.Conv2d(64, channels, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t.float().unsqueeze(1))

        # Encoder
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(self.pool(e1)))
        e3 = F.relu(self.enc3(self.pool(e2)))

        # Decoder
        d3 = F.relu(self.dec3(e3))
        d2 = F.relu(self.dec2(self.upsample(d3) + e2))
        d1 = self.dec1(self.upsample(d2) + e1)

        return d1

def train_diffusion(model, train_loader, epochs=50, lr=1e-4):
    """Train diffusion model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch_idx, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()

            # Random timestep
            t = torch.randint(0, model.time_steps, (images.size(0),))

            # Forward diffusion
            x_t, noise = model.forward_diffusion(images, t)

            # Predict noise
            predicted_noise = model.predict_noise(x_t, t)

            # Loss
            loss = F.mse_loss(predicted_noise, noise)

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Usage
model = DiffusionModel(channels=1, time_steps=1000)
train_diffusion(model, train_loader, epochs=50)

# Generate samples
samples = model.sample(shape=(16, 1, 28, 28))
```

---

## Autoregressive Models

### PixelCNN

**Key Idea:** Generate pixels one at a time, conditioned on previous pixels.

```python
class MaskedConv2d(nn.Conv2d):
    """Masked convolution for autoregressive modeling"""

    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))

        # Create mask
        self.mask[:, :, :self.kernel_size[0] // 2, :] = 1
        self.mask[:, :, self.kernel_size[0] // 2, :self.kernel_size[1] // 2] = 1

        if mask_type == 'B':
            self.mask[:, :, self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class PixelCNN(nn.Module):
    """PixelCNN for image generation"""

    def __init__(self, channels=1, hidden_channels=64, num_layers=7):
        super().__init__()

        # First layer (type A mask)
        self.layers = nn.ModuleList([
            MaskedConv2d('A', channels, hidden_channels, 7, padding=3)
        ])

        # Hidden layers (type B mask)
        for _ in range(num_layers - 2):
            self.layers.append(
                MaskedConv2d('B', hidden_channels, hidden_channels, 7, padding=3)
            )

        # Output layer
        self.layers.append(
            MaskedConv2d('B', hidden_channels, channels * 256, 1)
        )

    def forward(self, x):
        out = x
        for layer in self.layers[:-1]:
            out = F.relu(layer(out))

        out = self.layers[-1](out)
        return out

    def sample(self, shape):
        """Generate samples autoregressively"""
        samples = torch.zeros(shape)

        for i in range(shape[2]):
            for j in range(shape[3]):
                logits = self.forward(samples)
                probs = F.softmax(logits[:, :, i, j], dim=1)
                samples[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.0

        return samples
```

---

## Advanced Generative Techniques

### Conditional Generation

```python
class ConditionalGAN(nn.Module):
    """Conditional GAN for class-conditional generation"""

    def __init__(self, latent_dim=100, num_classes=10, img_channels=1):
        super().__init__()
        self.num_classes = num_classes

        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_channels * 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Concatenate noise with label embedding
        label_emb = self.label_embedding(labels)
        gen_input = torch.cat([z, label_emb], dim=1)
        return self.generator(gen_input)
```

---

### Latent Diffusion Models (Stable Diffusion)

```python
class LatentDiffusion(nn.Module):
    """Latent Diffusion Model (like Stable Diffusion)"""

    def __init__(self, vae, unet, text_encoder):
        super().__init__()
        self.vae = vae  # Pre-trained VAE
        self.unet = unet  # Denoising U-Net
        self.text_encoder = text_encoder  # CLIP text encoder

    def encode_image(self, x):
        """Encode image to latent"""
        with torch.no_grad():
            latent = self.vae.encode(x)
        return latent

    def decode_latent(self, z):
        """Decode latent to image"""
        with torch.no_grad():
            image = self.vae.decode(z)
        return image

    def text_to_image(self, text_prompt, num_steps=50):
        """Generate image from text"""
        # Encode text
        text_emb = self.text_encoder(text_prompt)

        # Start from noise
        latent = torch.randn(1, 4, 64, 64)

        # Iterative denoising
        for t in reversed(range(num_steps)):
            # Predict noise with text conditioning
            noise_pred = self.unet(latent, t, text_emb)

            # Denoise step
            latent = self.denoise_step(latent, noise_pred, t)

        # Decode to image
        image = self.decode_latent(latent)

        return image
```

---

## Applications

### Image-to-Image Translation (Pix2Pix)

```python
class Pix2PixGenerator(nn.Module):
    """U-Net generator for Pix2Pix"""

    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(input_channels, 64, norm=False)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Decoder with skip connections
        self.dec4 = self._deconv_block(512, 256)
        self.dec3 = self._deconv_block(512, 128)  # 512 = 256 + 256 (skip)
        self.dec2 = self._deconv_block(256, 64)   # 256 = 128 + 128 (skip)
        self.dec1 = nn.ConvTranspose2d(128, output_channels, 4, 2, 1)

    def _conv_block(self, in_channels, out_channels, norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def _deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder
        d4 = self.dec4(e4)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return torch.tanh(d1)
```

---

### Text-to-Speech (Generative)

```python
class Tacotron2(nn.Module):
    """Text-to-Speech with attention"""

    def __init__(self, vocab_size, embedding_dim=512, encoder_dim=512):
        super().__init__()

        # Text encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, encoder_dim // 2,
                               num_layers=3, bidirectional=True, batch_first=True)

        # Attention
        self.attention = BahdanauAttention(encoder_dim, 1024)

        # Decoder
        self.decoder = nn.LSTMCell(encoder_dim + 80, 1024)  # 80 = mel_channels

        # Mel-spectrogram predictor
        self.mel_linear = nn.Linear(1024, 80)

    def forward(self, text, mel_targets=None):
        # Encode text
        embedded = self.embedding(text)
        encoder_outputs, _ = self.encoder(embedded)

        # Decode
        mel_outputs = []
        decoder_input = torch.zeros(text.size(0), 80)
        hidden = None

        for t in range(mel_targets.size(1) if mel_targets is not None else 100):
            # Attention
            context = self.attention(encoder_outputs, hidden)

            # Decoder step
            decoder_input_cat = torch.cat([decoder_input, context], dim=1)
            hidden = self.decoder(decoder_input_cat, hidden)

            # Predict mel
            mel_output = self.mel_linear(hidden[0])
            mel_outputs.append(mel_output)

            # Next input
            decoder_input = mel_output if mel_targets is None else mel_targets[:, t]

        return torch.stack(mel_outputs, dim=1)
```

---

## Summary

| Model | Type | Training Difficulty | Quality | Control |
|-------|------|---------------------|---------|---------|
| **VAE** | Explicit | Easy | Medium | High (latent space) |
| **GAN** | Implicit | Hard | High | Medium |
| **WGAN** | Implicit | Medium | High | Medium |
| **StyleGAN** | Implicit | Hard | Very High | Very High |
| **Diffusion** | Energy | Medium | Very High | High |
| **Autoregressive** | Explicit | Medium | High | High |

---

## Key Takeaways

1. **VAEs** for structured latent spaces and easy training
2. **GANs** for high-quality images but unstable training
3. **Diffusion models** are SOTA for image/audio generation
4. **Autoregressive models** for sequential data (text, audio)
5. **Conditioning** enables controllable generation
6. **Latent diffusion** (Stable Diffusion) for efficiency
7. **Evaluation** is challenging - use FID, IS, and human judgment

**Production Considerations:**
- **Inference speed**: GANs > VAEs > Diffusion > Autoregressive
- **Training stability**: Diffusion > VAE > GAN
- **Sample quality**: Diffusion ≈ StyleGAN > VAE
- **Controllability**: Conditional models + guidance

**Next Steps:**
- Implement VAE for your dataset
- Train StyleGAN2 or Diffusion model
- Explore Stable Diffusion for text-to-image
- Build conditional generation system
- Study latest papers (DALL-E 3, Midjourney architecture)
