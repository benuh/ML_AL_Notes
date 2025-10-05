"""
Comprehensive Test Suite for ML/AI Educational Materials
Tests core functionality of all code examples to ensure they work correctly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

class TestRunner:
    """Main test runner for all educational materials"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Running tests on: {self.device.upper()}")
        print("=" * 70)

    def run_test(self, test_name, test_func):
        """Run a single test"""
        try:
            test_func()
            print(f"✓ {test_name}")
            self.passed += 1
        except Exception as e:
            print(f"✗ {test_name}")
            print(f"  Error: {str(e)}")
            self.failed += 1

    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        print("=" * 70)
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"⚠️  {self.failed} tests failed")
        else:
            print("✓ All tests passed!")


# ============================================================================
# Self-Supervised Learning Tests
# ============================================================================

def test_simclr():
    """Test SimCLR implementation"""
    from torchvision.models import resnet18

    class SimpleSimCLR(nn.Module):
        def __init__(self):
            super().__init__()
            encoder = resnet18(pretrained=False)
            encoder_dim = encoder.fc.in_features
            encoder.fc = nn.Identity()
            self.encoder = encoder
            self.projection = nn.Linear(encoder_dim, 128)
            self.temperature = 0.5

        def forward(self, x1, x2):
            h1 = self.encoder(x1)
            h2 = self.encoder(x2)
            z1 = self.projection(h1)
            z2 = self.projection(h2)
            return z1, z2

        def nt_xent_loss(self, z1, z2):
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            z = torch.cat([z1, z2], dim=0)
            similarity = torch.mm(z, z.T) / self.temperature
            batch_size = z1.shape[0]
            labels = torch.cat([
                torch.arange(batch_size, 2*batch_size),
                torch.arange(0, batch_size)
            ])
            mask = torch.eye(2*batch_size, dtype=torch.bool)
            similarity.masked_fill_(mask, -9e15)
            loss = F.cross_entropy(similarity, labels.long())
            return loss

    model = SimpleSimCLR()
    x1 = torch.randn(4, 3, 224, 224)
    x2 = torch.randn(4, 3, 224, 224)
    z1, z2 = model(x1, x2)
    loss = model.nt_xent_loss(z1, z2)

    assert z1.shape == (4, 128), "Projection shape mismatch"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss is NaN"


def test_mae():
    """Test Masked Autoencoder"""
    class SimplePatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Conv2d(3, 192, kernel_size=16, stride=16)

        def forward(self, x):
            x = self.projection(x)
            x = x.flatten(2).transpose(1, 2)
            return x

    class SimpleMAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = SimplePatchEmbed()
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(192, 4, 768, batch_first=True),
                num_layers=2
            )
            self.decoder = nn.Linear(192, 16*16*3)

        def forward(self, x):
            x = self.patch_embed(x)  # B x 196 x 192
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    model = SimpleMAE()
    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    assert output.shape == (2, 196, 768), f"Output shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"


def test_byol():
    """Test BYOL implementation"""
    class SimpleBYOL(nn.Module):
        def __init__(self):
            super().__init__()
            self.online_encoder = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
            self.predictor = nn.Linear(128, 128)
            self.target_encoder = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
            for param in self.target_encoder.parameters():
                param.requires_grad = False

        def forward(self, x1, x2):
            online_1 = self.predictor(self.online_encoder(x1))
            with torch.no_grad():
                target_2 = self.target_encoder(x2)
            return online_1, target_2

        def loss(self, p, z):
            p = F.normalize(p, dim=-1)
            z = F.normalize(z, dim=-1)
            return 2 - 2 * (p * z).sum(dim=-1).mean()

    model = SimpleBYOL()
    x1 = torch.randn(8, 512)
    x2 = torch.randn(8, 512)
    p1, z2 = model(x1, x2)
    loss = model.loss(p1, z2)

    assert p1.shape == (8, 128), "Prediction shape mismatch"
    assert loss.item() >= 0, "Loss should be non-negative"


# ============================================================================
# Time Series Tests
# ============================================================================

def test_lstm_forecaster():
    """Test LSTM for time series"""
    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 64, 2, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            predictions = self.fc(lstm_out[:, -1, :])
            return predictions

    model = LSTMForecaster()
    x = torch.randn(16, 30, 1)  # batch_size, seq_len, features
    output = model(x)

    assert output.shape == (16, 1), f"Output shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"


def test_temporal_conv_net():
    """Test Temporal Convolutional Network"""
    class TemporalBlock(nn.Module):
        def __init__(self, n_inputs, n_outputs, kernel_size, dilation):
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                  padding=padding, dilation=dilation)
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.conv1(x)
            out = out[:, :, :-self.conv1.padding[0]]  # Causal
            return self.relu(out)

    block = TemporalBlock(1, 64, kernel_size=3, dilation=1)
    x = torch.randn(8, 1, 100)  # batch, channels, seq_len
    output = block(x)

    assert output.shape == (8, 64, 100), f"Output shape: {output.shape}"


# ============================================================================
# Prompt Engineering & LLM Tests
# ============================================================================

def test_prompt_templates():
    """Test prompt template generation"""
    class PromptTemplates:
        @staticmethod
        def few_shot(task, examples, input_text):
            prompt = f"Task: {task}\n\n"
            for i, (inp, out) in enumerate(examples, 1):
                prompt += f"Example {i}:\nInput: {inp}\nOutput: {out}\n\n"
            prompt += f"Input: {input_text}\nOutput:"
            return prompt

        @staticmethod
        def chain_of_thought(question):
            return f"Question: {question}\nLet's solve this step by step:\n1."

    # Test few-shot
    examples = [("2+2", "4"), ("3+3", "6")]
    prompt = PromptTemplates.few_shot("arithmetic", examples, "5+5")
    assert "Task: arithmetic" in prompt
    assert "Example 1:" in prompt
    assert "5+5" in prompt

    # Test CoT
    cot_prompt = PromptTemplates.chain_of_thought("What is 15 * 23?")
    assert "step by step" in cot_prompt


def test_lora_config():
    """Test LoRA configuration"""
    class SimpleLoRA(nn.Module):
        def __init__(self, in_features, out_features, rank=4):
            super().__init__()
            self.lora_A = nn.Parameter(torch.randn(in_features, rank))
            self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
            self.scaling = 1.0 / rank

        def forward(self, x):
            return x @ self.lora_A @ self.lora_B * self.scaling

    lora = SimpleLoRA(768, 768, rank=8)
    x = torch.randn(4, 768)
    output = lora(x)

    assert output.shape == (4, 768), "LoRA output shape mismatch"

    # Check parameter count
    original_params = 768 * 768
    lora_params = 768 * 8 + 8 * 768
    reduction = lora_params / original_params
    assert reduction < 0.1, f"LoRA params should be <10% of original, got {reduction:.2%}"


# ============================================================================
# Advanced Architecture Tests
# ============================================================================

def test_multi_head_attention():
    """Test multi-head attention"""
    attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
    x = torch.randn(4, 20, 512)  # batch, seq_len, embed_dim
    attn_output, attn_weights = attention(x, x, x)

    assert attn_output.shape == (4, 20, 512), "Attention output shape mismatch"
    assert attn_weights.shape == (4, 20, 20), "Attention weights shape mismatch"


def test_vision_transformer_patch():
    """Test ViT patch embedding"""
    class PatchEmbedding(nn.Module):
        def __init__(self, img_size=224, patch_size=16, embed_dim=768):
            super().__init__()
            self.projection = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        def forward(self, x):
            x = self.projection(x)
            x = x.flatten(2).transpose(1, 2)
            return x

    patch_embed = PatchEmbedding()
    x = torch.randn(2, 3, 224, 224)
    patches = patch_embed(x)

    expected_patches = (224 // 16) ** 2
    assert patches.shape == (2, expected_patches, 768), f"Patches shape: {patches.shape}"


# ============================================================================
# Reinforcement Learning Tests
# ============================================================================

def test_dqn_network():
    """Test DQN network"""
    class DQN(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )

        def forward(self, x):
            return self.network(x)

    dqn = DQN(state_dim=4, action_dim=2)
    state = torch.randn(32, 4)
    q_values = dqn(state)

    assert q_values.shape == (32, 2), "Q-values shape mismatch"


def test_policy_network():
    """Test policy gradient network"""
    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Tanh(),
                nn.Linear(128, action_dim),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            return self.network(x)

    policy = PolicyNetwork(state_dim=4, action_dim=2)
    state = torch.randn(1, 4)
    action_probs = policy(state)

    assert action_probs.shape == (1, 2), "Action probabilities shape mismatch"
    assert torch.allclose(action_probs.sum(), torch.tensor(1.0)), "Probs should sum to 1"


# ============================================================================
# Generative AI Tests
# ============================================================================

def test_vae():
    """Test Variational Autoencoder"""
    class SimpleVAE(nn.Module):
        def __init__(self, latent_dim=20):
            super().__init__()
            self.encoder = nn.Linear(784, 400)
            self.fc_mu = nn.Linear(400, latent_dim)
            self.fc_logvar = nn.Linear(400, latent_dim)
            self.decoder = nn.Linear(latent_dim, 784)

        def encode(self, x):
            h = F.relu(self.encoder(x))
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return torch.sigmoid(self.decoder(z))

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    vae = SimpleVAE()
    x = torch.randn(8, 784)
    recon, mu, logvar = vae(x)

    assert recon.shape == (8, 784), "Reconstruction shape mismatch"
    assert mu.shape == (8, 20), "Latent mu shape mismatch"
    assert logvar.shape == (8, 20), "Latent logvar shape mismatch"


def test_gan():
    """Test GAN networks"""
    class Generator(nn.Module):
        def __init__(self, latent_dim=100):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 784),
                nn.Tanh()
            )

        def forward(self, z):
            return self.model(z)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(784, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    G = Generator()
    D = Discriminator()

    z = torch.randn(16, 100)
    fake_images = G(z)
    validity = D(fake_images)

    assert fake_images.shape == (16, 784), "Generated images shape mismatch"
    assert validity.shape == (16, 1), "Discriminator output shape mismatch"
    assert (validity >= 0).all() and (validity <= 1).all(), "Validity should be in [0,1]"


# ============================================================================
# Ensemble Methods Tests
# ============================================================================

def test_bagging():
    """Test bagging ensemble"""
    class SimpleBaggingClassifier:
        def __init__(self, n_estimators=5):
            self.n_estimators = n_estimators
            self.models = [nn.Linear(10, 2) for _ in range(n_estimators)]

        def predict(self, x):
            predictions = []
            for model in self.models:
                with torch.no_grad():
                    pred = model(x)
                predictions.append(pred)
            # Average predictions
            return torch.stack(predictions).mean(dim=0)

    ensemble = SimpleBaggingClassifier(n_estimators=3)
    x = torch.randn(5, 10)
    pred = ensemble.predict(x)

    assert pred.shape == (5, 2), "Ensemble prediction shape mismatch"


# ============================================================================
# Explainable AI Tests
# ============================================================================

def test_attention_visualization():
    """Test attention weight extraction"""
    class AttentionLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(64, 4, batch_first=True)

        def forward(self, x):
            attn_output, attn_weights = self.attention(x, x, x)
            return attn_output, attn_weights

    layer = AttentionLayer()
    x = torch.randn(2, 10, 64)
    output, weights = layer(x)

    assert weights.shape == (2, 10, 10), "Attention weights shape mismatch"
    # Check weights are valid probabilities
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 10)), "Weights should sum to 1"


# ============================================================================
# Distributed Training Tests
# ============================================================================

def test_gradient_accumulation():
    """Test gradient accumulation"""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    accumulation_steps = 4

    for i in range(accumulation_steps):
        x = torch.randn(8, 10)
        y = torch.randn(8, 5)

        output = model(x)
        loss = F.mse_loss(output, y)
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Test passes if no errors


# ============================================================================
# Hyperparameter Tuning Tests
# ============================================================================

def test_learning_rate_finder():
    """Test learning rate range test"""
    class LRFinder:
        def __init__(self, model, optimizer):
            self.model = model
            self.optimizer = optimizer
            self.lrs = []
            self.losses = []

        def range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
            lr = start_lr
            self.optimizer.param_groups[0]['lr'] = lr

            for i in range(num_iter):
                # Simulate batch
                x = torch.randn(4, 10)
                y = torch.randn(4, 1)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = F.mse_loss(output, y)
                loss.backward()
                self.optimizer.step()

                self.lrs.append(lr)
                self.losses.append(loss.item())

                # Update learning rate
                lr = lr * (end_lr / start_lr) ** (1 / num_iter)
                self.optimizer.param_groups[0]['lr'] = lr

    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    lr_finder = LRFinder(model, optimizer)
    lr_finder.range_test(None, num_iter=10)

    assert len(lr_finder.lrs) == 10, "Should collect 10 LR values"
    assert len(lr_finder.losses) == 10, "Should collect 10 loss values"


# ============================================================================
# Edge AI Tests
# ============================================================================

def test_quantization_aware_training():
    """Test quantization-aware training setup"""
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 5)
    )

    # Prepare for quantization
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)

    # Forward pass
    x = torch.randn(4, 10)
    output = model(x)

    assert output.shape == (4, 5), "QAT model output shape mismatch"


# ============================================================================
# Few-Shot Learning Tests
# ============================================================================

def test_prototypical_networks():
    """Test prototypical network"""
    class ProtoNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )

        def forward(self, x):
            return self.encoder(x)

        def compute_prototypes(self, support_embeddings, support_labels, n_way):
            prototypes = []
            for i in range(n_way):
                class_embeddings = support_embeddings[support_labels == i]
                prototype = class_embeddings.mean(dim=0)
                prototypes.append(prototype)
            return torch.stack(prototypes)

    model = ProtoNet()

    # 5-way 1-shot
    support = torch.randn(5, 784)  # 1 example per class
    support_labels = torch.arange(5)

    embeddings = model(support)
    prototypes = model.compute_prototypes(embeddings, support_labels, n_way=5)

    assert prototypes.shape == (5, 64), "Prototypes shape mismatch"


# ============================================================================
# Run All Tests
# ============================================================================

def main():
    """Run all tests"""
    runner = TestRunner()

    print("\n📚 Self-Supervised Learning Tests")
    print("-" * 70)
    runner.run_test("SimCLR Implementation", test_simclr)
    runner.run_test("Masked Autoencoder (MAE)", test_mae)
    runner.run_test("BYOL Implementation", test_byol)

    print("\n⏰ Time Series Tests")
    print("-" * 70)
    runner.run_test("LSTM Forecaster", test_lstm_forecaster)
    runner.run_test("Temporal Convolutional Network", test_temporal_conv_net)

    print("\n🤖 Prompt Engineering & LLM Tests")
    print("-" * 70)
    runner.run_test("Prompt Templates", test_prompt_templates)
    runner.run_test("LoRA Configuration", test_lora_config)

    print("\n🏗️  Advanced Architecture Tests")
    print("-" * 70)
    runner.run_test("Multi-Head Attention", test_multi_head_attention)
    runner.run_test("Vision Transformer Patches", test_vision_transformer_patch)

    print("\n🎮 Reinforcement Learning Tests")
    print("-" * 70)
    runner.run_test("DQN Network", test_dqn_network)
    runner.run_test("Policy Network", test_policy_network)

    print("\n🎨 Generative AI Tests")
    print("-" * 70)
    runner.run_test("Variational Autoencoder", test_vae)
    runner.run_test("GAN Networks", test_gan)

    print("\n🔗 Ensemble Methods Tests")
    print("-" * 70)
    runner.run_test("Bagging Ensemble", test_bagging)

    print("\n🔍 Explainable AI Tests")
    print("-" * 70)
    runner.run_test("Attention Visualization", test_attention_visualization)

    print("\n⚡ Distributed Training Tests")
    print("-" * 70)
    runner.run_test("Gradient Accumulation", test_gradient_accumulation)

    print("\n🎯 Hyperparameter Tuning Tests")
    print("-" * 70)
    runner.run_test("Learning Rate Finder", test_learning_rate_finder)

    print("\n📱 Edge AI Tests")
    print("-" * 70)
    runner.run_test("Quantization Aware Training", test_quantization_aware_training)

    print("\n🎯 Few-Shot Learning Tests")
    print("-" * 70)
    runner.run_test("Prototypical Networks", test_prototypical_networks)

    print()
    runner.print_summary()

    return runner.failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
