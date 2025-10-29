# ML/AI Visualization Guide: Complete Visual Toolkit

## Table of Contents
1. [Neural Network Architecture Visualization](#neural-network-architecture)
2. [Training Dynamics & Metrics](#training-dynamics)
3. [Attention Mechanism Visualization](#attention-visualization)
4. [Embedding Visualization](#embedding-visualization)
5. [Model Interpretation & Explainability](#model-interpretation)
6. [Performance Metrics Visualization](#performance-metrics)
7. [Interactive Dashboards](#interactive-dashboards)
8. [3D Visualizations](#3d-visualizations)

---

## Neural Network Architecture Visualization

Visualize model architectures and layer connections.

### Network Architecture Diagram

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

class NetworkVisualizer:
    """Visualize neural network architectures"""

    def __init__(self, figsize=(14, 8)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.axis('off')

    def draw_layer(self, x, y, width, height, neurons, label, color='lightblue'):
        """Draw a single layer"""
        # Draw box
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=2
        )
        self.ax.add_patch(box)

        # Add label
        self.ax.text(
            x + width/2, y + height + 0.3,
            label,
            ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )

        # Add neuron count
        self.ax.text(
            x + width/2, y + height/2,
            f'{neurons} neurons',
            ha='center', va='center',
            fontsize=9
        )

        return (x + width/2, y + height/2)

    def draw_arrow(self, start, end, label=''):
        """Draw connection arrow between layers"""
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle='->,head_width=0.4,head_length=0.8',
            color='gray',
            linewidth=2,
            connectionstyle="arc3,rad=0"
        )
        self.ax.add_patch(arrow)

        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            self.ax.text(mid_x, mid_y + 0.3, label,
                        ha='center', fontsize=8, style='italic')

    def visualize_cnn(self):
        """Visualize a CNN architecture"""
        layers = [
            {'x': 0.5, 'y': 3, 'w': 1.2, 'h': 4, 'neurons': '28×28×1',
             'label': 'Input', 'color': 'lightgreen'},

            {'x': 2.2, 'y': 3, 'w': 1.2, 'h': 3.5, 'neurons': '24×24×32',
             'label': 'Conv2D\n5×5', 'color': 'lightblue'},

            {'x': 3.9, 'y': 3.5, 'w': 1.0, 'h': 2.5, 'neurons': '12×12×32',
             'label': 'MaxPool\n2×2', 'color': 'lightyellow'},

            {'x': 5.4, 'y': 3.5, 'w': 1.2, 'h': 2.5, 'neurons': '8×8×64',
             'label': 'Conv2D\n5×5', 'color': 'lightblue'},

            {'x': 7.1, 'y': 4, 'w': 1.0, 'h': 1.5, 'neurons': '4×4×64',
             'label': 'MaxPool\n2×2', 'color': 'lightyellow'},

            {'x': 8.6, 'y': 4, 'w': 0.8, 'h': 1.5, 'neurons': '128',
             'label': 'Dense', 'color': 'lightcoral'},

            {'x': 9.9, 'y': 4.2, 'w': 0.8, 'h': 1.1, 'neurons': '10',
             'label': 'Output', 'color': 'lightgreen'}
        ]

        positions = []
        for layer in layers:
            pos = self.draw_layer(
                layer['x'], layer['y'], layer['w'], layer['h'],
                layer['neurons'], layer['label'], layer['color']
            )
            positions.append(pos)

        # Draw connections
        for i in range(len(positions) - 1):
            self.draw_arrow(positions[i], positions[i + 1])

        self.ax.set_title('CNN Architecture for MNIST Classification',
                         fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        return self.fig

    def visualize_transformer(self):
        """Visualize Transformer architecture"""
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.ax.set_xlim(0, 16)
        self.ax.set_ylim(0, 10)
        self.ax.axis('off')

        # Encoder side
        encoder_x = 2
        self.draw_layer(encoder_x, 0.5, 2.5, 0.8, 'N×d',
                       'Input\nEmbedding', 'lightgreen')
        self.draw_layer(encoder_x, 1.8, 2.5, 0.6, '',
                       'Positional\nEncoding', 'lightyellow')

        # Multi-head attention
        self.draw_layer(encoder_x, 3, 2.5, 1.2, 'h heads',
                       'Multi-Head\nAttention', 'lightblue')
        self.draw_layer(encoder_x, 4.5, 2.5, 0.5, '',
                       'Add & Norm', 'lightgray')

        # Feed forward
        self.draw_layer(encoder_x, 5.5, 2.5, 1, 'd_ff',
                       'Feed\nForward', 'lightcoral')
        self.draw_layer(encoder_x, 6.8, 2.5, 0.5, '',
                       'Add & Norm', 'lightgray')

        # Encoder label
        self.ax.text(encoder_x + 1.25, 8, 'ENCODER',
                    ha='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat'))

        # Decoder side
        decoder_x = 10
        self.draw_layer(decoder_x, 0.5, 2.5, 0.8, 'N×d',
                       'Output\nEmbedding', 'lightgreen')
        self.draw_layer(decoder_x, 1.8, 2.5, 0.6, '',
                       'Positional\nEncoding', 'lightyellow')

        # Masked multi-head attention
        self.draw_layer(decoder_x, 3, 2.5, 1.2, 'h heads',
                       'Masked Multi-Head\nAttention', 'lightblue')
        self.draw_layer(decoder_x, 4.5, 2.5, 0.5, '',
                       'Add & Norm', 'lightgray')

        # Cross attention
        self.draw_layer(decoder_x, 5.5, 2.5, 1.2, 'h heads',
                       'Multi-Head\nAttention', 'lightblue')
        self.draw_layer(decoder_x, 7, 2.5, 0.5, '',
                       'Add & Norm', 'lightgray')

        # Output
        self.draw_layer(decoder_x, 8, 2.5, 0.6, 'vocab',
                       'Linear + Softmax', 'lightgreen')

        # Decoder label
        self.ax.text(decoder_x + 1.25, 9.2, 'DECODER',
                    ha='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat'))

        # Cross connection
        self.draw_arrow((4.5, 6.5), (10, 6.1), 'Encoder\nOutput')

        self.ax.set_title('Transformer Architecture',
                         fontsize=18, fontweight='bold', pad=20)

        plt.tight_layout()
        return self.fig

# Usage
viz = NetworkVisualizer()
fig_cnn = viz.visualize_cnn()
plt.savefig('cnn_architecture.png', dpi=300, bbox_inches='tight')
plt.show()

viz2 = NetworkVisualizer()
fig_transformer = viz2.visualize_transformer()
plt.savefig('transformer_architecture.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Layer-wise Feature Map Visualization

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class FeatureMapVisualizer:
    """Visualize intermediate feature maps in CNN"""

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to capture activations"""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        # Register hooks for conv layers
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                hook = layer.register_forward_hook(get_activation(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def visualize_feature_maps(self, image, max_channels=16):
        """
        Visualize feature maps for each conv layer

        Args:
            image: Input image tensor (1, C, H, W)
            max_channels: Maximum number of channels to display
        """
        self.register_hooks()

        # Forward pass
        with torch.no_grad():
            _ = self.model(image)

        # Plot feature maps for each layer
        num_layers = len(self.activations)

        for layer_idx, (name, activation) in enumerate(self.activations.items()):
            num_channels = min(activation.shape[1], max_channels)

            fig, axes = plt.subplots(
                4, 4, figsize=(12, 12)
            )
            fig.suptitle(f'Feature Maps: {name}', fontsize=16, fontweight='bold')

            for i in range(min(16, num_channels)):
                ax = axes[i // 4, i % 4]

                # Get feature map
                feature_map = activation[0, i].cpu().numpy()

                # Plot
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'Channel {i}', fontsize=10)
                ax.axis('off')

                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Hide empty subplots
            for i in range(num_channels, 16):
                axes[i // 4, i % 4].axis('off')

            plt.tight_layout()
            plt.savefig(f'feature_maps_{name.replace(".", "_")}.png',
                       dpi=150, bbox_inches='tight')
            plt.show()

        self.remove_hooks()

# Example usage
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

# Load model
model = resnet18(pretrained=True)
model.eval()

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Visualize
# image = Image.open('sample_image.jpg')
# image_tensor = transform(image).unsqueeze(0)
#
# visualizer = FeatureMapVisualizer(model)
# visualizer.visualize_feature_maps(image_tensor)
```

---

## Training Dynamics & Metrics

Visualize training progress, loss curves, and learning dynamics.

### Advanced Training Monitor

```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

class TrainingMonitor:
    """Real-time training visualization"""

    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'gradient_norm': []
        }
        self.epochs = []

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, lr, grad_norm):
        """Update metrics"""
        self.epochs.append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['learning_rate'].append(lr)
        self.metrics['gradient_norm'].append(grad_norm)

    def plot(self, save_path=None):
        """Create comprehensive training visualization"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Loss curves
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.epochs, self.metrics['train_loss'],
                label='Train Loss', linewidth=2, color='blue', alpha=0.8)
        ax1.plot(self.epochs, self.metrics['val_loss'],
                label='Val Loss', linewidth=2, color='red', alpha=0.8)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Find minimum validation loss
        min_val_idx = np.argmin(self.metrics['val_loss'])
        ax1.scatter([self.epochs[min_val_idx]],
                   [self.metrics['val_loss'][min_val_idx]],
                   color='red', s=100, zorder=5,
                   label=f'Best Val: {self.metrics["val_loss"][min_val_idx]:.4f}')
        ax1.legend(fontsize=11)

        # Accuracy curves
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(self.epochs, self.metrics['train_acc'],
                label='Train Acc', linewidth=2, color='green', alpha=0.8)
        ax2.plot(self.epochs, self.metrics['val_acc'],
                label='Val Acc', linewidth=2, color='orange', alpha=0.8)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Overfitting gap
        gap = np.array(self.metrics['train_acc']) - np.array(self.metrics['val_acc'])
        ax2_twin = ax2.twinx()
        ax2_twin.fill_between(self.epochs, gap, alpha=0.2, color='purple',
                              label='Overfitting Gap')
        ax2_twin.set_ylabel('Overfitting Gap (%)', fontsize=12)
        ax2_twin.legend(loc='lower right', fontsize=10)

        # Learning rate schedule
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(self.epochs, self.metrics['learning_rate'],
                linewidth=2, color='purple', alpha=0.8)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # Gradient norm
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(self.epochs, self.metrics['gradient_norm'],
                linewidth=2, color='brown', alpha=0.8)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Gradient Norm', fontsize=12)
        ax4.set_title('Gradient Norm (Training Stability)',
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Add horizontal line for gradient explosion threshold
        ax4.axhline(y=10, color='red', linestyle='--', alpha=0.5,
                   label='Explosion Threshold')
        ax4.legend(fontsize=10)

        # Summary statistics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        stats_text = f"""
        Training Summary
        ════════════════════════════

        Best Val Loss:     {min(self.metrics['val_loss']):.4f}
        Best Val Acc:      {max(self.metrics['val_acc']):.2f}%
        Final Train Acc:   {self.metrics['train_acc'][-1]:.2f}%
        Final Val Acc:     {self.metrics['val_acc'][-1]:.2f}%

        Overfitting Gap:   {gap[-1]:.2f}%

        Total Epochs:      {len(self.epochs)}
        Final LR:          {self.metrics['learning_rate'][-1]:.6f}
        Avg Grad Norm:     {np.mean(self.metrics['gradient_norm']):.4f}
        """

        ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Training Monitoring Dashboard',
                    fontsize=18, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        return fig

# Example usage
monitor = TrainingMonitor()

# Simulate training (replace with actual training loop)
for epoch in range(100):
    train_loss = 2.0 * np.exp(-epoch/20) + 0.1 + np.random.rand()*0.05
    val_loss = train_loss + 0.2 + np.random.rand()*0.1
    train_acc = (1 - np.exp(-epoch/15)) * 95 + np.random.rand()*2
    val_acc = train_acc - 5 - np.random.rand()*3
    lr = 0.001 * (0.95 ** (epoch // 10))
    grad_norm = 5 - 4 * (epoch / 100) + np.random.rand()

    monitor.update(epoch, train_loss, val_loss, train_acc, val_acc, lr, grad_norm)

# Plot results
monitor.plot(save_path='training_monitor.png')
```

### Loss Landscape Visualization

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LossLandscapeVisualizer:
    """Visualize loss landscape around current parameters"""

    def __init__(self, model, loss_fn, data_loader):
        self.model = model
        self.loss_fn = loss_fn
        self.data_loader = data_loader

    def compute_loss(self, alpha, beta, direction1, direction2, original_params):
        """Compute loss at a point in parameter space"""
        # Perturb parameters
        with torch.no_grad():
            for i, param in enumerate(self.model.parameters()):
                param.data = original_params[i] + alpha * direction1[i] + beta * direction2[i]

        # Compute loss
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.data_loader:
                inputs, targets = batch[0].cuda(), batch[1].cuda()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

                if num_batches >= 10:  # Limit for speed
                    break

        # Restore original parameters
        with torch.no_grad():
            for i, param in enumerate(self.model.parameters()):
                param.data = original_params[i]

        return total_loss / num_batches

    def visualize_2d(self, num_points=25, distance=1.0):
        """Create 2D loss landscape visualization"""
        # Save original parameters
        original_params = [p.clone() for p in self.model.parameters()]

        # Generate random directions
        direction1 = [torch.randn_like(p) for p in self.model.parameters()]
        direction2 = [torch.randn_like(p) for p in self.model.parameters()]

        # Normalize directions
        for d in [direction1, direction2]:
            norm = sum(torch.norm(w).item() ** 2 for w in d) ** 0.5
            for i in range(len(d)):
                d[i] = d[i] / norm * distance

        # Create grid
        alphas = np.linspace(-1, 1, num_points)
        betas = np.linspace(-1, 1, num_points)

        losses = np.zeros((num_points, num_points))

        print("Computing loss landscape...")
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                losses[i, j] = self.compute_loss(
                    alpha, beta, direction1, direction2, original_params
                )
            print(f"Progress: {(i+1)/num_points*100:.1f}%", end='\r')

        # Plot
        fig = plt.figure(figsize=(16, 6))

        # 2D contour plot
        ax1 = fig.add_subplot(121)
        contour = ax1.contourf(alphas, betas, losses.T, levels=20, cmap='viridis')
        ax1.scatter([0], [0], color='red', s=200, marker='*',
                   zorder=5, label='Current Position', edgecolors='white', linewidth=2)
        plt.colorbar(contour, ax=ax1, label='Loss')
        ax1.set_xlabel('Direction 1', fontsize=12)
        ax1.set_ylabel('Direction 2', fontsize=12)
        ax1.set_title('Loss Landscape (Contour)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 3D surface plot
        ax2 = fig.add_subplot(122, projection='3d')
        X, Y = np.meshgrid(alphas, betas)
        surf = ax2.plot_surface(X, Y, losses.T, cmap='viridis', alpha=0.8)
        ax2.scatter([0], [0], [losses[num_points//2, num_points//2]],
                   color='red', s=200, marker='*', zorder=5)
        ax2.set_xlabel('Direction 1', fontsize=11)
        ax2.set_ylabel('Direction 2', fontsize=11)
        ax2.set_zlabel('Loss', fontsize=11)
        ax2.set_title('Loss Landscape (3D)', fontsize=14, fontweight='bold')
        plt.colorbar(surf, ax=ax2, shrink=0.5)

        plt.tight_layout()
        plt.savefig('loss_landscape.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

# Example usage (commented out - requires actual model and data)
# model = YourModel().cuda()
# loss_fn = nn.CrossEntropyLoss()
# visualizer = LossLandscapeVisualizer(model, loss_fn, train_loader)
# visualizer.visualize_2d(num_points=30, distance=0.5)
```

---

## Attention Mechanism Visualization

Visualize attention weights and patterns in transformers.

### Attention Heatmap Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class AttentionVisualizer:
    """Visualize attention weights in transformers"""

    def plot_attention_head(self, attention_weights, tokens, head_idx=0):
        """
        Plot attention weights for a single head

        Args:
            attention_weights: (num_heads, seq_len, seq_len) attention matrix
            tokens: List of token strings
            head_idx: Which attention head to visualize
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get attention for specific head
        attn = attention_weights[head_idx].cpu().numpy()

        # Create heatmap
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='RdYlBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Attention Weight'},
            ax=ax
        )

        ax.set_title(f'Attention Weights - Head {head_idx}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Key (Attending to)', fontsize=12)
        ax.set_ylabel('Query (Attending from)', fontsize=12)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        return fig

    def plot_all_heads(self, attention_weights, tokens, num_heads=8):
        """Plot attention patterns for all heads"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Multi-Head Attention Patterns',
                    fontsize=16, fontweight='bold')

        for head_idx in range(min(num_heads, 8)):
            ax = axes[head_idx // 4, head_idx % 4]

            attn = attention_weights[head_idx].cpu().numpy()

            sns.heatmap(
                attn,
                xticklabels=tokens if head_idx >= 4 else [],
                yticklabels=tokens if head_idx % 4 == 0 else [],
                cmap='RdYlBu_r',
                center=0,
                square=True,
                cbar=False,
                ax=ax
            )

            ax.set_title(f'Head {head_idx}', fontsize=11, fontweight='bold')

            if head_idx >= 4:
                ax.set_xlabel('Key', fontsize=9)
            if head_idx % 4 == 0:
                ax.set_ylabel('Query', fontsize=9)

        plt.tight_layout()
        return fig

    def plot_attention_flow(self, attention_weights, tokens, layer_names):
        """
        Visualize attention flow across layers

        Args:
            attention_weights: List of (num_heads, seq_len, seq_len) per layer
            tokens: List of tokens
            layer_names: Names of layers
        """
        num_layers = len(attention_weights)
        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 6))

        if num_layers == 1:
            axes = [axes]

        fig.suptitle('Attention Flow Across Transformer Layers',
                    fontsize=16, fontweight='bold')

        for layer_idx, (attn, layer_name) in enumerate(zip(attention_weights, layer_names)):
            # Average across heads
            avg_attn = attn.mean(dim=0).cpu().numpy()

            ax = axes[layer_idx]
            sns.heatmap(
                avg_attn,
                xticklabels=tokens if layer_idx == num_layers-1 else [],
                yticklabels=tokens if layer_idx == 0 else [],
                cmap='viridis',
                square=True,
                cbar=layer_idx == num_layers-1,
                cbar_kws={'label': 'Avg Attention'},
                ax=ax
            )

            ax.set_title(layer_name, fontsize=12, fontweight='bold')

            if layer_idx == 0:
                ax.set_ylabel('Query', fontsize=11)
            if layer_idx == num_layers - 1:
                ax.set_xlabel('Key', fontsize=11)

        plt.tight_layout()
        return fig

# Example: Generate and visualize sample attention
def generate_sample_attention():
    """Generate sample attention weights for demonstration"""
    seq_len = 8
    num_heads = 8

    # Create attention pattern
    attention = torch.zeros(num_heads, seq_len, seq_len)

    for head in range(num_heads):
        if head < 2:
            # Local attention
            for i in range(seq_len):
                for j in range(max(0, i-2), min(seq_len, i+3)):
                    attention[head, i, j] = np.random.rand() * 0.5 + 0.3
        elif head < 4:
            # Global attention to first token (like [CLS])
            attention[head, :, 0] = np.random.rand(seq_len) * 0.5 + 0.3
            attention[head, range(seq_len), range(seq_len)] += 0.3
        else:
            # Random patterns
            attention[head] = torch.rand(seq_len, seq_len)

        # Normalize
        attention[head] = attention[head] / attention[head].sum(dim=-1, keepdim=True)

    return attention

# Visualize
tokens = ['[CLS]', 'The', 'quick', 'brown', 'fox', 'jumps', 'over', '[SEP]']
attention_weights = generate_sample_attention()

viz = AttentionVisualizer()

# Single head
fig1 = viz.plot_attention_head(attention_weights, tokens, head_idx=0)
plt.savefig('attention_single_head.png', dpi=300, bbox_inches='tight')
plt.show()

# All heads
fig2 = viz.plot_all_heads(attention_weights, tokens)
plt.savefig('attention_all_heads.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Embedding Visualization

Visualize high-dimensional embeddings in 2D/3D space.

### t-SNE and UMAP Visualization

```python
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import plotly.graph_objects as go

class EmbeddingVisualizer:
    """Visualize high-dimensional embeddings"""

    def __init__(self, embeddings, labels=None, label_names=None):
        """
        Args:
            embeddings: (n_samples, embedding_dim) numpy array
            labels: (n_samples,) class labels
            label_names: List of class names
        """
        self.embeddings = embeddings
        self.labels = labels
        self.label_names = label_names

    def plot_tsne_2d(self, perplexity=30, n_iter=1000):
        """2D t-SNE visualization"""
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=perplexity,
                   n_iter=n_iter, random_state=42)
        embeddings_2d = tsne.fit_transform(self.embeddings)

        fig, ax = plt.subplots(figsize=(12, 10))

        if self.labels is not None:
            unique_labels = np.unique(self.labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = self.labels == label
                label_name = self.label_names[label] if self.label_names else f'Class {label}'

                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[color],
                    label=label_name,
                    alpha=0.6,
                    s=50,
                    edgecolors='white',
                    linewidth=0.5
                )
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('t-SNE Embedding Visualization',
                    fontsize=14, fontweight='bold')

        if self.labels is not None:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     fontsize=10, framealpha=0.9)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

        return embeddings_2d

    def plot_umap_2d(self, n_neighbors=15, min_dist=0.1):
        """2D UMAP visualization"""
        print("Computing UMAP...")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                          random_state=42)
        embeddings_2d = reducer.fit_transform(self.embeddings)

        fig, ax = plt.subplots(figsize=(12, 10))

        if self.labels is not None:
            unique_labels = np.unique(self.labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = self.labels == label
                label_name = self.label_names[label] if self.label_names else f'Class {label}'

                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[color],
                    label=label_name,
                    alpha=0.6,
                    s=50,
                    edgecolors='white',
                    linewidth=0.5
                )
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      alpha=0.6, s=50)

        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title('UMAP Embedding Visualization',
                    fontsize=14, fontweight='bold')

        if self.labels is not None:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     fontsize=10, framealpha=0.9)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('umap_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

        return embeddings_2d

    def plot_interactive_3d(self, method='umap'):
        """Interactive 3D visualization with Plotly"""
        if method == 'umap':
            print("Computing 3D UMAP...")
            reducer = umap.UMAP(n_components=3, random_state=42)
            embeddings_3d = reducer.fit_transform(self.embeddings)
        else:
            print("Computing 3D t-SNE...")
            tsne = TSNE(n_components=3, random_state=42)
            embeddings_3d = tsne.fit_transform(self.embeddings)

        # Create interactive plot
        if self.labels is not None:
            label_names_list = [self.label_names[l] if self.label_names else f'Class {l}'
                              for l in self.labels]

            fig = px.scatter_3d(
                x=embeddings_3d[:, 0],
                y=embeddings_3d[:, 1],
                z=embeddings_3d[:, 2],
                color=label_names_list,
                labels={'x': 'Dim 1', 'y': 'Dim 2', 'z': 'Dim 3'},
                title=f'3D {method.upper()} Embedding Visualization',
                opacity=0.7
            )
        else:
            fig = px.scatter_3d(
                x=embeddings_3d[:, 0],
                y=embeddings_3d[:, 1],
                z=embeddings_3d[:, 2],
                labels={'x': 'Dim 1', 'y': 'Dim 2', 'z': 'Dim 3'},
                title=f'3D {method.upper()} Embedding Visualization'
            )

        fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color='white')))
        fig.update_layout(
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ),
            width=1000,
            height=800
        )

        fig.write_html(f'{method}_3d_interactive.html')
        fig.show()

        return embeddings_3d

# Example usage
# Generate sample embeddings (e.g., from a trained model)
np.random.seed(42)
n_samples = 1000
n_features = 128
n_classes = 10

# Simulated embeddings with cluster structure
embeddings = []
labels = []

for class_id in range(n_classes):
    # Generate clustered embeddings
    center = np.random.randn(n_features) * 5
    class_embeddings = center + np.random.randn(n_samples // n_classes, n_features) * 0.5
    embeddings.append(class_embeddings)
    labels.extend([class_id] * (n_samples // n_classes))

embeddings = np.vstack(embeddings)
labels = np.array(labels)

class_names = [f'Class {i}' for i in range(n_classes)]

# Visualize
viz = EmbeddingVisualizer(embeddings, labels, class_names)
viz.plot_tsne_2d(perplexity=30)
viz.plot_umap_2d(n_neighbors=15)
viz.plot_interactive_3d(method='umap')
```

---

## Model Interpretation & Explainability

Visualize model decisions and feature importance.

### Grad-CAM Visualization

```python
class GradCAM:
    """Gradient-weighted Class Activation Mapping"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, class_idx=None):
        """
        Generate Class Activation Map

        Args:
            input_image: (1, C, H, W) input image
            class_idx: Target class index (None for predicted class)
        """
        # Forward pass
        output = self.model(input_image)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Generate CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam[0, 0].cpu().numpy(), class_idx

    def visualize(self, input_image, original_image, class_names=None):
        """Create Grad-CAM visualization overlay"""
        cam, pred_class = self.generate_cam(input_image)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Heatmap
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Overlay
        cam_resized = np.array(Image.fromarray(cam).resize(
            original_image.size, Image.BILINEAR
        ))
        axes[2].imshow(original_image, alpha=0.5)
        axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)

        title = f'Grad-CAM Overlay'
        if class_names:
            title += f'\nPredicted: {class_names[pred_class]}'
        axes[2].set_title(title, fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig('gradcam_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

# Example usage (commented - requires actual model)
# from torchvision.models import resnet50
# model = resnet50(pretrained=True).eval()
# target_layer = model.layer4[-1]  # Last conv layer
#
# gradcam = GradCAM(model, target_layer)
# gradcam.visualize(input_tensor, original_pil_image, imagenet_classes)
```

### SHAP Value Visualization

```python
import shap

class SHAPVisualizer:
    """Visualize SHAP values for model interpretability"""

    def __init__(self, model, background_data):
        """
        Args:
            model: Trained model
            background_data: Background dataset for SHAP
        """
        self.model = model
        self.explainer = shap.DeepExplainer(model, background_data)

    def plot_summary(self, test_data, max_display=20):
        """Create SHAP summary plot"""
        shap_values = self.explainer.shap_values(test_data)

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            test_data,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_waterfall(self, instance_idx, test_data):
        """Create SHAP waterfall plot for single prediction"""
        shap_values = self.explainer.shap_values(test_data)

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[instance_idx],
                base_values=self.explainer.expected_value,
                data=test_data[instance_idx]
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}',
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_{instance_idx}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_force(self, instance_idx, test_data, feature_names=None):
        """Create SHAP force plot"""
        shap_values = self.explainer.shap_values(test_data)

        shap.force_plot(
            self.explainer.expected_value,
            shap_values[instance_idx],
            test_data[instance_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )

        plt.title(f'SHAP Force Plot - Instance {instance_idx}',
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'shap_force_{instance_idx}.png', dpi=300, bbox_inches='tight')
        plt.show()
```

---

## Performance Metrics Visualization

Comprehensive visualization of model performance.

### Confusion Matrix & ROC Curves

```python
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import itertools

class PerformanceVisualizer:
    """Visualize model performance metrics"""

    def plot_confusion_matrix(self, y_true, y_pred, class_names, normalize=False):
        """
        Plot confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            normalize: Whether to normalize values
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        # Labels
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True Label',
            xlabel='Predicted Label'
        )

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=10)

        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''),
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def plot_roc_curves(self, y_true, y_score, class_names):
        """
        Plot ROC curves for multi-class classification

        Args:
            y_true: True labels
            y_score: Predicted probabilities (n_samples, n_classes)
            class_names: List of class names
        """
        n_classes = len(class_names)

        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot ROC curve for each class
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            ax.plot(
                fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
            )

        # Plot micro-average
        ax.plot(
            fpr["micro"], tpr["micro"],
            label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
            color='deeppink', linestyle='--', linewidth=3
        )

        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Multi-class ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def plot_precision_recall_curves(self, y_true, y_score, class_names):
        """Plot Precision-Recall curves"""
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        fig, ax = plt.subplots(figsize=(12, 10))

        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

        for i, color in zip(range(n_classes), colors):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_score[:, i]
            )
            avg_precision = auc(recall, precision)

            ax.plot(
                recall, precision, color=color, lw=2,
                label=f'{class_names[i]} (AP = {avg_precision:.2f})'
            )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

# Example usage
np.random.seed(42)
n_samples = 1000
n_classes = 5

# Simulate predictions
y_true = np.random.randint(0, n_classes, n_samples)
y_score = np.random.rand(n_samples, n_classes)
y_score = y_score / y_score.sum(axis=1, keepdims=True)  # Normalize
y_pred = y_score.argmax(axis=1)

class_names = [f'Class {i}' for i in range(n_classes)]

viz = PerformanceVisualizer()
viz.plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
viz.plot_roc_curves(y_true, y_score, class_names)
viz.plot_precision_recall_curves(y_true, y_score, class_names)
```

---

## Interactive Dashboards

Create interactive monitoring dashboards with Plotly Dash.

### Real-time Training Dashboard

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd

class InteractiveDashboard:
    """Create interactive training dashboard"""

    def __init__(self, port=8050):
        self.app = dash.Dash(__name__)
        self.port = port

        # Sample data structure
        self.training_data = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }

    def create_layout(self):
        """Create dashboard layout"""
        self.app.layout = html.Div([
            html.H1('ML Training Dashboard',
                   style={'textAlign': 'center', 'color': '#2c3e50'}),

            html.Div([
                html.Div([
                    html.H3('Training Metrics'),
                    dcc.Graph(id='loss-graph'),
                ], className='six columns'),

                html.Div([
                    html.H3('Accuracy Metrics'),
                    dcc.Graph(id='accuracy-graph'),
                ], className='six columns'),
            ], className='row'),

            html.Div([
                html.Div([
                    html.H3('Learning Rate'),
                    dcc.Graph(id='lr-graph'),
                ], className='six columns'),

                html.Div([
                    html.H3('Model Statistics'),
                    html.Div(id='stats-div'),
                ], className='six columns'),
            ], className='row'),

            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every second
                n_intervals=0
            )
        ])

    def create_plots(self):
        """Create plotly figures"""

        # Loss plot
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(
            x=self.training_data['epoch'],
            y=self.training_data['train_loss'],
            mode='lines+markers',
            name='Train Loss',
            line=dict(color='blue', width=2)
        ))
        loss_fig.add_trace(go.Scatter(
            x=self.training_data['epoch'],
            y=self.training_data['val_loss'],
            mode='lines+markers',
            name='Val Loss',
            line=dict(color='red', width=2)
        ))
        loss_fig.update_layout(
            title='Loss Over Time',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='x unified'
        )

        # Accuracy plot
        acc_fig = go.Figure()
        acc_fig.add_trace(go.Scatter(
            x=self.training_data['epoch'],
            y=self.training_data['train_acc'],
            mode='lines+markers',
            name='Train Acc',
            line=dict(color='green', width=2)
        ))
        acc_fig.add_trace(go.Scatter(
            x=self.training_data['epoch'],
            y=self.training_data['val_acc'],
            mode='lines+markers',
            name='Val Acc',
            line=dict(color='orange', width=2)
        ))
        acc_fig.update_layout(
            title='Accuracy Over Time',
            xaxis_title='Epoch',
            yaxis_title='Accuracy (%)',
            hovermode='x unified'
        )

        # Learning rate plot
        lr_fig = go.Figure()
        lr_fig.add_trace(go.Scatter(
            x=self.training_data['epoch'],
            y=self.training_data['learning_rate'],
            mode='lines+markers',
            line=dict(color='purple', width=2)
        ))
        lr_fig.update_layout(
            title='Learning Rate Schedule',
            xaxis_title='Epoch',
            yaxis_title='Learning Rate',
            yaxis_type='log'
        )

        return loss_fig, acc_fig, lr_fig

    def run(self):
        """Run the dashboard"""
        self.create_layout()

        @self.app.callback(
            [Output('loss-graph', 'figure'),
             Output('accuracy-graph', 'figure'),
             Output('lr-graph', 'figure'),
             Output('stats-div', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            # Simulate new data (replace with actual training data)
            if len(self.training_data['epoch']) < 100:
                epoch = len(self.training_data['epoch'])
                self.training_data['epoch'].append(epoch)
                self.training_data['train_loss'].append(
                    2.0 * np.exp(-epoch/20) + 0.1 + np.random.rand()*0.05
                )
                self.training_data['val_loss'].append(
                    self.training_data['train_loss'][-1] + 0.2 + np.random.rand()*0.1
                )
                self.training_data['train_acc'].append(
                    (1 - np.exp(-epoch/15)) * 95 + np.random.rand()*2
                )
                self.training_data['val_acc'].append(
                    self.training_data['train_acc'][-1] - 5 - np.random.rand()*3
                )
                self.training_data['learning_rate'].append(
                    0.001 * (0.95 ** (epoch // 10))
                )

            loss_fig, acc_fig, lr_fig = self.create_plots()

            # Statistics
            stats = html.Div([
                html.P(f"Current Epoch: {self.training_data['epoch'][-1] if self.training_data['epoch'] else 0}",
                      style={'fontSize': 16}),
                html.P(f"Best Val Loss: {min(self.training_data['val_loss']) if self.training_data['val_loss'] else 0:.4f}",
                      style={'fontSize': 16}),
                html.P(f"Best Val Acc: {max(self.training_data['val_acc']) if self.training_data['val_acc'] else 0:.2f}%",
                      style={'fontSize': 16}),
                html.P(f"Current LR: {self.training_data['learning_rate'][-1] if self.training_data['learning_rate'] else 0:.6f}",
                      style={'fontSize': 16}),
            ])

            return loss_fig, acc_fig, lr_fig, stats

        self.app.run_server(debug=True, port=self.port)

# Usage (run this to start the dashboard)
# dashboard = InteractiveDashboard(port=8050)
# dashboard.run()
# Then open browser to http://localhost:8050
```

---

## Summary

This visualization guide provides:

1. **Neural Network Architecture** - Visualize CNN, Transformer, and custom architectures
2. **Training Dynamics** - Monitor loss, accuracy, learning rate, gradient norms
3. **Attention Mechanisms** - Heatmaps and flow diagrams for transformer attention
4. **Embeddings** - t-SNE, UMAP, interactive 3D visualizations
5. **Model Interpretation** - Grad-CAM, SHAP values, feature importance
6. **Performance Metrics** - Confusion matrices, ROC curves, PR curves
7. **Interactive Dashboards** - Real-time monitoring with Plotly Dash

**Key Tools Used:**
- Matplotlib/Seaborn for static plots
- Plotly for interactive visualizations
- Dash for real-time dashboards
- scikit-learn for metrics
- SHAP for interpretability

All visualizations are production-ready with high-resolution output and can be integrated into research papers, presentations, and monitoring systems.