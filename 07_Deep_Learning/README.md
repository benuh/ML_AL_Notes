# Module 07: Deep Learning - Neural Networks and Beyond

## 🎯 Learning Objectives
By the end of this module, you will:
- Understand the mathematical foundations of neural networks
- Master backpropagation and gradient descent optimization
- Build and train deep networks from scratch and with frameworks
- Implement CNNs, RNNs, and modern architectures like Transformers
- Apply deep learning to real-world problems

## 🧠 Introduction to Deep Learning

*Source: "Deep Learning" - Goodfellow, Bengio, Courville, Chapter 1*
> "Deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction."

### Why Deep Learning?
- **Representation Learning**: Automatic feature discovery
- **Universal Approximation**: Can approximate any continuous function
- **End-to-End Learning**: Joint optimization of entire pipeline
- **Scalability**: Performance improves with more data and compute

### Historical Context
- **1943**: McCulloch-Pitts neuron model
- **1957**: Perceptron (Rosenblatt)
- **1986**: Backpropagation popularized (Rumelhart et al.)
- **2006**: Deep learning renaissance (Hinton et al.)
- **2012**: AlexNet breakthrough in computer vision

## 🔄 Neural Network Fundamentals

### 7.1 The Perceptron and Multi-Layer Networks

*Source: "Neural Networks and Deep Learning" - Michael Nielsen, Chapter 1*

```python
"""
Neural Networks from Scratch: Understanding the Fundamentals
==========================================================

We'll build neural networks from the ground up to understand:
- Forward propagation
- Backpropagation algorithm
- Gradient descent optimization
- Activation functions
- Loss functions

Mathematical Foundation:
- Forward pass: a⁽ˡ⁺¹⁾ = σ(W⁽ˡ⁺¹⁾a⁽ˡ⁾ + b⁽ˡ⁺¹⁾)
- Backprop: ∂C/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
- Weight update: W := W - η∇C
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_circles, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NeuralNetworkFromScratch:
    """
    Complete neural network implementation from scratch

    Source: "Deep Learning" - Goodfellow et al., Chapter 6
    """

    def __init__(self, layers, learning_rate=0.01, epochs=1000, random_state=42):
        """
        Initialize neural network

        Parameters:
        -----------
        layers : list
            List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
        learning_rate : float
            Learning rate for gradient descent
        epochs : int
            Number of training epochs
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        # Initialize weights and biases
        np.random.seed(random_state)
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            # Xavier/Glorot initialization
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / (layers[i] + layers[i+1]))
            b = np.zeros((1, layers[i+1]))

            self.weights.append(w)
            self.biases.append(b)

        self.cost_history = []

    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)

    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """Derivative of ReLU function"""
        return (z > 0).astype(float)

    def softmax(self, z):
        """Softmax activation function for output layer"""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward_propagation(self, X):
        """
        Forward propagation through the network

        Returns:
        --------
        activations : list
            Activations for each layer
        z_values : list
            Pre-activation values for each layer
        """
        activations = [X]
        z_values = []

        current_input = X

        for i in range(len(self.weights)):
            # Linear transformation
            z = current_input @ self.weights[i] + self.biases[i]
            z_values.append(z)

            # Activation function
            if i == len(self.weights) - 1:  # Output layer
                if self.layers[-1] == 1:  # Binary classification
                    a = self.sigmoid(z)
                else:  # Multi-class classification
                    a = self.softmax(z)
            else:  # Hidden layers
                a = self.relu(z)

            activations.append(a)
            current_input = a

        return activations, z_values

    def backward_propagation(self, X, y, activations, z_values):
        """
        Backward propagation algorithm

        Source: "Neural Networks and Deep Learning" - Nielsen, Chapter 2
        """
        m = X.shape[0]  # Number of samples
        n_layers = len(self.layers)

        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        if self.layers[-1] == 1:  # Binary classification
            delta = activations[-1] - y.reshape(-1, 1)
        else:  # Multi-class classification
            # Convert labels to one-hot encoding
            y_one_hot = np.eye(self.layers[-1])[y]
            delta = activations[-1] - y_one_hot

        # Gradients for output layer
        dW[-1] = activations[-2].T @ delta / m
        db[-1] = np.mean(delta, axis=0, keepdims=True)

        # Backpropagate through hidden layers
        for i in range(n_layers - 3, -1, -1):  # Work backwards
            # Error for current layer
            delta = (delta @ self.weights[i+1].T) * self.relu_derivative(z_values[i])

            # Gradients for current layer
            dW[i] = activations[i].T @ delta / m
            db[i] = np.mean(delta, axis=0, keepdims=True)

        return dW, db

    def compute_cost(self, y_true, y_pred):
        """Compute cross-entropy cost"""
        m = y_true.shape[0]

        if self.layers[-1] == 1:  # Binary classification
            cost = -np.mean(y_true * np.log(y_pred + 1e-8) +
                           (1 - y_true) * np.log(1 - y_pred + 1e-8))
        else:  # Multi-class classification
            y_one_hot = np.eye(self.layers[-1])[y_true]
            cost = -np.mean(np.sum(y_one_hot * np.log(y_pred + 1e-8), axis=1))

        return cost

    def fit(self, X, y):
        """Train the neural network"""
        for epoch in range(self.epochs):
            # Forward propagation
            activations, z_values = self.forward_propagation(X)

            # Compute cost
            cost = self.compute_cost(y, activations[-1])
            self.cost_history.append(cost)

            # Backward propagation
            dW, db = self.backward_propagation(X, y, activations, z_values)

            # Update weights and biases
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * db[i]

            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")

    def predict_proba(self, X):
        """Predict probabilities"""
        activations, _ = self.forward_propagation(X)
        return activations[-1]

    def predict(self, X):
        """Make predictions"""
        probabilities = self.predict_proba(X)

        if self.layers[-1] == 1:  # Binary classification
            return (probabilities > 0.5).astype(int).flatten()
        else:  # Multi-class classification
            return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

def demonstrate_neural_network_basics():
    """Demonstrate neural network fundamentals"""

    print("🧠 Neural Network Fundamentals")
    print("=" * 50)

    # Generate datasets
    # 1. Binary classification
    X_binary, y_binary = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)

    # 2. Multi-class classification
    X_multi, y_multi = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                                          n_informative=2, n_clusters_per_class=1,
                                          n_classes=3, random_state=42)

    datasets = [
        ("Binary Classification", X_binary, y_binary, [2, 10, 5, 1]),
        ("Multi-class Classification", X_multi, y_multi, [2, 15, 10, 3])
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🧠 Neural Network Fundamentals', fontsize=16, fontweight='bold')

    for idx, (name, X, y, architecture) in enumerate(datasets):
        print(f"\\n{name}")
        print("-" * 40)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train neural network
        nn = NeuralNetworkFromScratch(
            layers=architecture,
            learning_rate=0.1,
            epochs=1000,
            random_state=42
        )

        nn.fit(X_train_scaled, y_train)

        # Evaluate
        train_accuracy = nn.score(X_train_scaled, y_train)
        test_accuracy = nn.score(X_test_scaled, y_test)

        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Visualizations
        row = idx

        # 1. Data distribution
        scatter = axes[row, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        axes[row, 0].set_title(f'{name}\\nData Distribution')
        axes[row, 0].set_xlabel('Feature 1')
        axes[row, 0].set_ylabel('Feature 2')

        # 2. Learning curve
        axes[row, 1].plot(nn.cost_history, linewidth=2)
        axes[row, 1].set_title('Learning Curve')
        axes[row, 1].set_xlabel('Epoch')
        axes[row, 1].set_ylabel('Cost')
        axes[row, 1].set_yscale('log')
        axes[row, 1].grid(True, alpha=0.3)

        # 3. Decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = scaler.transform(mesh_points)

        if architecture[-1] == 1:  # Binary classification
            Z = nn.predict_proba(mesh_points_scaled).flatten()
        else:  # Multi-class classification
            Z = nn.predict(mesh_points_scaled)

        Z = Z.reshape(xx.shape)

        axes[row, 2].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        axes[row, 2].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        axes[row, 2].set_title('Decision Boundary')
        axes[row, 2].set_xlabel('Feature 1')
        axes[row, 2].set_ylabel('Feature 2')

    plt.tight_layout()
    plt.show()

    return nn

# Run neural network demonstration
nn_model = demonstrate_neural_network_basics()
```

### 7.2 Activation Functions and Their Properties

*Source: "Deep Learning" - Goodfellow et al., Chapter 6*

```python
def explore_activation_functions():
    """
    Comprehensive exploration of activation functions

    Activation functions introduce non-linearity and affect:
    - Gradient flow (vanishing/exploding gradients)
    - Convergence speed
    - Model expressiveness
    """

    print("\\n🔥 Activation Functions Analysis")
    print("-" * 40)

    # Define activation functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(x):
        return np.tanh(x)

    def relu(x):
        return np.maximum(0, x)

    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def swish(x):
        return x * sigmoid(x)

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    # Define derivatives
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)

    def tanh_derivative(x):
        return 1 - np.tanh(x)**2

    def relu_derivative(x):
        return (x > 0).astype(float)

    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def elu_derivative(x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))

    activations = {
        'Sigmoid': (sigmoid, sigmoid_derivative),
        'Tanh': (tanh, tanh_derivative),
        'ReLU': (relu, relu_derivative),
        'Leaky ReLU': (leaky_relu, leaky_relu_derivative),
        'ELU': (elu, elu_derivative),
        'Swish': (swish, None),  # Complex derivative
        'GELU': (gelu, None)     # Complex derivative
    }

    # Plot activation functions and their derivatives
    x = np.linspace(-3, 3, 1000)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('🔥 Activation Functions and Their Derivatives', fontsize=16, fontweight='bold')

    colors = plt.cm.Set1(np.linspace(0, 1, len(activations)))

    for i, (name, (func, derivative)) in enumerate(activations.items()):
        row = i // 4
        col = i % 4

        if i >= len(axes.flat):
            break

        ax = axes[row, col]

        # Plot activation function
        y = func(x)
        ax.plot(x, y, linewidth=2, label=f'{name}', color=colors[i])

        # Plot derivative if available
        if derivative is not None:
            dy = derivative(x)
            ax.plot(x, dy, '--', linewidth=2, alpha=0.7, label=f"{name} derivative")

        ax.set_title(f'{name} Activation')
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5)

    # Remove empty subplots
    for i in range(len(activations), len(axes.flat)):
        fig.delaxes(axes.flat[i])

    plt.tight_layout()
    plt.show()

    # Demonstrate vanishing gradient problem
    demonstrate_vanishing_gradients()

def demonstrate_vanishing_gradients():
    """Demonstrate the vanishing gradient problem"""

    print("\\n⚠️ Vanishing Gradient Problem")
    print("-" * 40)

    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)

    def relu_derivative(x):
        return (x > 0).astype(float)

    # Simulate gradient flow through deep network
    n_layers = 10
    layer_inputs = np.linspace(-3, 3, 100)

    # For sigmoid activation
    sigmoid_gradients = []
    current_gradient = 1.0

    for layer in range(n_layers):
        layer_gradient = sigmoid_derivative(layer_inputs).mean()
        current_gradient *= layer_gradient
        sigmoid_gradients.append(current_gradient)

    # For ReLU activation (assuming 50% neurons active)
    relu_gradients = []
    current_gradient = 1.0

    for layer in range(n_layers):
        layer_gradient = 0.5  # 50% of neurons active
        current_gradient *= layer_gradient
        relu_gradients.append(current_gradient)

    # Plot gradient flow
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('⚠️ Vanishing Gradient Problem', fontsize=16, fontweight='bold')

    layers = range(1, n_layers + 1)

    axes[0].semilogy(layers, sigmoid_gradients, 'o-', label='Sigmoid', linewidth=2, color='red')
    axes[0].semilogy(layers, relu_gradients, 's-', label='ReLU', linewidth=2, color='blue')
    axes[0].set_xlabel('Layer Depth')
    axes[0].set_ylabel('Gradient Magnitude (log scale)')
    axes[0].set_title('Gradient Flow Through Deep Network')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Show gradient distribution
    x = np.linspace(-3, 3, 1000)
    sigmoid_grad = sigmoid_derivative(x)
    relu_grad = relu_derivative(x)

    axes[1].plot(x, sigmoid_grad, label='Sigmoid Derivative', linewidth=2, color='red')
    axes[1].plot(x, relu_grad, label='ReLU Derivative', linewidth=2, color='blue')
    axes[1].set_xlabel('Input Value')
    axes[1].set_ylabel('Derivative Value')
    axes[1].set_title('Activation Function Derivatives')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("💡 Key Insights:")
    print("• Sigmoid derivatives are small (max 0.25), causing vanishing gradients")
    print("• ReLU derivatives are either 0 or 1, avoiding vanishing gradients")
    print("• Modern activations (Swish, GELU) balance smooth gradients with non-linearity")

# Run activation function exploration
explore_activation_functions()
```

## 🖼️ Convolutional Neural Networks

### 7.3 CNNs for Computer Vision

*Source: "Deep Learning" - Goodfellow et al., Chapter 9*

```python
"""
Convolutional Neural Networks: Revolutionizing Computer Vision
============================================================

CNNs use:
- Local connectivity (receptive fields)
- Parameter sharing (translation invariance)
- Hierarchical feature learning
- Pooling for translation invariance

Key Concepts:
- Convolution operation: (f * g)[n] = Σ f[m]g[n-m]
- Feature maps: Different filters detect different patterns
- Pooling: Reduces spatial dimensions, adds translation invariance
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def build_cnn_from_scratch():
    """Build and train a CNN from scratch using TensorFlow/Keras"""

    print("\\n🖼️ Convolutional Neural Networks")
    print("-" * 40)

    # Load and preprocess CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    # Visualize sample images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('🖼️ CIFAR-10 Sample Images', fontsize=16, fontweight='bold')

    for i in range(10):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(x_train[i])
        axes[row, col].set_title(f'{class_names[np.argmax(y_train[i])]}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

    # Build CNN architecture
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3),
                     padding='same', name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25, name='dropout1'),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.BatchNormalization(name='bn2'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25, name='dropout2'),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5'),
        layers.BatchNormalization(name='bn3'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv6'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.25, name='dropout3'),

        # Classifier
        layers.Flatten(name='flatten'),
        layers.Dense(512, activation='relu', name='fc1'),
        layers.BatchNormalization(name='bn4'),
        layers.Dropout(0.5, name='dropout4'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])

    # Print model architecture
    print("\\n🏗️ CNN Architecture:")
    model.summary()

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model (small subset for demonstration)
    print("\\n🎓 Training CNN...")

    # Use subset for faster training in demo
    train_subset = 5000
    val_subset = 1000

    history = model.fit(
        x_train[:train_subset], y_train[:train_subset],
        batch_size=32,
        epochs=10,
        validation_data=(x_test[:val_subset], y_test[:val_subset]),
        verbose=1
    )

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\\nTest Accuracy: {test_accuracy:.4f}")

    return model, history, x_test, y_test, class_names

def visualize_cnn_features(model, x_test, class_names):
    """Visualize CNN feature maps and filters"""

    print("\\n🔍 CNN Feature Visualization")
    print("-" * 40)

    # Get intermediate layer outputs
    layer_outputs = [layer.output for layer in model.layers[:8]]  # First few layers
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

    # Choose a test image
    test_image = x_test[0:1]  # First test image
    activations = activation_model.predict(test_image)

    # Visualize feature maps
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('🔍 CNN Feature Maps Visualization', fontsize=16, fontweight='bold')

    layer_names = ['conv1', 'bn1', 'conv2', 'pool1', 'conv3', 'bn2', 'conv4', 'pool2']

    for i, (activation, layer_name) in enumerate(zip(activations, layer_names)):
        if i >= 8:
            break

        row = i // 4
        col = i % 4

        # Show first feature map
        if len(activation.shape) == 4:  # Convolutional layer
            feature_map = activation[0, :, :, 0]  # First channel
        else:  # Other layers
            continue

        im = axes[row, col].imshow(feature_map, cmap='viridis')
        axes[row, col].set_title(f'{layer_name}\\nShape: {activation.shape[1:]}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # Visualize filters
    visualize_filters(model)

def visualize_filters(model):
    """Visualize learned convolutional filters"""

    # Get first convolutional layer
    conv_layer = model.layers[0]
    filters, biases = conv_layer.get_weights()

    # Normalize filters for visualization
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    # Plot first 16 filters
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('🎨 Learned Convolutional Filters (First Layer)', fontsize=16, fontweight='bold')

    for i in range(16):
        row = i // 4
        col = i % 4

        # Get filter (3x3x3 for RGB)
        f = filters[:, :, :, i]

        # Convert to displayable format
        if f.shape[2] == 3:  # RGB
            axes[row, col].imshow(f)
        else:  # Grayscale
            axes[row, col].imshow(f[:, :, 0], cmap='gray')

        axes[row, col].set_title(f'Filter {i+1}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

def analyze_cnn_performance(history):
    """Analyze CNN training performance"""

    print("\\n📊 CNN Training Analysis")
    print("-" * 40)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('📊 CNN Training Progress', fontsize=16, fontweight='bold')

    # Plot training history
    epochs = range(1, len(history.history['loss']) + 1)

    # Loss
    axes[0].plot(epochs, history.history['loss'], 'bo-', label='Training Loss')
    axes[0].plot(epochs, history.history['val_loss'], 'ro-', label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history.history['accuracy'], 'bo-', label='Training Accuracy')
    axes[1].plot(epochs, history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Run CNN demonstration
print("🖼️ Building CNN for CIFAR-10 Classification...")
cnn_model, cnn_history, x_test, y_test, class_names = build_cnn_from_scratch()
visualize_cnn_features(cnn_model, x_test, class_names)
analyze_cnn_performance(cnn_history)
```

## 🔄 Recurrent Neural Networks

### 7.4 RNNs for Sequential Data

*Source: "Deep Learning" - Goodfellow et al., Chapter 10*

```python
"""
Recurrent Neural Networks: Processing Sequential Data
===================================================

RNNs process sequences by maintaining hidden state:
- h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
- y_t = W_hy * h_t + b_y

Challenges:
- Vanishing gradients in long sequences
- Difficulty learning long-term dependencies

Solutions:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
"""

def demonstrate_rnn_concepts():
    """Demonstrate RNN concepts with time series prediction"""

    print("\\n🔄 Recurrent Neural Networks")
    print("-" * 40)

    # Generate synthetic time series data
    def generate_time_series(n_samples=1000, n_steps=50):
        """Generate synthetic sine wave with noise"""
        t = np.linspace(0, 100, n_samples + n_steps)

        # Multiple sine waves with different frequencies
        series = (np.sin(0.1 * t) +
                 0.5 * np.sin(0.3 * t) +
                 0.3 * np.sin(0.05 * t) +
                 0.1 * np.random.randn(len(t)))

        return series

    # Create sequences for RNN training
    def create_sequences(data, n_steps):
        """Create input-output sequences for RNN"""
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:(i + n_steps)])
            y.append(data[i + n_steps])
        return np.array(X), np.array(y)

    # Generate data
    n_steps = 20
    series = generate_time_series()
    X, y = create_sequences(series, n_steps)

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Reshape for RNN (samples, time_steps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Build different RNN architectures
    models = {}

    # Simple RNN
    models['Simple RNN'] = keras.Sequential([
        layers.SimpleRNN(50, return_sequences=True, input_shape=(n_steps, 1)),
        layers.Dropout(0.2),
        layers.SimpleRNN(50),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    # LSTM
    models['LSTM'] = keras.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=(n_steps, 1)),
        layers.Dropout(0.2),
        layers.LSTM(50),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    # GRU
    models['GRU'] = keras.Sequential([
        layers.GRU(50, return_sequences=True, input_shape=(n_steps, 1)),
        layers.Dropout(0.2),
        layers.GRU(50),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    # Train and evaluate models
    results = {}

    for name, model in models.items():
        print(f"\\nTraining {name}...")

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        # Evaluate
        train_loss = model.evaluate(X_train, y_train, verbose=0)[0]
        test_loss = model.evaluate(X_test, y_test, verbose=0)[0]

        # Make predictions
        y_pred = model.predict(X_test, verbose=0)

        results[name] = {
            'model': model,
            'history': history,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'predictions': y_pred
        }

        print(f"  Train MSE: {train_loss:.4f}")
        print(f"  Test MSE: {test_loss:.4f}")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔄 RNN Comparison for Time Series Prediction', fontsize=16, fontweight='bold')

    # 1. Original time series
    axes[0, 0].plot(series[:200], linewidth=2)
    axes[0, 0].set_title('Original Time Series')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Model comparison
    model_names = list(results.keys())
    train_losses = [results[name]['train_loss'] for name in model_names]
    test_losses = [results[name]['test_loss'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    axes[0, 1].bar(x - width/2, train_losses, width, label='Train MSE', alpha=0.7)
    axes[0, 1].bar(x + width/2, test_losses, width, label='Test MSE', alpha=0.7)
    axes[0, 1].set_title('Model Performance Comparison')
    axes[0, 1].set_ylabel('Mean Squared Error')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Predictions vs actual
    test_range = range(len(y_test[:100]))
    axes[1, 0].plot(test_range, y_test[:100], 'k-', label='Actual', linewidth=2)

    colors = ['blue', 'red', 'green']
    for i, (name, result) in enumerate(results.items()):
        axes[1, 0].plot(test_range, result['predictions'][:100], '--',
                       color=colors[i], label=f'{name} Prediction', linewidth=2, alpha=0.7)

    axes[1, 0].set_title('Predictions vs Actual (First 100 test points)')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Training loss curves
    for i, (name, result) in enumerate(results.items()):
        epochs = range(1, len(result['history'].history['loss']) + 1)
        axes[1, 1].plot(epochs, result['history'].history['loss'],
                       color=colors[i], label=f'{name}', linewidth=2)

    axes[1, 1].set_title('Training Loss Curves')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.show()

    return results

# Run RNN demonstration
rnn_results = demonstrate_rnn_concepts()
```

## 🔄 Modern Architectures: Attention and Transformers

### 7.5 Attention Mechanisms and Transformers

*Source: "Attention Is All You Need" - Vaswani et al. (2017)*

```python
"""
Attention Mechanisms and Transformers
====================================

Attention allows models to focus on relevant parts of input:
- Self-attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Multi-head attention: Parallel attention mechanisms
- Positional encoding: Add position information to embeddings

Transformers revolutionized NLP and beyond:
- No recurrence needed
- Parallel processing
- Long-range dependencies
- Transfer learning (BERT, GPT)
"""

def demonstrate_attention_mechanism():
    """Demonstrate simplified attention mechanism"""

    print("\\n🔍 Attention Mechanism Visualization")
    print("-" * 40)

    # Simplified attention example
    def scaled_dot_product_attention(Q, K, V, mask=None):
        """
        Scaled dot-product attention

        Args:
            Q: Query matrix
            K: Key matrix
            V: Value matrix
            mask: Optional mask
        """
        d_k = Q.shape[-1]

        # Compute attention scores
        scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)

        if mask is not None:
            scores += (mask * -1e9)

        # Apply softmax
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

        # Apply attention to values
        output = np.matmul(attention_weights, V)

        return output, attention_weights

    # Example: Simple sequence
    sequence_length = 5
    d_model = 4

    # Create example embeddings
    np.random.seed(42)
    embeddings = np.random.randn(sequence_length, d_model)

    # For self-attention, Q, K, V are all derived from the same embeddings
    Q = embeddings
    K = embeddings
    V = embeddings

    # Compute attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    # Visualize attention weights
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('🔍 Attention Mechanism Visualization', fontsize=16, fontweight='bold')

    # Attention heatmap
    im = axes[0].imshow(attention_weights, cmap='Blues')
    axes[0].set_title('Attention Weights Matrix')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')

    # Add text annotations
    for i in range(sequence_length):
        for j in range(sequence_length):
            text = axes[0].text(j, i, f'{attention_weights[i, j]:.2f}',
                              ha="center", va="center", color="black")

    plt.colorbar(im, ax=axes[0])

    # Show how attention weights sum to 1
    attention_sums = np.sum(attention_weights, axis=1)
    axes[1].bar(range(sequence_length), attention_sums)
    axes[1].set_title('Attention Weight Sums (Should be 1.0)')
    axes[1].set_xlabel('Query Position')
    axes[1].set_ylabel('Sum of Attention Weights')
    axes[1].set_ylim(0, 1.2)

    # Add value labels
    for i, sum_val in enumerate(attention_sums):
        axes[1].text(i, sum_val + 0.05, f'{sum_val:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    print("💡 Key Insights:")
    print("• Each query attends to all keys with different weights")
    print("• Attention weights sum to 1 (probability distribution)")
    print("• Self-attention allows tokens to attend to each other")
    print("• Multi-head attention captures different types of relationships")

    return attention_weights

def build_transformer_classifier():
    """Build a simplified Transformer for classification"""

    print("\\n🤖 Transformer Architecture")
    print("-" * 40)

    # Custom Transformer block
    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = keras.Sequential([
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ])
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(rate)
            self.dropout2 = layers.Dropout(rate)

        def call(self, inputs, training):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    # Positional encoding
    class PositionalEncoding(layers.Layer):
        def __init__(self, max_len, d_model):
            super(PositionalEncoding, self).__init__()
            self.pos_encoding = self.positional_encoding(max_len, d_model)

        def get_angles(self, pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
            return pos * angle_rates

        def positional_encoding(self, max_len, d_model):
            angle_rads = self.get_angles(np.arange(max_len)[:, np.newaxis],
                                       np.arange(d_model)[np.newaxis, :],
                                       d_model)

            # Apply sin to even indices
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

            # Apply cos to odd indices
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

            pos_encoding = angle_rads[np.newaxis, ...]

            return tf.cast(pos_encoding, dtype=tf.float32)

        def call(self, x):
            return x + self.pos_encoding[:, :tf.shape(x)[1], :]

    # Build Transformer model for IMDB sentiment classification
    print("Building Transformer for sentiment analysis...")

    # Load IMDB dataset (subset for demo)
    max_features = 10000
    maxlen = 200

    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)

    # Take subset for faster training
    x_train = x_train[:5000]
    y_train = y_train[:5000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    # Pad sequences
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    # Model parameters
    embed_dim = 64
    num_heads = 4
    ff_dim = 64

    # Build model
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = layers.Embedding(max_features, embed_dim)
    x = embedding_layer(inputs)

    # Add positional encoding
    pos_encoding = PositionalEncoding(maxlen, embed_dim)
    x = pos_encoding(x)

    # Transformer block
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)

    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    # Compile and train
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=5,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return model, history

# Run attention and transformer demonstrations
attention_weights = demonstrate_attention_mechanism()
# transformer_model, transformer_history = build_transformer_classifier()

print("\\n🎯 Deep Learning Summary:")
print("=" * 50)
print("🔸 **Neural Networks**: Universal function approximators")
print("🔸 **CNNs**: Revolutionized computer vision with local connectivity")
print("🔸 **RNNs/LSTMs**: Handle sequential data and temporal dependencies")
print("🔸 **Transformers**: State-of-the-art for language and beyond")
print("🔸 **Attention**: Allows models to focus on relevant information")

print("\\n💡 Key Deep Learning Principles:")
print("• Representation learning: Learn features automatically")
print("• End-to-end training: Optimize entire pipeline jointly")
print("• Transfer learning: Leverage pre-trained models")
print("• Regularization: Prevent overfitting in large models")
print("• Architecture design: Match model to problem structure")
```

## 📚 Sources & Further Reading

### 📖 **Essential Deep Learning Textbooks**:

1. **"Deep Learning"** - Goodfellow, Bengio, Courville
   - **🆓 Free HTML**: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
   - **📑 Key Chapters**:
     - Chapter 6: Deep Feedforward Networks *(MLPs, backpropagation)*
     - Chapter 9: Convolutional Networks *(CNN architectures)*
     - Chapter 10: Sequence Modeling *(RNNs, LSTMs)*
     - Chapter 12: Applications *(computer vision, NLP)*

2. **"Neural Networks and Deep Learning"** - Michael Nielsen
   - **🆓 Free Online**: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)
   - **💡 Best For**: Intuitive explanations with visual aids
   - **📑 Key Chapters**:
     - Chapter 1: Neural Networks *(perceptrons, basic concepts)*
     - Chapter 2: Backpropagation *(mathematical derivation)*

### 📑 **Foundational Papers**:

1. **"Attention Is All You Need"** - Vaswani et al. (2017)
   - **🔗 PDF**: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
   - **💡 Impact**: Introduced Transformer architecture

2. **"ImageNet Classification with Deep CNNs"** - Krizhevsky et al. (2012)
   - **💡 Impact**: AlexNet sparked deep learning revolution

3. **"Long Short-Term Memory"** - Hochreiter & Schmidhuber (1997)
   - **💡 Impact**: Solved vanishing gradient problem in RNNs

## ✅ Hands-On Projects

### Project 1: Build CNN from Scratch
Implement convolutional layers, pooling, and backpropagation without frameworks.

### Project 2: LSTM Text Generation
Train an LSTM to generate text in the style of a specific author.

### Project 3: Attention Visualization
Build a tool to visualize attention weights in Transformer models.

## 🚀 Next Steps

Ready for cutting-edge ML? Continue to [Module 08: Advanced Topics](../08_Advanced_Topics/README.md) for reinforcement learning, generative models, and more!

---
*Estimated completion time: 12-15 hours*
*Prerequisites: Strong programming background, calculus knowledge recommended*