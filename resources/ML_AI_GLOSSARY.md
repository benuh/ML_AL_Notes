# 📖 Complete ML/AI Glossary

## Your comprehensive reference for Machine Learning and AI terminology

---

## A

### Activation Function
Function applied to neuron output to introduce non-linearity.
- **ReLU**: f(x) = max(0, x)
- **Sigmoid**: f(x) = 1/(1+e^(-x))
- **Tanh**: f(x) = (e^x - e^(-x))/(e^x + e^(-x))

### Attention Mechanism
Allows model to focus on relevant parts of input. Key component of Transformers.
- **Self-Attention**: Relates different positions of single sequence
- **Cross-Attention**: Relates two different sequences
- **Multi-Head**: Multiple attention mechanisms in parallel

### Autoencoder
Neural network that learns to compress (encode) and reconstruct (decode) data.
- Used for: Dimensionality reduction, anomaly detection, denoising

### AUC-ROC
Area Under the Receiver Operating Characteristic curve. Measures classifier performance.
- Range: 0-1
- Good model: AUC > 0.8
- Random model: AUC = 0.5

---

## B

### Backpropagation
Algorithm for training neural networks by propagating errors backward through layers.
- Uses chain rule to compute gradients
- Foundation of deep learning

### Batch Normalization
Normalizes layer inputs to improve training stability and speed.
- Reduces internal covariate shift
- Acts as regularization

### BERT (Bidirectional Encoder Representations from Transformers)
Pre-trained language model from Google.
- Reads text bidirectionally
- Excellent for: Classification, NER, QA

### Bias-Variance Tradeoff
Balance between model simplicity (high bias) and complexity (high variance).
- High Bias: Underfitting
- High Variance: Overfitting
- Sweet spot: Optimal generalization

---

## C

### CNN (Convolutional Neural Network)
Neural network specialized for processing grid-like data (images).
- Layers: Convolution, Pooling, Fully Connected
- Used for: Image classification, object detection

### Clustering
Unsupervised learning to group similar data points.
- **K-Means**: Partition into k clusters
- **DBSCAN**: Density-based clustering
- **Hierarchical**: Build cluster tree

### Cross-Entropy Loss
Loss function for classification tasks.
- Binary: -[y log(ŷ) + (1-y) log(1-ŷ)]
- Categorical: -Σ y_i log(ŷ_i)

### CUDA
NVIDIA's parallel computing platform for GPU acceleration.
- Essential for deep learning
- 10-100x speedup vs CPU

---

## D

### Data Augmentation
Artificially increase training data by applying transformations.
- Images: Rotation, flip, crop, color jitter
- Text: Synonym replacement, back-translation
- Audio: Time stretch, pitch shift

### Decision Tree
Tree-like model where nodes represent features and leaves represent decisions.
- Pros: Interpretable, handles non-linear relationships
- Cons: Prone to overfitting

### Diffusion Models
Generative models that learn to denoise data.
- Used by: DALL-E 2, Stable Diffusion, Midjourney
- State-of-the-art for image generation

### Dropout
Regularization technique that randomly drops neurons during training.
- Prevents co-adaptation of neurons
- Typical rate: 0.2-0.5

---

## E

### Embedding
Dense vector representation of discrete data.
- **Word Embeddings**: Represent words as vectors
- **Sentence Embeddings**: Represent sentences
- Captures semantic meaning

### Ensemble Learning
Combine multiple models for better performance.
- **Bagging**: Bootstrap + Aggregate (Random Forest)
- **Boosting**: Sequential weak learners (XGBoost, AdaBoost)
- **Stacking**: Meta-model on base models

### Epoch
One complete pass through entire training dataset.
- Typical training: 10-1000 epochs
- Use validation loss to determine when to stop

### Explainable AI (XAI)
Techniques to understand model decisions.
- **SHAP**: SHapley Additive exPlanations
- **LIME**: Local Interpretable Model-agnostic Explanations
- **Attention visualization**: See what model focuses on

---

## F

### F1 Score
Harmonic mean of precision and recall.
- F1 = 2 × (Precision × Recall)/(Precision + Recall)
- Good for imbalanced datasets

### Feature Engineering
Creating new features from raw data.
- Domain knowledge crucial
- Can dramatically improve performance

### Fine-Tuning
Adapting pre-trained model to specific task.
- **Full**: Update all parameters
- **LoRA**: Update small adapter layers
- **Prompt Tuning**: Learn optimal prompts

### FLOPs
Floating Point Operations - measure of computational complexity.
- GPT-3 training: 3.14 × 10^23 FLOPs
- Used to compare model efficiency

---

## G

### GAN (Generative Adversarial Network)
Two neural networks competing: Generator vs Discriminator.
- Generator: Creates fake data
- Discriminator: Distinguishes real from fake
- Used for: Image generation, data augmentation

### Gradient Descent
Optimization algorithm to minimize loss.
- **Batch GD**: Use all data
- **SGD**: Use one sample
- **Mini-batch**: Use batch of samples

### GPT (Generative Pre-trained Transformer)
Autoregressive language model from OpenAI.
- GPT-4: 1.7T parameters
- Used for: Text generation, completion, translation

### GPU (Graphics Processing Unit)
Specialized processor for parallel computations.
- Essential for deep learning
- NVIDIA dominates ML market

---

## H

### Hallucination
When LLM generates false but plausible-sounding information.
- Major challenge for production systems
- Mitigation: RAG, grounding, fact-checking

### Hugging Face
Platform for sharing ML models and datasets.
- 100,000+ pre-trained models
- Transformers library: Industry standard

### Hyperparameter
Configuration external to model, set before training.
- Learning rate, batch size, number of layers
- Tuning crucial for performance

### HNSW (Hierarchical Navigable Small World)
Graph-based algorithm for fast similarity search.
- Used in: Vector databases
- 100x faster than brute-force

---

## I

### Imbalanced Dataset
When classes have different frequencies.
- Example: 99% normal, 1% fraud
- Solutions: Oversampling, undersampling, class weights

### Inference
Using trained model to make predictions.
- vs Training: No backpropagation
- Goal: Low latency, high throughput

### InstructGPT
GPT-3 fine-tuned with human feedback (RLHF).
- Better at following instructions
- Less toxic outputs

---

## K

### K-Fold Cross-Validation
Split data into k folds, train k times.
- Use k-1 folds for training, 1 for validation
- Rotate which fold is validation
- Average results for robust estimate

### KNN (K-Nearest Neighbors)
Classify based on k closest training examples.
- Simple, no training phase
- Slow at inference
- Sensitive to irrelevant features

---

## L

### LLM (Large Language Model)
Neural network trained on massive text corpus.
- 1B+ parameters
- Examples: GPT-4, Claude, LLaMA

### LoRA (Low-Rank Adaptation)
Efficient fine-tuning method.
- Train 0.1% of parameters
- 100x cheaper than full fine-tuning
- Minimal performance loss

### Loss Function
Measures how wrong model predictions are.
- **Regression**: MSE, MAE
- **Classification**: Cross-Entropy
- **Ranking**: Triplet Loss

### LSTM (Long Short-Term Memory)
RNN variant that handles long-term dependencies.
- Gates: Input, Forget, Output
- Used for: Time series, translation
- Largely replaced by Transformers

---

## M

### mAP (mean Average Precision)
Evaluation metric for object detection.
- Considers both precision and recall
- IoU threshold: Usually 0.5

### Mixed Precision Training
Use FP16 and FP32 together.
- 2x faster training
- Half memory usage
- No accuracy loss (usually)

### MLOps
DevOps for ML systems.
- Version control, CI/CD, monitoring
- Tools: MLflow, Kubeflow, Weights & Biases

### Multimodal
Processing multiple data types.
- Text + Images: CLIP, DALL-E
- Text + Audio: Whisper
- All: GPT-4V

---

## N

### Neural Architecture Search (NAS)
Automated search for optimal network architecture.
- Expensive but effective
- Examples: EfficientNet, AmoebaNet

### NLP (Natural Language Processing)
AI for understanding and generating human language.
- Tasks: Translation, sentiment, summarization
- Modern approach: Transformers

### Normalization
Scale features to similar ranges.
- **Min-Max**: Scale to [0,1]
- **Z-Score**: Mean=0, Std=1
- **Batch Norm**: Per-batch normalization

---

## O

### Overfitting
Model learns training data too well, fails on new data.
- Signs: Train accuracy >> Test accuracy
- Solutions: Regularization, more data, simpler model

### One-Hot Encoding
Convert categorical to binary vectors.
- "Red" → [1, 0, 0]
- "Green" → [0, 1, 0]
- "Blue" → [0, 0, 1]

---

## P

### Perceptron
Simplest neural network: single neuron.
- Input → Weighted sum → Activation → Output
- Foundation of deep learning

### Precision
Of predicted positives, how many are actually positive?
- Precision = TP / (TP + FP)
- Important when false positives are costly

### Pre-training
Train model on large dataset before fine-tuning.
- Learn general patterns
- Transfer to specific task

---

## Q

### Quantization
Reduce numerical precision of weights.
- FP32 → INT8: 4x smaller, 2-4x faster
- Minimal accuracy loss (<2%)

### Query
In attention: What information we're looking for.
- Key: What information we have
- Value: The actual information

---

## R

### RAG (Retrieval Augmented Generation)
Enhance LLM with external knowledge retrieval.
- Search → Retrieve → Generate
- Reduces hallucinations

### Recall
Of actual positives, how many did we find?
- Recall = TP / (TP + FN)
- Important when missing positives is costly

### Regularization
Techniques to prevent overfitting.
- **L1**: Sparse weights
- **L2**: Small weights
- **Dropout**: Random neuron dropout

### ResNet (Residual Network)
CNN with skip connections.
- Enables very deep networks (100+ layers)
- Solved vanishing gradient problem

### RLHF (Reinforcement Learning from Human Feedback)
Train model using human preferences.
- Used in: ChatGPT, InstructGPT
- Improves helpfulness and safety

### RNN (Recurrent Neural Network)
Neural network with loops for sequences.
- Maintains hidden state
- Used for: Time series, text
- Largely replaced by Transformers

---

## S

### Semantic Search
Search by meaning, not keywords.
- Uses embeddings
- "car" matches "automobile", "vehicle"

### SGD (Stochastic Gradient Descent)
Update weights using one sample at a time.
- Noisy but fast
- Better than batch GD for large datasets

### SHAP (SHapley Additive exPlanations)
Explain individual predictions.
- Based on game theory
- Shows feature importance

### Softmax
Converts logits to probabilities.
- Output sums to 1
- Used in classification output layer

---

## T

### Tensor
Multi-dimensional array.
- 1D: Vector
- 2D: Matrix
- 3D+: Tensor

### TensorFlow
Google's ML framework.
- Production-focused
- TensorBoard for visualization

### Tokenization
Split text into tokens.
- Word-level: "Hello world" → ["Hello", "world"]
- Subword: "unhappiness" → ["un", "happiness"]
- Character: "Hi" → ["H", "i"]

### TPU (Tensor Processing Unit)
Google's custom chip for ML.
- Optimized for matrix multiplication
- 15x faster than GPU for transformers

### Transfer Learning
Use knowledge from one task on another.
- Pre-train on ImageNet → Fine-tune on medical images
- Saves time and data

### Transformer
Architecture based on attention mechanism.
- Parallel processing (vs RNN sequential)
- Used in: BERT, GPT, T5, Vision Transformers

---

## U

### Underfitting
Model too simple to capture patterns.
- Train and test accuracy both low
- Solution: More complex model, more features

---

## V

### Validation Set
Data used to tune hyperparameters.
- Different from test set
- Prevents information leakage

### Vanishing Gradient
Gradients become extremely small in deep networks.
- Early layers don't learn
- Solution: ReLU, ResNet, better initialization

### Vector Database
Database optimized for similarity search.
- Examples: Pinecone, Weaviate, FAISS
- Used in RAG, recommendation systems

---

## W

### Weight Decay
Regularization by penalizing large weights.
- Equivalent to L2 regularization
- Helps prevent overfitting

### Word2Vec
Technique to create word embeddings.
- Skip-gram or CBOW
- Captures semantic relationships

---

## X

### XGBoost (eXtreme Gradient Boosting)
Optimized gradient boosting framework.
- Wins many Kaggle competitions
- Fast, accurate, handles missing data

---

## Z

### Zero-Shot Learning
Model performs task without specific training.
- "Classify this image" with no examples
- GPT-3, CLIP enable zero-shot

---

## 📊 Quick Reference Tables

### Activation Functions

| Function | Range | Use Case | Pros | Cons |
|----------|-------|----------|------|------|
| ReLU | [0, ∞) | Hidden layers | Fast, sparse | Dead neurons |
| Sigmoid | (0, 1) | Binary output | Smooth | Vanishing gradient |
| Tanh | (-1, 1) | Hidden layers | Zero-centered | Vanishing gradient |
| Softmax | [0, 1] | Multi-class | Probabilities | - |

### Loss Functions

| Task | Loss | Formula |
|------|------|---------|
| Regression | MSE | (y - ŷ)² |
| Binary Classification | BCE | -[y log(ŷ) + (1-y) log(1-ŷ)] |
| Multi-class | Cross-Entropy | -Σ y_i log(ŷ_i) |
| Ranking | Triplet | max(0, d(a,p) - d(a,n) + margin) |

### Optimizers

| Optimizer | Learning Rate | Momentum | Adaptive | Best For |
|-----------|---------------|----------|----------|----------|
| SGD | Fixed | No | No | Simple tasks |
| Momentum | Fixed | Yes | No | Faster convergence |
| Adam | Adaptive | Yes | Yes | Most tasks (default) |
| AdaGrad | Adaptive | No | Yes | Sparse features |
| RMSprop | Adaptive | No | Yes | RNNs |

### Model Architectures

| Type | Input | Architecture | Use Case |
|------|-------|-------------|----------|
| MLP | Tabular | Fully connected | Simple patterns |
| CNN | Images | Conv + Pool | Vision |
| RNN/LSTM | Sequences | Recurrent | Time series |
| Transformer | Sequences | Attention | NLP, modern |
| GAN | Noise | Generator + Discriminator | Generation |
| VAE | Data | Encoder + Decoder | Generation |

---

## 🔗 See Also

- **[LLM Fundamentals](llm_fundamentals/README.md)** - Deep dive into language models
- **[Neural Networks](neural_networks/README.md)** - From scratch to deep learning
- **[RAG Tutorial](rag_tutorial/README.md)** - Semantic search and retrieval

**Keep this glossary handy while learning! 📚**
