# 🚀 Modern ML/AI Techniques (2024-2025)

**A Comprehensive Guide to State-of-the-Art Machine Learning and AI**

> Last Updated: November 2025
> This guide covers cutting-edge techniques, architectures, and methodologies that define the current state of AI.
> **Note:** Model specifications, especially for proprietary models, are based on publicly available information and may not reflect actual implementations.

---

## 📋 Table of Contents

1. [Large Language Models (LLMs)](#large-language-models)
2. [Diffusion Models](#diffusion-models)
3. [Vision Transformers](#vision-transformers)
4. [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation)
5. [Mixture of Experts (MoE)](#mixture-of-experts)
6. [Constitutional AI & RLHF](#constitutional-ai-and-rlhf)
7. [Model Quantization](#model-quantization)
8. [Multimodal Models](#multimodal-models)
9. [Efficient Training Techniques](#efficient-training-techniques)
10. [Emerging Architectures](#emerging-architectures)

---

## 🤖 Large Language Models (LLMs)

### Overview
Large Language Models have revolutionized NLP and are now being applied across domains. Models with billions to trillions of parameters can understand and generate human-like text.

### Key Models (2024-2025)

#### **GPT-4 and GPT-4 Turbo** (OpenAI)
- **Architecture:** Decoder-only transformer
  - *Exact parameters undisclosed; speculation suggests MoE architecture with ~1T+ total parameters*
  - *Note: OpenAI has not confirmed architectural details publicly*
- **Key Features:**
  - 128K context window
  - Multimodal (text + vision)
  - Improved reasoning and factuality
- **Applications:** Code generation, analysis, creative writing, complex reasoning

#### **Claude (Sonnet 4.5, Opus, Sonnet 3.5, Haiku)** (Anthropic)
- **Architecture:** Constitutional AI with RLHF
- **Key Features:**
  - 200K context window
  - Strong safety alignment
  - Excellent instruction following
  - Extended thinking capabilities (Sonnet 4+)
- **Applications:** Long-document analysis, safe AI assistants, code review, complex reasoning

#### **Llama 3** (Meta)
- **Parameters:** 8B, 70B, 405B variants
- **Key Features:**
  - Open source and permissive license
  - Strong performance on benchmarks
  - Efficient fine-tuning
- **Applications:** Research, custom deployments, on-premise solutions

#### **Gemini Ultra/Pro/Nano** (Google DeepMind)
- **Architecture:** Multimodal from the ground up
- **Key Features:**
  - Native multimodal understanding
  - Efficient Nano variant for mobile
  - Strong mathematical reasoning
- **Applications:** Multimodal tasks, mobile deployment, scientific computing

### Training Techniques

#### **1. Chinchilla Scaling Laws**
Optimal model training requires balancing model size and training tokens for compute-optimal performance:
```
Optimal tokens ≈ 20 × parameters (for a given compute budget)
```
**More precisely:** For compute-optimal training, the number of training tokens should scale proportionally with the number of parameters. The Chinchilla paper found that many large models (like GPT-3) were significantly undertrained relative to their size.

**Example:**
- 70B parameter model → ~1.4T training tokens (compute-optimal)
- GPT-3 (175B params) was trained on 300B tokens → undertrained by ~6x

**Key Insight:** Smaller models trained on more data can outperform larger undertrained models, especially when compute budget is fixed. This suggests investing more in high-quality training data rather than just increasing model size.

**Reference:** *Training Compute-Optimal Large Language Models* - Hoffmann et al. (2022)

#### **2. Instruction Tuning**
Fine-tuning pretrained LLMs on instruction-following datasets:
```python
# Example instruction format
{
    "instruction": "Summarize the following text",
    "input": "Long text here...",
    "output": "Summary here..."
}
```

**Popular Datasets:**
- FLAN (Fine-tuned Language Net)
- Alpaca (52K instructions)
- Dolly (15K human-generated)
- OpenOrca (synthetic instructions)

**Reference:** *Scaling Instruction-Finetuned Language Models* - Chung et al. (2022)

#### **3. Parameter-Efficient Fine-Tuning (PEFT)**

**LoRA (Low-Rank Adaptation):**
Instead of updating all weights, inject trainable low-rank matrices:
```
W = W₀ + BA
```
Where:
- W₀: frozen pretrained weights (d × k)
- B: trainable matrix (d × r)
- A: trainable matrix (r × k)
- r << min(d, k) (typically r = 8-64)

**Memory Savings:** Only train ~0.1-1% of parameters!

```python
# Example with HuggingFace PEFT
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                      # Rank
    lora_alpha=32,             # Scaling factor
    target_modules=["q", "v"], # Which layers to adapt
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(base_model, config)
# Train only ~0.5% of parameters!
```

**QLoRA (Quantized LoRA):**
Combines LoRA with 4-bit quantization:
- Base model in 4-bit (NormalFloat4)
- LoRA adapters in 16-bit
- **Enables fine-tuning 65B models on a single 48GB GPU!**

**Reference:** *LoRA: Low-Rank Adaptation of Large Language Models* - Hu et al. (2021)
**Reference:** *QLoRA: Efficient Finetuning of Quantized LLMs* - Dettmers et al. (2023)

### Prompt Engineering Best Practices

#### **Chain-of-Thought (CoT) Prompting**
```python
# Zero-shot CoT
prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?

Let's think step by step:
"""
# Model breaks down reasoning automatically
```

#### **Few-Shot Prompting**
```python
prompt = """
Sentiment classification:

Text: "The movie was fantastic!"
Sentiment: Positive

Text: "Terrible experience, very disappointed."
Sentiment: Negative

Text: "The product works as expected."
Sentiment: {model completes}
"""
```

#### **Tree-of-Thoughts (ToT)**
Explore multiple reasoning paths:
```
Problem
├── Approach 1
│   ├── Step 1a → Step 2a → Solution A
│   └── Step 1b → Step 2b → Solution B
└── Approach 2
    └── Step 1c → Step 2c → Solution C
```

**Reference:** *Tree of Thoughts: Deliberate Problem Solving with LLMs* - Yao et al. (2023)

---

## 🎨 Diffusion Models

### Overview
Diffusion models generate high-quality images by learning to reverse a noise-adding process.

### Mathematical Foundation

**Forward Process (Adding Noise):**
```
q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)
```
- Start with real image x₀
- Gradually add Gaussian noise over T steps
- End with pure noise xₜ ~ N(0, I)

**Reverse Process (Denoising):**
```
pθ(xₜ₋₁ | xₜ) = N(xₜ₋₁; μθ(xₜ, t), Σθ(xₜ, t))
```
- Learn to predict and remove noise
- Start from random noise
- Gradually denoise to generate image

**Training Objective:**
```
L = 𝔼ₜ,x₀,ε[||ε - εθ(xₜ, t)||²]
```
Predict the noise ε that was added at timestep t.

### Key Models

#### **Stable Diffusion** (Stability AI)
- **Architecture:** Latent Diffusion Model (LDM)
- **Key Innovation:** Work in latent space instead of pixel space
  - Encode images to latent space (VAE encoder)
  - Perform diffusion in latent space (8x smaller!)
  - Decode back to pixels (VAE decoder)
- **Variants:**
  - SD 1.5 (512×512)
  - SDXL (1024×1024, better quality)
  - SD Turbo (1-4 steps instead of 50!)

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1"
)

image = pipe(
    "A serene landscape with mountains",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
```

#### **DALL-E 3** (OpenAI)
- **Improvements over DALL-E 2:**
  - Better prompt following
  - Higher image quality
  - Improved text rendering
  - Safety mitigations

#### **Midjourney v6**
- **Architecture:** Proprietary
- **Strengths:** Artistic quality, prompt understanding
- **Use cases:** Creative design, concept art

### Advanced Techniques

#### **Classifier-Free Guidance**
Balance between diversity and prompt adherence:
```
ε̃ = εᵤ + s·(εᶜ - εᵤ)
```
Where:
- εᵤ: unconditional prediction (no prompt)
- εᶜ: conditional prediction (with prompt)
- s: guidance scale (typically 7-15)

Higher s = stronger prompt following, less diversity

#### **ControlNet**
Add spatial control to diffusion models:
- Canny edges
- Depth maps
- Segmentation masks
- Pose estimation

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet
)

image = pipe(
    prompt="modern living room",
    image=canny_edge_image,
    num_inference_steps=20
).images[0]
```

**Reference:** *Adding Conditional Control to Text-to-Image Diffusion Models* - Zhang et al. (2023)

#### **DreamBooth & LoRA Fine-tuning**
Personalize diffusion models with few images (3-5):
```python
# Train custom concept with LoRA
# "a photo of sks person"
# Only 5 images needed!
# Fine-tune in <10 minutes on 1 GPU
```

**Reference:** *DreamBooth: Fine Tuning Text-to-Image Diffusion Models* - Ruiz et al. (2022)

---

## 👁️ Vision Transformers

### Architecture Evolution

#### **Original ViT (Vision Transformer)**
```
Image (224×224×3)
  ↓ Split into patches (16×16)
  ↓ Linear embedding
  ↓ Add position embeddings
  ↓ Transformer encoder (12-24 layers)
  ↓ Classification head
```

**Key Insight:** Treat image patches as tokens, just like words in NLP!

**Patch Embedding:**
```
Number of patches = (H/P) × (W/P)
Example: 224×224 image, 16×16 patches → 14×14 = 196 tokens
```

**Reference:** *An Image is Worth 16x16 Words* - Dosovitskiy et al. (2020)

#### **CLIP (Contrastive Language-Image Pretraining)**
Joint text-image embeddings:
```
Images ────┐
           ├──→ Contrastive Learning ──→ Aligned Embeddings
Texts  ────┘
```

**Training:**
```
Maximize similarity: sim(img_i, text_i)
Minimize similarity: sim(img_i, text_j) for i≠j
```

**Applications:**
- Zero-shot image classification
- Image search with text queries
- Multimodal embeddings

```python
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

images = [...]  # List of PIL images
texts = ["a dog", "a cat", "a bird"]

inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Get similarity scores
logits_per_image = outputs.logits_per_image  # Image-text similarity
probs = logits_per_image.softmax(dim=1)
```

**Reference:** *Learning Transferable Visual Models From Natural Language Supervision* - Radford et al. (2021)

#### **SAM (Segment Anything Model)**
Universal image segmentation:
```
Image → Image Encoder (ViT) → Embeddings
Prompt (points/boxes/text) → Prompt Encoder → Query
Embeddings + Query → Mask Decoder → Segmentation Mask
```

**Key Features:**
- Zero-shot segmentation
- Interactive prompting
- 1 billion+ masks trained
- Works on any image domain

```python
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),
    multimask_output=True
)
```

**Reference:** *Segment Anything* - Kirillov et al. (2023)

#### **DINOv2 (Self-Supervised Vision)**
Self-supervised learning for vision features:
- No labels needed!
- Learns robust image representations
- Excellent for transfer learning

**Applications:**
- Feature extraction
- Image retrieval
- Few-shot learning
- Downstream task fine-tuning

**Reference:** *DINOv2: Learning Robust Visual Features without Supervision* - Oquab et al. (2023)

---

## 🔍 Retrieval-Augmented Generation (RAG)

### Overview
Combine LLMs with external knowledge retrieval for factual, up-to-date responses.

### Architecture
```
User Query
    ↓
Retriever (find relevant docs)
    ↓
Retrieved Context + Query
    ↓
LLM Generator
    ↓
Grounded Response
```

### Implementation Pipeline

#### **1. Document Processing**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(documents)
```

#### **2. Embedding & Vector Storage**
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Store in vector database
vectorstore = FAISS.from_documents(chunks, embeddings)
```

#### **3. Retrieval**
```python
# Retrieve relevant chunks
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

relevant_docs = retriever.get_relevant_documents(query)
```

#### **4. Generation**
```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain({"query": "What is the main topic?"})
answer = result["result"]
sources = result["source_documents"]
```

### Advanced RAG Techniques

#### **Hybrid Search**
Combine dense (semantic) and sparse (keyword) retrieval:
```python
# BM25 (sparse) + Dense embeddings
from rank_bm25 import BM25Okapi

# Weighted combination
final_score = α * dense_score + (1-α) * sparse_score
```

#### **Reranking**
Use cross-encoder to rerank retrieved documents:
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict([(query, doc) for doc in docs])
reranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
```

#### **HyDE (Hypothetical Document Embeddings)**
Generate hypothetical answers, then retrieve:
```
Query → LLM generates hypothetical answer → Embed → Retrieve similar docs
```

**Reference:** *Precise Zero-Shot Dense Retrieval without Relevance Labels* - Gao et al. (2022)

#### **Self-RAG**
Model decides when to retrieve and how to use retrieved info:
```
Generate → Self-evaluate → Retrieve if needed → Refine → Output
```

**Reference:** *Self-RAG: Learning to Retrieve, Generate, and Critique* - Asai et al. (2023)

### Best Practices

1. **Chunking Strategy:**
   - Use semantic chunking (sentence/paragraph boundaries)
   - Overlap chunks (20-30%) for context
   - Typical size: 500-1500 tokens

2. **Embedding Models:**
   - OpenAI: `text-embedding-3-large` (best quality)
   - Open source: `BAAI/bge-large-en-v1.5`
   - Sentence transformers: `all-MiniLM-L6-v2` (fast)

3. **Vector Databases:**
   - Small scale: FAISS (in-memory)
   - Production: Pinecone, Weaviate, Qdrant
   - Self-hosted: Milvus, Chroma

4. **Evaluation Metrics:**
   - Retrieval: Recall@k, MRR, NDCG
   - Generation: Faithfulness, answer relevance
   - End-to-end: Human evaluation, LLM-as-judge

---

## 🧩 Mixture of Experts (MoE)

### Architecture
```
Input
  ↓
Router Network (decides which experts to use)
  ↓
Top-K Experts (only K out of N experts activated)
  ↓
Weighted combination of expert outputs
  ↓
Output
```

### Mathematical Formulation
```
y = Σᵢ Gᵢ(x) · Eᵢ(x)
```
Where:
- G(x): Gating function (softmax of router logits)
- Eᵢ(x): i-th expert network
- Only top-k experts have non-zero G(x)

### Advantages

**1. Sparse Activation:**
- Total parameters: 1T
- Active parameters per token: 100B
- **8x memory efficiency during inference!**

**2. Specialization:**
- Different experts learn different skills
- Expert 1: Math & logic
- Expert 2: Creative writing
- Expert 3: Code generation
- Expert 4: Factual knowledge

**3. Scalability:**
- Add more experts without proportionally increasing compute
- Scale parameters independently of FLOP

### Modern MoE Models

#### **GPT-4** (Speculative - Unconfirmed)
- *Architecture details not officially disclosed by OpenAI*
- *Speculation based on analysis suggests possible MoE with:*
  - *~8 experts × ~220B parameters each ≈ ~1.76T total*
  - *Top-2 routing per token*
  - *Active: ~440B parameters per forward pass*
- **Important:** These are unverified estimates from the research community

#### **Mixtral 8x7B** (Mistral AI)
- 8 experts × 7B parameters = 56B total
- Top-2 routing
- Active: 14B parameters (matches GPT-3.5 performance!)
- **Open source and Apache 2.0 licensed**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    device_map="auto",
    load_in_4bit=True  # Quantized for 1 GPU!
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

inputs = tokenizer("Explain quantum computing", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
```

#### **Switch Transformer** (Google)
- 1.6 trillion parameters
- Top-1 routing (simplest)
- Trained on massive TPU clusters

**Reference:** *Switch Transformers: Scaling to Trillion Parameter Models* - Fedus et al. (2021)

### Training Challenges

**1. Load Balancing:**
All experts should be used equally:
```
L_balance = α · Σᵢ fᵢ · Pᵢ
```
Where:
- fᵢ: fraction of tokens routed to expert i
- Pᵢ: fraction of router probability for expert i

**2. Expert Collapse:**
Router learns to use only a few experts → Add auxiliary loss

**3. Communication Overhead:**
Experts on different devices → Use expert parallelism

---

## 🎯 Constitutional AI and RLHF

### RLHF (Reinforcement Learning from Human Feedback)

#### **Three-Stage Process:**

**Stage 1: Supervised Fine-Tuning (SFT)**
```
Pretrained LLM → Fine-tune on high-quality demonstrations → SFT Model
```

**Stage 2: Reward Model Training**
```
Pairs of outputs → Human ranks preferred output → Train reward model
```
Reward model learns: `r(x, y)` = quality score for output y given input x

**Stage 3: RL Optimization**
Use PPO (Proximal Policy Optimization):
```
Objective: max 𝔼[r(x, y)] - β·KL(π || π_SFT)
```
Where:
- r(x, y): reward from trained reward model
- KL divergence: keep model close to SFT model
- β: controls how much we deviate from SFT

```python
# Conceptual RLHF pipeline
# Stage 1: SFT
sft_model = train_supervised(base_model, demonstrations)

# Stage 2: Reward model
reward_model = train_reward_model(
    comparisons=[
        (prompt, good_response, bad_response),
        ...
    ]
)

# Stage 3: PPO training
for batch in dataloader:
    prompts = batch["prompts"]
    responses = sft_model.generate(prompts)
    rewards = reward_model(prompts, responses)

    # PPO update
    ppo_loss = compute_ppo_loss(rewards, responses)
    update_model(ppo_loss)
```

**Reference:** *Training Language Models to Follow Instructions with Human Feedback* - Ouyang et al. (2022)

### Constitutional AI (CAI)

Align models using AI feedback instead of human feedback!

#### **Process:**

**1. Critique Phase:**
```
Model Output → AI Critic (using constitution) → Identified problems
```

**2. Revision Phase:**
```
Original output + Critique → Revise → Improved output
```

**3. Train on Revisions:**
```
Train model to directly produce revised outputs
```

**Constitution Example:**
```yaml
principles:
  - "Choose the response that is most helpful, honest, and harmless"
  - "Avoid outputs that could be dangerous or illegal"
  - "Prefer responses that respect human autonomy"
  - "Don't help with unethical requests"
```

**Advantages:**
- Scalable (no human labeling bottleneck)
- Transparent (principles are explicit)
- Iterative improvement
- Reduces harmful outputs

**Reference:** *Constitutional AI: Harmlessness from AI Feedback* - Bai et al. (2022)

### DPO (Direct Preference Optimization)

Simpler alternative to RLHF - skip the reward model!

**Direct optimization:**
```
L = -𝔼[(log σ(β log(π(y_w|x)/π_ref(y_w|x)) - β log(π(y_l|x)/π_ref(y_l|x))))]
```
Where:
- y_w: preferred (winning) response
- y_l: dispreferred (losing) response
- π_ref: reference model (SFT model)

**Advantages:**
- No separate reward model needed
- Simpler training pipeline
- More stable than PPO
- Same or better performance

```python
# DPO training (simplified)
for batch in preference_data:
    prompt, chosen, rejected = batch

    chosen_logprob = model.log_prob(chosen | prompt)
    rejected_logprob = model.log_prob(rejected | prompt)

    ref_chosen_logprob = ref_model.log_prob(chosen | prompt)
    ref_rejected_logprob = ref_model.log_prob(rejected | prompt)

    loss = -log_sigmoid(
        beta * (chosen_logprob - ref_chosen_logprob) -
        beta * (rejected_logprob - ref_rejected_logprob)
    )

    update_model(loss)
```

**Reference:** *Direct Preference Optimization* - Rafailov et al. (2023)

---

## ⚡ Model Quantization

### Overview
Reduce model precision to decrease memory and increase inference speed.

### Quantization Levels

| Precision | Bits | Memory | Speed | Quality |
|-----------|------|--------|-------|---------|
| FP32 | 32 | 1.0x | 1.0x | 100% |
| FP16 | 16 | 0.5x | 2.0x | ~99.9% |
| BF16 | 16 | 0.5x | 2.0x | ~99.95% |
| INT8 | 8 | 0.25x | 3-4x | ~99% |
| INT4 | 4 | 0.125x | 5-6x | ~95-98% |

### Quantization Methods

#### **1. Post-Training Quantization (PTQ)**
Quantize after training - no retraining needed!

**Symmetric Quantization:**
```
Q(x) = round(x / scale) · scale
scale = max(|x|) / (2^(bits-1) - 1)
```

**Asymmetric Quantization:**
```
Q(x) = round((x - zero_point) / scale)
scale = (max(x) - min(x)) / (2^bits - 1)
zero_point = round(-min(x) / scale)
```

#### **2. Quantization-Aware Training (QAT)**
Include quantization in training loop:
```python
# Fake quantization during training
def fake_quantize(x, scale, zero_point, bits=8):
    qmin, qmax = 0, 2**bits - 1
    x_int = torch.round(x / scale + zero_point)
    x_int = torch.clamp(x_int, qmin, qmax)
    x_dequant = (x_int - zero_point) * scale
    return x_dequant

# Forward pass
x_quant = fake_quantize(x, scale, zero_point)
# Backward pass uses gradients w.r.t. original x
```

#### **3. GPTQ (GPT Quantization)**
Optimal 4-bit quantization for LLMs:
- Layer-wise quantization
- Minimizes reconstruction error
- Calibration on small dataset (~128 samples)

**7B model in 4GB of memory instead of 28GB!**

```python
from transformers import AutoModelForCausalLM, GPTQConfig

quantization_config = GPTQConfig(
    bits=4,
    dataset="c4",
    tokenizer=tokenizer
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**Reference:** *GPTQ: Accurate Post-Training Quantization for GPT Models* - Frantar et al. (2022)

#### **4. GGUF/GGML Format**
Quantization format optimized for CPU inference:
- Multiple quantization schemes (Q4_0, Q4_1, Q5_0, Q8_0)
- Highly optimized for Apple Silicon and AVX2
- Used by llama.cpp

**Quantization Schemes:**
```
Q4_0: 4-bit, small file size, lowest quality
Q4_K_M: 4-bit, medium quality (recommended)
Q5_K_M: 5-bit, higher quality
Q8_0: 8-bit, best quality
```

#### **5. AWQ (Activation-aware Weight Quantization)**
Protect important weights from quantization:
- Analyze activation patterns
- Keep salient weights in higher precision
- Better quality than naive quantization

**Reference:** *AWQ: Activation-aware Weight Quantization* - Lin et al. (2023)

### Practical Example: Quantize Llama 2

```python
# Option 1: GPTQ (4-bit for GPU)
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0",
    use_safetensors=True
)

# Option 2: GGUF (for CPU, especially Apple Silicon)
# Using llama-cpp-python
from llama_cpp import Llama

model = Llama(
    model_path="llama-2-7b.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8
)

# Option 3: BitsAndBytes (simple integration)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto"
)
```

### When to Use Each Method

| Method | Use Case | Quality | Speed | Ease |
|--------|----------|---------|-------|------|
| GPTQ | GPU inference, LLMs | High | Fast | Medium |
| GGUF | CPU inference, local | Medium-High | Medium | Easy |
| AWQ | GPU inference, quality-critical | Highest | Fast | Medium |
| BitsAndBytes | Quick prototyping | Medium | Fast | Very Easy |

---

## 🎭 Multimodal Models

### Overview
Models that understand multiple modalities: text, images, audio, video.

### State-of-the-Art Models

#### **GPT-4V (Vision)**
- Text + image understanding
- Can analyze charts, diagrams, screenshots
- Spatial reasoning and OCR

#### **Gemini Ultra**
- Native multimodal training
- Text + image + audio
- Strong on mathematical diagrams

#### **LLaVA (Large Language and Vision Assistant)**
Open-source vision-language model:
```python
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates

# Load model
model = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")

# Multimodal conversation
conv = conv_templates["v1"].copy()
conv.append_message(conv.roles[0], "What's in this image?")
prompt = conv.get_prompt()

# Process image + text
outputs = model.generate(
    input_ids=text_inputs,
    images=image_tensor,
    max_new_tokens=200
)
```

**Training Strategy:**
1. Pretrain vision encoder (CLIP)
2. Pretrain LLM
3. Train projection layer (vision → text space)
4. Fine-tune end-to-end on instruction data

**Reference:** *Visual Instruction Tuning* - Liu et al. (2023)

#### **BLIP-2**
Bootstrapping vision-language pretraining:
- Frozen vision encoder
- Frozen LLM
- Learnable Q-Former (queries visual features)

**Advantage:** Reuse existing pretrained models efficiently!

**Reference:** *BLIP-2: Bootstrapping Language-Image Pre-training* - Li et al. (2023)

---

## 🚄 Efficient Training Techniques

### 1. Flash Attention
Faster and more memory-efficient attention:
```
Standard Attention: O(N²) memory
Flash Attention: O(N) memory
Speed: 2-4x faster!
```

**How it works:**
- Tiling: Compute attention in blocks
- Recomputation: Recompute attention in backward pass
- Avoid materializing full N×N matrix

```python
# PyTorch 2.0+
import torch.nn.functional as F

# Automatic with torch.compile
output = F.scaled_dot_product_attention(
    query, key, value,
    is_causal=True  # Uses Flash Attention if available
)
```

**Reference:** *FlashAttention: Fast and Memory-Efficient Exact Attention* - Dao et al. (2022)

### 2. Gradient Checkpointing
Trade compute for memory:
```
Normal: Store all activations (high memory)
Checkpointing: Recompute activations in backward pass (low memory)
```

```python
# HuggingFace Transformers
model.gradient_checkpointing_enable()

# PyTorch
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(x):
    return checkpoint(expensive_function, x)
```

### 3. Mixed Precision Training
Use FP16/BF16 for speed, FP32 for stability:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = model(batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits:**
- 2-3x faster training
- 50% less memory
- Minimal accuracy loss

### 4. DeepSpeed ZeRO
Distribute optimizer states, gradients, and parameters:
```
ZeRO-1: Partition optimizer states → 4x memory reduction
ZeRO-2: + Partition gradients → 8x memory reduction
ZeRO-3: + Partition parameters → Linear scaling!
```

```python
# DeepSpeed config
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        }
    }
}
```

**Train 175B model on 64 GPUs (vs 1024 GPUs without ZeRO)!**

**Reference:** *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* - Rajbhandari et al. (2020)

### 5. FSDP (Fully Sharded Data Parallel)
PyTorch's native alternative to DeepSpeed:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(...)
)
```

---

## 🔮 Emerging Architectures

### 1. Mamba (State Space Models)
Alternative to transformers with linear-time inference:
```
Transformers: O(N²) attention
Mamba: O(N) selective state space
```

**Advantages:**
- Constant memory during inference
- Scales to very long sequences (100K+ tokens)
- Competitive with transformers on language tasks

**Reference:** *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* - Gu & Dao (2023)

### 2. Retentive Networks
Parallel training, recurrent inference:
```
Training: O(N²) parallel (like transformers)
Inference: O(1) memory per step (like RNNs)
```

**Best of both worlds!**

**Reference:** *Retentive Network: A Successor to Transformer* - Sun et al. (2023)

### 3. Hyena Hierarchy
Sub-quadratic attention alternative using convolutions:
```
Long convolutions (FFT-based) + Gating
→ Sub-quadratic complexity
→ Competitive with attention
```

**Reference:** *Hyena Hierarchy: Towards Larger Convolutional Language Models* - Poli et al. (2023)

---

## 📚 Key Papers Reference

### LLMs
- GPT-4 Technical Report - OpenAI (2023)
- Llama 2 - Touvron et al. (2023)
- Constitutional AI - Bai et al. (2022)
- Scaling Laws - Kaplan et al. (2020)

### Diffusion Models
- Denoising Diffusion Probabilistic Models - Ho et al. (2020)
- Latent Diffusion Models - Rombach et al. (2022)
- ControlNet - Zhang et al. (2023)

### Vision
- Vision Transformer - Dosovitskiy et al. (2020)
- CLIP - Radford et al. (2021)
- Segment Anything - Kirillov et al. (2023)

### Efficiency
- LoRA - Hu et al. (2021)
- FlashAttention - Dao et al. (2022)
- QLoRA - Dettmers et al. (2023)
- ZeRO - Rajbhandari et al. (2020)

### RAG
- Self-RAG - Asai et al. (2023)
- HyDE - Gao et al. (2022)
- Dense Passage Retrieval - Karpukhin et al. (2020)

---

## 🛠️ Practical Tools & Frameworks

### LLM Development
- **HuggingFace Transformers:** Unified API for all models
- **LangChain:** LLM application framework
- **LlamaIndex:** Data framework for LLM apps
- **vLLM:** High-throughput inference engine
- **OpenLLM:** Deploy LLMs in production

### Fine-tuning
- **Axolotl:** Simplified fine-tuning tool
- **PEFT:** Parameter-efficient fine-tuning
- **Ludwig:** Low-code ML platform
- **Unsloth:** Fast and memory-efficient fine-tuning

### Quantization
- **llama.cpp:** CPU/Apple Silicon inference
- **GPTQ-for-LLaMa:** GPU quantization
- **BitsAndBytes:** Easy 4/8-bit quantization
- **AutoGPTQ:** Automated GPTQ

### Deployment
- **TensorRT-LLM:** NVIDIA optimized inference
- **Text Generation Inference (TGI):** HuggingFace inference
- **Ollama:** Local LLM runner
- **LM Studio:** GUI for local LLMs

### Monitoring & Evaluation
- **Weights & Biases:** Experiment tracking
- **MLflow:** ML lifecycle management
- **Arize:** ML observability
- **LangSmith:** LLM app debugging

---

## 🎯 Interview Questions

### LLMs
1. **Explain the difference between GPT-4 and Llama 2 architectures.**
2. **How does LoRA achieve parameter-efficient fine-tuning?**
3. **What are the trade-offs between instruction tuning and RLHF?**
4. **Explain the Chinchilla scaling laws and their implications.**
5. **How would you deploy a 70B parameter model with limited GPU memory?**

### Diffusion Models
1. **Explain the forward and reverse diffusion processes.**
2. **What is classifier-free guidance and why is it important?**
3. **How does Stable Diffusion differ from DALL-E 2?**
4. **What is the role of the VAE in latent diffusion models?**
5. **How would you add spatial control to image generation?**

### RAG
1. **Explain the RAG architecture and its components.**
2. **What are the differences between dense and sparse retrieval?**
3. **How would you evaluate a RAG system?**
4. **What is HyDE and when would you use it?**
5. **How do you prevent hallucinations in RAG?**

### Efficiency
1. **Explain the difference between 4-bit and 8-bit quantization.**
2. **How does Flash Attention achieve memory efficiency?**
3. **What is the difference between ZeRO-2 and ZeRO-3?**
4. **When would you use gradient checkpointing?**
5. **Compare GPTQ and GGUF quantization formats.**

---

## 🚀 Next Steps

### Hands-On Projects
1. **Fine-tune Llama 2 with LoRA** on custom dataset
2. **Build a RAG system** for your documents
3. **Deploy quantized model** on consumer hardware
4. **Create custom ControlNet** for specific art style
5. **Implement multi-modal chatbot** with vision

### Further Learning
- Follow ML papers on arXiv (cs.AI, cs.LG, cs.CL, cs.CV)
- Join Discord communities (HuggingFace, LocalLLaMA)
- Contribute to open-source projects
- Participate in Kaggle competitions
- Build and share your own models

---

**Last Updated:** November 2025
**Next Update:** January 2026 (Q1 2026 techniques)

**Disclaimer:** Model specifications for proprietary models (GPT-4, Gemini, Claude) are based on publicly available information and may not reflect actual implementations. Always verify critical information from official sources.

---

*For the latest updates and community discussions, see the main [README](./README.md)*
