# Prompt Engineering & LLM Fine-tuning Guide

Master the art of prompt engineering and fine-tuning large language models for production.

## Table of Contents
1. [Prompt Engineering Fundamentals](#prompt-engineering-fundamentals)
2. [Advanced Prompting Techniques](#advanced-prompting-techniques)
3. [LLM Fine-tuning Methods](#llm-fine-tuning-methods)
4. [Parameter-Efficient Fine-tuning](#parameter-efficient-fine-tuning)
5. [Alignment & RLHF](#alignment-rlhf)
6. [Production Deployment](#production-deployment)

---

## Prompt Engineering Fundamentals

### Basic Prompting Patterns

```python
class PromptTemplates:
    """Collection of effective prompt templates"""

    @staticmethod
    def zero_shot(task, input_text):
        """Zero-shot prompting"""
        return f"""Task: {task}

Input: {input_text}

Output:"""

    @staticmethod
    def few_shot(task, examples, input_text):
        """Few-shot prompting with examples"""
        prompt = f"Task: {task}\n\n"

        for i, (inp, out) in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {inp}\n"
            prompt += f"Output: {out}\n\n"

        prompt += f"Input: {input_text}\nOutput:"
        return prompt

    @staticmethod
    def chain_of_thought(question):
        """Chain-of-thought prompting"""
        return f"""Question: {question}

Let's solve this step by step:
1."""

    @staticmethod
    def instruction_following(instruction, context, input_text):
        """Instruction-following format"""
        return f"""### Instruction:
{instruction}

### Context:
{context}

### Input:
{input_text}

### Response:"""

    @staticmethod
    def role_based(role, task, input_text):
        """Role-based prompting"""
        return f"""You are a {role}.

Your task is to: {task}

Input: {input_text}

Output:"""

# Usage Examples
templates = PromptTemplates()

# Zero-shot
prompt = templates.zero_shot(
    task="Classify the sentiment of the text as positive, negative, or neutral",
    input_text="I love this product!"
)

# Few-shot
examples = [
    ("The movie was amazing!", "Positive"),
    ("Terrible experience, never again.", "Negative"),
    ("It was okay, nothing special.", "Neutral")
]
prompt = templates.few_shot(
    task="Sentiment classification",
    examples=examples,
    input_text="Best purchase I've made this year!"
)

# Chain-of-thought
prompt = templates.chain_of_thought(
    question="Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"
)

# Instruction
prompt = templates.instruction_following(
    instruction="Extract all named entities from the text",
    context="Named entities include: person names, organizations, locations, dates",
    input_text="Apple CEO Tim Cook announced new products in Cupertino on September 12th."
)
```

---

### Prompt Optimization

```python
import openai
from typing import List, Dict

class PromptOptimizer:
    """Automatically optimize prompts"""

    def __init__(self, model="gpt-4", api_key=None):
        self.model = model
        openai.api_key = api_key

    def test_prompt(self, prompt: str, test_cases: List[Dict]) -> float:
        """Evaluate prompt on test cases"""

        correct = 0
        for case in test_cases:
            full_prompt = prompt.format(**case['input'])

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0
            )

            prediction = response.choices[0].message.content.strip()

            if self._check_correctness(prediction, case['expected_output']):
                correct += 1

        return correct / len(test_cases)

    def optimize_prompt(self, initial_prompt: str, test_cases: List[Dict], iterations=5):
        """Iteratively improve prompt"""

        best_prompt = initial_prompt
        best_score = self.test_prompt(initial_prompt, test_cases)

        for iteration in range(iterations):
            # Generate variations
            variation_prompt = f"""Given this prompt:
\"\"\"{best_prompt}\"\"\"

Generate 3 improved variations that would give better results. Focus on:
1. Clarity and specificity
2. Better examples
3. Clear output format

Variations:"""

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": variation_prompt}],
                temperature=0.7
            )

            variations = self._parse_variations(response.choices[0].message.content)

            # Test each variation
            for variation in variations:
                score = self.test_prompt(variation, test_cases)

                if score > best_score:
                    best_score = score
                    best_prompt = variation
                    print(f"Iteration {iteration+1}: New best score = {best_score:.2%}")

        return best_prompt, best_score

    def _check_correctness(self, prediction, expected):
        """Check if prediction matches expected output"""
        # Implement your correctness logic
        return prediction.lower().strip() == expected.lower().strip()

    def _parse_variations(self, response):
        """Parse prompt variations from response"""
        # Simple parsing - split by numbers
        variations = []
        for line in response.split('\n'):
            if line.strip() and not line.strip()[0].isdigit():
                variations.append(line.strip())
        return variations[:3]

# Usage
optimizer = PromptOptimizer(model="gpt-4", api_key="your-api-key")

test_cases = [
    {
        'input': {'text': 'I love this!'},
        'expected_output': 'Positive'
    },
    {
        'input': {'text': 'This is terrible'},
        'expected_output': 'Negative'
    }
]

best_prompt, score = optimizer.optimize_prompt(
    initial_prompt="Classify sentiment: {text}\nSentiment:",
    test_cases=test_cases,
    iterations=5
)

print(f"Best prompt (accuracy {score:.2%}):\n{best_prompt}")
```

---

## Advanced Prompting Techniques

### Chain-of-Thought (CoT) with Self-Consistency

```python
class ChainOfThoughtReasoning:
    """Advanced CoT with self-consistency"""

    def __init__(self, model="gpt-4"):
        self.model = model

    def chain_of_thought(self, question: str, num_samples=5):
        """Generate multiple reasoning paths and vote"""

        reasoning_paths = []

        for _ in range(num_samples):
            prompt = f"""Question: {question}

Let's approach this step-by-step:
1."""

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Higher temp for diversity
                max_tokens=500
            )

            reasoning = response.choices[0].message.content
            reasoning_paths.append(reasoning)

        # Extract final answers
        final_answers = [self._extract_answer(path) for path in reasoning_paths]

        # Majority vote
        from collections import Counter
        answer_counts = Counter(final_answers)
        most_common = answer_counts.most_common(1)[0]

        return {
            'final_answer': most_common[0],
            'confidence': most_common[1] / num_samples,
            'reasoning_paths': reasoning_paths
        }

    def _extract_answer(self, reasoning: str):
        """Extract final answer from reasoning"""
        # Look for "Therefore", "Answer:", etc.
        for trigger in ["Therefore", "Answer:", "Final answer:"]:
            if trigger in reasoning:
                answer = reasoning.split(trigger)[-1].strip()
                return answer.split('\n')[0].strip()
        return reasoning.split('\n')[-1].strip()

# Usage
cot = ChainOfThoughtReasoning(model="gpt-4")

result = cot.chain_of_thought(
    question="A store sells apples for $2 each and oranges for $3 each. If I buy 5 fruits and spend $13, how many apples did I buy?",
    num_samples=5
)

print(f"Answer: {result['final_answer']}")
print(f"Confidence: {result['confidence']:.1%}")
```

---

### ReAct (Reasoning + Acting)

```python
class ReActAgent:
    """ReAct pattern: Reason about actions and observations"""

    def __init__(self, model="gpt-4", tools=None):
        self.model = model
        self.tools = tools or {}

    def add_tool(self, name: str, function, description: str):
        """Add a tool the agent can use"""
        self.tools[name] = {
            'function': function,
            'description': description
        }

    def solve(self, question: str, max_steps=10):
        """Solve problem using ReAct pattern"""

        # Build tool descriptions
        tool_desc = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])

        conversation = []
        current_step = 1

        initial_prompt = f"""Question: {question}

You have access to the following tools:
{tool_desc}

Use the ReAct pattern:
1. Thought: Reason about what to do next
2. Action: Choose a tool and provide input
3. Observation: Get result from tool
4. Repeat until you can answer

Step 1:
Thought:"""

        conversation.append({"role": "user", "content": initial_prompt})

        while current_step <= max_steps:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=conversation,
                temperature=0
            )

            agent_response = response.choices[0].message.content
            conversation.append({"role": "assistant", "content": agent_response})

            # Check if final answer
            if "Answer:" in agent_response or "Final Answer:" in agent_response:
                return self._extract_final_answer(agent_response)

            # Execute action if present
            if "Action:" in agent_response:
                action_name, action_input = self._parse_action(agent_response)

                if action_name in self.tools:
                    observation = self.tools[action_name]['function'](action_input)

                    # Add observation
                    obs_message = f"Observation: {observation}\n\nStep {current_step + 1}:\nThought:"
                    conversation.append({"role": "user", "content": obs_message})

            current_step += 1

        return "Could not solve within step limit"

    def _parse_action(self, response: str):
        """Parse action from response"""
        action_line = [line for line in response.split('\n') if line.startswith('Action:')][0]
        parts = action_line.replace('Action:', '').strip().split('(')
        action_name = parts[0].strip()
        action_input = parts[1].rstrip(')').strip(' "\'')
        return action_name, action_input

    def _extract_final_answer(self, response: str):
        """Extract final answer"""
        for trigger in ["Answer:", "Final Answer:"]:
            if trigger in response:
                return response.split(trigger)[-1].strip()
        return response

# Usage
agent = ReActAgent(model="gpt-4")

# Add tools
agent.add_tool(
    name="Calculator",
    function=lambda x: eval(x),  # In production, use safe evaluation
    description="Evaluate mathematical expressions. Input: '2 + 2', Output: 4"
)

agent.add_tool(
    name="Wikipedia",
    function=lambda x: f"Wikipedia article about {x}...",  # Mock
    description="Search Wikipedia. Input: topic name"
)

answer = agent.solve(
    question="What is the square root of the year Albert Einstein was born?",
    max_steps=10
)

print(f"Final Answer: {answer}")
```

---

## LLM Fine-tuning Methods

### Full Fine-tuning

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

class LLMFineTuner:
    """Fine-tune LLM on custom dataset"""

    def __init__(self, model_name="gpt2", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_dataset(self, texts, max_length=512):
        """Prepare dataset for training"""

        def tokenize_function(examples):
            return self.tokenizer(
                examples,
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )

        tokenized = [tokenize_function(text) for text in texts]
        return tokenized

    def train(self, train_texts, val_texts, output_dir="./finetuned_model", epochs=3):
        """Train the model"""

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts)
        val_dataset = self.prepare_dataset(val_texts)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
            gradient_accumulation_steps=4,
            fp16=True  # Mixed precision
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )

        # Train
        trainer.train()

        # Save
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return trainer

    def generate(self, prompt, max_length=100):
        """Generate text"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
finetuner = LLMFineTuner(model_name="gpt2", device="cuda")

# Prepare training data
train_texts = [
    "Question: What is Python? Answer: Python is a high-level programming language...",
    "Question: What is ML? Answer: Machine Learning is a subset of AI...",
    # ... more examples
]

val_texts = [
    "Question: What is deep learning? Answer: Deep learning uses neural networks...",
    # ... validation examples
]

# Fine-tune
trainer = finetuner.train(train_texts, val_texts, epochs=3)

# Generate
response = finetuner.generate("Question: What is AI? Answer:")
print(response)
```

---

## Parameter-Efficient Fine-Tuning

### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model, TaskType

class LoRAFineTuner:
    """Fine-tune with LoRA - only train 0.1% of parameters"""

    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,  # 8-bit quantization
            device_map="auto"
        )

        # LoRA config
        lora_config = LoraConfig(
            r=8,  # Rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Which layers to adapt
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Apply LoRA
        self.model = get_peft_model(self.base_model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()
        # Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%

    def train(self, train_dataset, output_dir="./lora_model"):
        """Train with LoRA"""

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            save_total_limit=3,
            logging_steps=10
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )

        trainer.train()

        # Save LoRA weights (very small - ~10MB)
        self.model.save_pretrained(output_dir)

    def load_for_inference(self, lora_weights_path):
        """Load LoRA weights for inference"""
        from peft import PeftModel

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, lora_weights_path)

        return model

# Usage
lora_trainer = LoRAFineTuner(model_name="gpt2")  # or larger model
lora_trainer.train(train_dataset)

# For inference (merge LoRA with base)
model = lora_trainer.load_for_inference("./lora_model")
```

---

### QLoRA (Quantized LoRA)

**4-bit quantization + LoRA = Fine-tune 65B model on single GPU!**

```python
from transformers import BitsAndBytesConfig

class QLoRAFineTuner:
    """QLoRA: 4-bit quantized LoRA"""

    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Load model in 4-bit
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Prepare for LoRA
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

    def train(self, dataset):
        """Train with QLoRA"""

        trainer = Trainer(
            model=self.model,
            train_dataset=dataset,
            args=TrainingArguments(
                output_dir="./qlora_model",
                num_train_epochs=3,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=16,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=10
            )
        )

        trainer.train()

# Fine-tune 70B model on 24GB GPU
qlora = QLoRAFineTuner(model_name="meta-llama/Llama-2-70b-hf")
qlora.train(dataset)
```

---

## Alignment & RLHF

### Supervised Fine-Tuning (SFT)

```python
class InstructionFineTuner:
    """Fine-tune for instruction following"""

    def format_instruction(self, instruction, input_text, output_text):
        """Format as instruction-following example"""

        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""

    def prepare_sft_dataset(self, examples):
        """Prepare instruction dataset"""

        formatted = []
        for example in examples:
            text = self.format_instruction(
                example['instruction'],
                example['input'],
                example['output']
            )
            formatted.append(text)

        return formatted

    def train(self, model, tokenizer, examples):
        """Supervised fine-tuning"""

        # Format dataset
        formatted_texts = self.prepare_sft_dataset(examples)

        # Tokenize
        dataset = tokenizer(
            formatted_texts,
            truncation=True,
            padding="max_length",
            max_length=512
        )

        # Training
        # ... (same as before)

# Usage
sft_trainer = InstructionFineTuner()

examples = [
    {
        'instruction': 'Summarize the following text',
        'input': 'Long article text...',
        'output': 'Summary of the article...'
    },
    # ... more examples
]

sft_trainer.train(model, tokenizer, examples)
```

---

### RLHF (Reinforcement Learning from Human Feedback)

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

class RLHFTrainer:
    """Train with human feedback using PPO"""

    def __init__(self, model_name="gpt2"):
        # Model with value head for RL
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # PPO config
        self.ppo_config = PPOConfig(
            model_name=model_name,
            learning_rate=1e-5,
            batch_size=16,
            mini_batch_size=4
        )

        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            tokenizer=self.tokenizer
        )

    def reward_function(self, response):
        """Reward model - in practice, this is a trained classifier"""

        # Simple heuristic reward (replace with trained reward model)
        rewards = []

        for text in response:
            reward = 0.0

            # Reward for being helpful
            if len(text) > 20:
                reward += 0.5

            # Penalize for being too long
            if len(text) > 200:
                reward -= 0.3

            # Reward for politeness
            if any(word in text.lower() for word in ['please', 'thank', 'sorry']):
                reward += 0.2

            rewards.append(reward)

        return torch.tensor(rewards)

    def train(self, prompts, num_epochs=10):
        """Train with RLHF"""

        for epoch in range(num_epochs):
            for prompt in prompts:
                # Tokenize prompt
                prompt_tensors = self.tokenizer(prompt, return_tensors="pt")['input_ids']

                # Generate response
                response_tensors = self.ppo_trainer.generate(
                    prompt_tensors,
                    max_length=100,
                    temperature=0.7
                )

                # Decode responses
                responses = [self.tokenizer.decode(r) for r in response_tensors]

                # Compute rewards
                rewards = self.reward_function(responses)

                # PPO step
                stats = self.ppo_trainer.step(
                    prompt_tensors,
                    response_tensors,
                    rewards
                )

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Mean reward: {rewards.mean():.2f}")

# Usage
rlhf = RLHFTrainer(model_name="gpt2")

prompts = [
    "How do I learn Python?",
    "What is the capital of France?",
    # ... more prompts
]

rlhf.train(prompts, num_epochs=1000)
```

---

## Production Deployment

### LLM Serving with vLLM

```python
# High-throughput LLM serving
from vllm import LLM, SamplingParams

class ProductionLLMServer:
    """Production LLM inference with vLLM"""

    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", tensor_parallel_size=2):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,  # Multi-GPU
            dtype="float16",
            max_model_len=4096
        )

    def generate(self, prompts, temperature=0.7, max_tokens=512):
        """Batch generation"""

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            results.append({
                'prompt': output.prompt,
                'generated_text': output.outputs[0].text,
                'tokens_generated': len(output.outputs[0].token_ids)
            })

        return results

# Deploy with FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
llm_server = ProductionLLMServer(model_name="gpt2", tensor_parallel_size=1)

class GenerateRequest(BaseModel):
    prompts: list[str]
    temperature: float = 0.7
    max_tokens: int = 512

@app.post("/generate")
async def generate(request: GenerateRequest):
    results = llm_server.generate(
        prompts=request.prompts,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    return {"results": results}

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
```

---

### Prompt Caching & Optimization

```python
from functools import lru_cache
import hashlib

class OptimizedLLMService:
    """LLM service with caching and batching"""

    def __init__(self, model):
        self.model = model
        self.batch_queue = []
        self.batch_size = 8

    @lru_cache(maxsize=1000)
    def cached_generate(self, prompt_hash, prompt, max_tokens):
        """Cache responses for identical prompts"""
        return self.model.generate(prompt, max_tokens=max_tokens)

    def generate(self, prompt, max_tokens=512, use_cache=True):
        """Generate with optional caching"""

        if use_cache:
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            return self.cached_generate(prompt_hash, prompt, max_tokens)

        return self.model.generate(prompt, max_tokens=max_tokens)

    def add_to_batch(self, prompt, callback):
        """Add prompt to batch queue"""
        self.batch_queue.append((prompt, callback))

        if len(self.batch_queue) >= self.batch_size:
            self.process_batch()

    def process_batch(self):
        """Process batched requests"""
        if not self.batch_queue:
            return

        prompts = [p for p, _ in self.batch_queue]
        callbacks = [c for _, c in self.batch_queue]

        # Batch generation
        results = self.model.generate(prompts)

        # Call callbacks
        for result, callback in zip(results, callbacks):
            callback(result)

        self.batch_queue = []
```

---

## Summary

| Technique | Use Case | Cost | Performance |
|-----------|----------|------|-------------|
| **Zero-shot** | Simple tasks | Low | Medium |
| **Few-shot** | Domain-specific | Low | High |
| **Chain-of-Thought** | Reasoning tasks | Medium | Very High |
| **Full Fine-tuning** | Custom models | Very High | Highest |
| **LoRA** | Efficient adaptation | Low | High |
| **QLoRA** | Large models, limited GPU | Very Low | High |
| **RLHF** | Alignment, safety | High | Highest |

---

## Key Takeaways

1. **Start with prompting** before fine-tuning
2. **Few-shot beats zero-shot** for most tasks
3. **Chain-of-Thought** dramatically improves reasoning
4. **LoRA/QLoRA** for efficient fine-tuning (0.1% params)
5. **RLHF** for alignment and safety
6. **Cache and batch** for production efficiency
7. **vLLM** for high-throughput serving

**Best Practices:**
- Test prompts systematically
- Use instruction-tuning format
- Fine-tune on high-quality data (quality > quantity)
- Monitor for hallucinations and bias
- Implement safety filters
- Version control prompts and models
- A/B test different approaches

**Next Steps:**
- Master prompt engineering patterns
- Fine-tune with LoRA on your data
- Implement RLHF for alignment
- Deploy with vLLM for production
- Build evaluation framework
- Create prompt library
