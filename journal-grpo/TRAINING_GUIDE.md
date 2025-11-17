# Training Guide: Understanding Fine-Tuning & GRPO

*A practical guide to fine-tuning language models with emphasis on fundamentals*

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Two-Stage Training Strategy](#two-stage-training)
3. [Understanding LoRA](#lora)
4. [Understanding Quantization](#quantization)
5. [Training Process Walkthrough](#training-walkthrough)
6. [GRPO Fundamentals](#grpo-fundamentals)
7. [Troubleshooting](#troubleshooting)

---

## Core Concepts {#core-concepts}

### What is Fine-Tuning?

**Basic Idea:** Take a pre-trained model and adapt it to your specific task.

```
Base Model (Qwen 2.5B) → Fine-Tuning → Specialized Model (Journal Entries)
   ↑                                           ↑
Knows general language              Knows accounting & constraints
```

**Analogy:** It's like hiring a smart generalist and training them for a specific job.

### Why Not Just Use Prompts?

| Approach | Pros | Cons |
|----------|------|------|
| **Prompting** | No training needed | Inconsistent, expensive tokens, no constraint enforcement |
| **Fine-Tuning** | Consistent, fast, learns constraints | Requires training time & data |

For our accounting use case, we need **reliability** → fine-tuning wins.

---

## Two-Stage Training Strategy {#two-stage-training}

### Stage 1: Supervised Fine-Tuning (SFT)

**Goal:** Teach the model what journal entries look like.

```python
Input:  "Sold services for $5,000 cash"
Output: {
  "date": "2024-01-15",
  "entries": [
    {"account": "Cash", "debit": 5000, "credit": 0},
    {"account": "Revenue", "debit": 0, "credit": 5000}
  ]
}
```

**Method:** Standard supervised learning
- Show model 500 examples
- Model learns to mimic the pattern
- Like learning from a textbook

**Expected Result:** ~70% balance accuracy

### Stage 2: GRPO (Reinforcement Learning)

**Goal:** Teach the model hard constraints (debits must equal credits).

```python
For each prompt:
  1. Generate 4 different answers
  2. Score each with reward model:
     - Answer 1: Balanced ✓ → +2.0
     - Answer 2: Unbalanced ✗ → -1.0
     - Answer 3: Invalid codes → +0.5
     - Answer 4: Balanced ✓ → +2.0

  3. Update model to prefer high-scoring answers
```

**Method:** Reinforcement learning
- Model explores different outputs
- Gets feedback (rewards)
- Learns to maximize reward
- Like learning from practice with a coach

**Expected Result:** >90% balance accuracy

### Why This Two-Stage Approach?

```
Stage 1 (SFT): "Get in the ballpark"
└─> Without this, RL takes forever to learn basic structure

Stage 2 (GRPO): "Nail the constraints"
└─> Refines outputs to satisfy hard requirements
```

**Analogy:**
- SFT = Learning to play basketball by watching videos
- GRPO = Practicing shots and getting feedback from coach

---

## Understanding LoRA {#lora}

### The Problem with Full Fine-Tuning

**Full Fine-Tuning:**
- Update all 1.5 billion parameters
- Requires 60GB+ VRAM
- Takes 12+ hours
- Risk of catastrophic forgetting

**LoRA (Low-Rank Adaptation):**
- Update only ~1% of parameters
- Requires 6GB VRAM
- Takes 2 hours
- Preserves base model knowledge

### How LoRA Works

Instead of updating the entire model, LoRA adds small "adapter" layers:

```
Original Model:        W (1024 × 1024 matrix)
                       ↓
LoRA Adds:            W + (A × B)
                            ↑
                    (1024×16) × (16×1024)
```

**Key Insight:** Most information in fine-tuning can be captured in a low-rank matrix!

### LoRA Parameters Explained

```yaml
lora:
  r: 16                  # Rank (dimensionality of adapters)
  lora_alpha: 32        # Scaling factor
  lora_dropout: 0.05    # Prevent overfitting
  target_modules:       # Which layers to adapt
    - q_proj            # Query projection (attention)
    - v_proj            # Value projection (attention)
    - k_proj            # Key projection (attention)
    - o_proj            # Output projection (attention)
```

**r (Rank):**
- Higher r = more capacity, but slower and more memory
- r=8: Quick experiments
- r=16: Standard (what we use)
- r=32: Complex tasks
- r=64: Usually overkill

**lora_alpha:**
- Controls how much LoRA influences the model
- Typically set to 2×r
- Higher = more aggressive adaptation

**target_modules:**
- Which layers get adapters
- We target attention layers (q, k, v, o)
- Can also target feedforward layers (up_proj, down_proj, gate_proj)

### LoRA Visual

```
┌─────────────────────────────────────┐
│       Original Model Weights        │
│     (frozen, not updated)          │
└─────────────────────────────────────┘
              │
              ├─────────────────────┐
              │                     │
              ▼                     ▼
         Original              ┌─────────┐
          Output               │  LoRA   │
                              │ Adapter │
                              │ (tiny!) │
                              └─────────┘
                                   │
                                   ▼
                               LoRA Output
                                   │
                                   ▼
                           Combined Output
```

---

## Understanding Quantization {#quantization}

### What is Quantization?

**Idea:** Store numbers with fewer bits.

```
Normal (Float16):    16 bits per number → 2 bytes
Quantized (4-bit):   4 bits per number  → 0.5 bytes

Result: 4x memory reduction!
```

### How Does 4-bit Work?

**NF4 (Normalized Float 4):**
- Special encoding optimized for neural network weights
- Weights typically follow normal distribution
- NF4 has more precision near zero (where most weights are)

```
Float16 Precision:
-∞ ──────────────────────────────────────── +∞
     ╲                                  ╱
      ╲     (65,536 possible values)   ╱
       ╲                              ╱

4-bit (NF4):
-∞ ──────────────────────────────────────── +∞
       ╲╱╲╱╲╱╲╱       ╲╱╲╱╲╱╲╱
    (16 values, denser near 0)
```

### Quality vs Memory Trade-off

| Precision | Memory | Quality | Use Case |
|-----------|--------|---------|----------|
| **Float32** | 100% | 100% | Training from scratch |
| **Float16** | 50% | ~99.9% | Standard fine-tuning |
| **8-bit** | 25% | ~99% | Budget GPUs |
| **4-bit (NF4)** | 12.5% | ~95% | Consumer GPUs (our choice!) |

**For fine-tuning:** 4-bit is usually good enough!

### Double Quantization

```yaml
bnb_4bit_use_double_quant: true
```

This quantizes the quantization constants themselves!
- Extra 5-10% memory savings
- Negligible quality loss

### Practical Impact

**Qwen 2.5B Model Size:**

```
Without Quantization:
  1.5B params × 2 bytes (float16) = 3GB weights
  + activations + optimizer states = 12GB total

With 4-bit Quantization:
  1.5B params × 0.5 bytes (4-bit) = 0.75GB weights
  + activations + optimizer states = 6GB total

Result: Fits on RTX 3070! (8GB VRAM)
```

---

## Training Process Walkthrough {#training-walkthrough}

### Step-by-Step: What Happens During Training

#### 1. **Load Base Model**

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
```

- Downloads ~3GB model from HuggingFace
- Loads into VRAM
- Model has 1.5B parameters

#### 2. **Apply Quantization**

```python
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
```

- Converts weights from 16-bit → 4-bit
- 3GB → 0.75GB
- Happens during loading

#### 3. **Add LoRA Adapters**

```python
lora_config = LoraConfig(r=16, ...)
model = get_peft_model(model, lora_config)
```

- Adds small adapter matrices
- Only ~16M new parameters (1% of model)
- These are the ONLY parameters we'll update

#### 4. **Training Loop**

```python
for epoch in range(3):
    for batch in dataloader:
        # Forward pass
        outputs = model(batch)
        loss = compute_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # Update (only LoRA parameters!)
        optimizer.step()
```

**Each iteration:**
1. Model predicts next tokens
2. Compare to actual tokens → compute loss
3. Backpropagate gradients
4. Update LoRA parameters

#### 5. **Gradient Accumulation**

```yaml
per_device_batch_size: 4
gradient_accumulation_steps: 4
```

**Why?** Simulates larger batch size without more memory.

```
Regular (batch=16):
  Process 16 examples → Update weights
  Memory: High

Gradient Accumulation (batch=4, accum=4):
  Process 4 examples → Save gradients
  Process 4 examples → Add to gradients
  Process 4 examples → Add to gradients
  Process 4 examples → Add to gradients
  Update weights with accumulated gradients
  Memory: Lower, same effect!
```

#### 6. **Learning Rate Schedule**

```python
lr_scheduler_type: "cosine"
warmup_steps: 100
```

```
Learning Rate Over Time:

High  │    ╱────╲
      │   ╱      ╲___
      │  ╱           ╲___
Low   │ ╱                ╲___
      └─────────────────────────→ Steps
       Warmup   Peak    Decay
```

- **Warmup:** Gradually increase LR (prevent early instability)
- **Peak:** Train at full speed
- **Decay:** Reduce LR for fine-grained adjustments

---

## GRPO Fundamentals {#grpo-fundamentals}

### What is GRPO?

**Group Relative Policy Optimization** - A reinforcement learning method for language models.

**Core Idea:** Learn by comparing multiple outputs.

### The GRPO Training Loop

```python
for prompt in training_prompts:
    # 1. GENERATE multiple candidates
    candidates = []
    for i in range(4):  # num_samples = 4
        output = model.generate(prompt, temperature=0.8)
        candidates.append(output)

    # 2. SCORE each candidate
    rewards = []
    for candidate in candidates:
        score = reward_model(candidate)  # Our custom reward function!
        rewards.append(score)

    # Example scores: [2.0, -1.0, 1.5, 0.8]

    # 3. COMPUTE loss
    # Higher reward candidates get upweighted
    # Lower reward candidates get downweighted
    loss = compute_grpo_loss(candidates, rewards)

    # 4. UPDATE model
    loss.backward()
    optimizer.step()
```

### Why Multiple Candidates?

**Traditional Supervised Learning:**
```
Model sees: "This is correct" ✓
Model doesn't see: What is INcorrect ✗
```

**GRPO:**
```
Model generates 4 different answers:
1. Balanced entry: +2.0 ← "Do more of this!"
2. Unbalanced: -1.0 ← "Avoid this!"
3. Invalid codes: +0.5 ← "This is slightly better than nothing"
4. Balanced entry: +2.0 ← "Yes, like this!"

Model learns: What makes answers good vs bad
```

### The Reward Function

Our reward model scores on 4 dimensions:

```python
def compute_reward(entry):
    # 1. Balance (most important!)
    if debits == credits:
        balance_score = +1.0
    else:
        balance_score = -1.0

    # 2. Valid account codes
    valid_ratio = num_valid_codes / total_codes
    account_score = valid_ratio * 0.5

    # 3. Schema compliance
    schema_score = 0.3 if valid_schema else 0.0

    # 4. Reasonable amounts
    amount_score = 0.2 if amounts_ok else 0.0

    return balance_score + account_score + schema_score + amount_score
```

**Range:** -1.0 to +2.0

### Key GRPO Parameters

#### **Temperature**

Controls diversity of generations:

```python
temperature = 0.1   # Almost deterministic (same output)
temperature = 0.7   # Moderate diversity
temperature = 1.0   # High diversity
temperature = 2.0   # Very random
```

**For GRPO:** We use 0.8
- Diverse enough to explore
- Not so random that it generates garbage

#### **KL Penalty**

Prevents model from drifting too far from base model:

```python
kl_penalty = 0.05  # Allow 5% deviation
```

**What is KL Divergence?**
- Measures how different two probability distributions are
- In our case: new model vs base model

```
Low KL (good): Model is similar to base, makes careful improvements
High KL (bad): Model has diverged, might forget general knowledge
```

**Why constrain?**
- Base model has useful knowledge (grammar, facts, reasoning)
- We want to adapt, not completely rewrite
- Prevents "reward hacking" (gaming the reward function)

#### **Number of Samples**

How many candidates to generate per prompt:

```python
num_samples = 4  # Generate 4 different answers
```

**Trade-off:**
- More samples = better learning signal, but slower
- Fewer samples = faster, but noisier gradients

**Rule of thumb:**
- 2 samples: Minimum
- 4 samples: Standard (our choice)
- 8 samples: If you have compute budget

### GRPO vs Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **SFT** | Simple, fast | Can't enforce constraints |
| **PPO** | Stable, proven | Complex (needs value network) |
| **DPO** | No reward model | Needs preference data (A > B) |
| **GRPO** | Simple, uses reward fn | Newer, less tested |

**Our choice:** GRPO/PPO hybrid
- Uses reward function (easy to define)
- Stable training
- No need for preference data

---

## Training Scripts Overview

### 1. `train_sft.py` - Supervised Fine-Tuning

**What it does:**
- Loads Qwen 2.5B with 4-bit quantization
- Adds LoRA adapters
- Trains on 500 labeled examples
- Saves checkpoint

**Usage:**
```bash
# First, generate training data
python data/generate_sft.py

# Then train
python src/train_sft.py
```

**Time:** ~2 hours on RTX 3070

**Output:** `checkpoints/sft/final/`

### 2. `train_grpo.py` - GRPO Refinement

**What it does:**
- Loads SFT checkpoint
- Generates multiple candidates per prompt
- Scores with reward model
- Updates using RL

**Usage:**
```bash
# Generate prompts
python data/generate_grpo_candidates.py

# Train
python src/train_grpo.py
```

**Time:** ~4 hours on RTX 3070

**Output:** `checkpoints/grpo/final/`

### 3. `train_with_unsloth.py` - Fast Training

**What it does:**
- Same as train_sft.py but 2-5x faster
- Uses Unsloth's optimized kernels
- Standalone script (easy to understand)

**Usage:**
```bash
python src/train_with_unsloth.py
```

**Time:** ~1 hour on RTX 3070 (2x faster!)

**Why Unsloth?**
- Optimized Flash Attention 2
- Better memory management
- Same quality, much faster

### 4. `evaluate.py` - Model Testing

**What it does:**
- Loads trained models
- Tests on 50 golden examples
- Computes metrics
- Compares base vs SFT vs GRPO

**Usage:**
```bash
python src/evaluate.py
```

**Metrics:**
- Balance accuracy
- Average reward score
- Schema compliance
- Account code accuracy

---

## Troubleshooting {#troubleshooting}

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**

1. **Reduce batch size:**
```yaml
per_device_batch_size: 2  # Down from 4
```

2. **Reduce sequence length:**
```yaml
max_seq_length: 512  # Down from 1024
```

3. **Enable gradient checkpointing:**
```python
use_gradient_checkpointing=True
```

4. **Use Unsloth (automatic optimization):**
```bash
python src/train_with_unsloth.py
```

### Model Generates Garbage

**Problem:** Output is nonsense or doesn't follow format.

**Causes & Solutions:**

1. **Learning rate too high:**
```yaml
learning_rate: 1e-4  # Down from 2e-4
```

2. **LoRA rank too low:**
```yaml
lora_r: 32  # Up from 16
```

3. **Need more training:**
```yaml
num_epochs: 5  # Up from 3
```

4. **Bad training data:**
```bash
# Check data quality
python -c "
from reward_model import JournalEntryRewardModel
reward_model = JournalEntryRewardModel()
# Score your training examples
"
```

### Training is Too Slow

**Solutions:**

1. **Use Unsloth (2-5x speedup):**
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python src/train_with_unsloth.py
```

2. **Enable Flash Attention:**
```python
# Automatically enabled in newer transformers
```

3. **Use bf16 if supported:**
```yaml
bf16: true
fp16: false
```

4. **Reduce logging:**
```yaml
logging_steps: 50  # Up from 10
```

### GRPO Not Improving Balance Accuracy

**Problem:** After GRPO, model still generates unbalanced entries.

**Solutions:**

1. **Lower KL penalty:**
```yaml
kl_penalty: 0.01  # Down from 0.05 (allow more change)
```

2. **Increase learning rate:**
```yaml
learning_rate: 1e-4  # Up from 5e-5
```

3. **Higher temperature:**
```yaml
temperature: 1.0  # Up from 0.8 (more exploration)
```

4. **Check reward function:**
```python
# Make sure balance has strong signal
balance_score = ±1.0  # Make this dominant
```

### Model Overfits to Training Data

**Signs:**
- Training loss goes down
- Validation loss goes up
- Model generates similar entries for different prompts

**Solutions:**

1. **Increase LoRA dropout:**
```yaml
lora_dropout: 0.1  # Up from 0.05
```

2. **Reduce epochs:**
```yaml
num_epochs: 2  # Down from 3
```

3. **Add weight decay:**
```yaml
weight_decay: 0.01
```

4. **Use early stopping:**
```python
early_stopping_patience = 3
```

---

## Quick Reference

### Recommended Settings for Different GPUs

#### RTX 3070 (8GB)
```yaml
per_device_batch_size: 4
gradient_accumulation_steps: 4
max_seq_length: 1024
load_in_4bit: true
lora_r: 16
```

#### RTX 3060 (6GB)
```yaml
per_device_batch_size: 2
gradient_accumulation_steps: 8
max_seq_length: 512
load_in_4bit: true
lora_r: 8
```

#### RTX 4090 (24GB)
```yaml
per_device_batch_size: 8
gradient_accumulation_steps: 2
max_seq_length: 2048
load_in_4bit: false  # Can use fp16!
lora_r: 32
```

#### Google Colab (Free T4)
```yaml
per_device_batch_size: 2
gradient_accumulation_steps: 8
max_seq_length: 512
load_in_4bit: true
lora_r: 16
use_unsloth: true  # Highly recommended!
```

---

## Further Reading

### Papers
- **LoRA:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **QLoRA:** "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- **GRPO:** "Group Relative Policy Optimization for Language Models" (2024)
- **PPO:** "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

### Tools
- **HuggingFace Transformers:** https://huggingface.co/docs/transformers
- **PEFT (LoRA):** https://huggingface.co/docs/peft
- **TRL (RL training):** https://huggingface.co/docs/trl
- **Unsloth:** https://github.com/unslothai/unsloth
- **BitsAndBytes (quantization):** https://github.com/TimDettmers/bitsandbytes

### Community
- HuggingFace Discord
- r/LocalLLaMA
- Unsloth Discord

---

**Last Updated:** 2024-11-17
**For:** Journal Entry GRPO Project
