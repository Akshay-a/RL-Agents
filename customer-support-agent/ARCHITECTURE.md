# Architecture: Customer Support AI Agent

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Customer Support Agent                   │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │    Data      │───>│   Training   │───>│  Deployment  │ │
│  │ Preparation  │    │   (GRPO)     │    │  (Inference) │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │         │
│         v                    v                    v         │
│    Training Data      Reward Function         Inference    │
│    (Bitext +          (Multi-Component)        API/CLI     │
│     Synthetic)                                             │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Data Layer (`data_prep.py`)

**Purpose**: Load, clean, and format customer support data for training

```python
┌─────────────────────────────────────────┐
│      CustomerSupportDataPrep            │
├─────────────────────────────────────────┤
│ + load_bitext_dataset()                 │
│ + _create_synthetic_dataset()           │
│ + format_for_training()                 │
│ + create_train_val_test_split()         │
│ + analyze_dataset()                     │
└─────────────────────────────────────────┘
         │
         ├──> Raw Data (Bitext)
         │
         └──> Formatted Data:
              {
                "messages": [...],
                "intent": "...",
                "metadata": {...}
              }
```

**Data Flow:**

1. **Load**: Fetch Bitext dataset from HuggingFace
2. **Parse**: Extract query-response pairs and intent labels
3. **Format**: Convert to conversation format (system, user, assistant)
4. **Enrich**: Add metadata (escalation flags, complexity)
5. **Split**: 70% train, 15% validation, 15% test
6. **Save**: Write JSON files for training

**Key Design Choices:**

- **Conversation Format**: Use ChatML-style messages for compatibility with instruction-tuned models
- **Metadata**: Track additional features (escalation_needed, complexity) for analysis
- **Synthetic Fallback**: Generate basic examples if dataset unavailable

### 2. Reward System (`reward_function.py`)

**Purpose**: Multi-component reward function for evaluating response quality

```
┌──────────────────────────────────────────────────┐
│     CustomerSupportRewardFunction                │
├──────────────────────────────────────────────────┤
│                                                   │
│  calculate_reward(query, response, intent)       │
│         │                                         │
│         ├──> Intent Score        (+5/-2)         │
│         ├──> Policy Score        (+3/-3)         │
│         ├──> Empathy Score       (+2)            │
│         ├──> Tone Score          (+1/-2)         │
│         ├──> Escalation Score    (+1/-1)         │
│         ├──> Safety Score        (-5)            │
│         └──> Resolution Score    (+1-2)          │
│                                                   │
│  Total Reward = Sum(components)                  │
│  Clipped to [-10, +10]                           │
└──────────────────────────────────────────────────┘
```

**Component Breakdown:**

1. **Intent Classification** (±5)
   - Checks if response addresses correct intent
   - Uses keyword matching (production would use classifier)

2. **Policy Compliance** (+3/-3)
   - Regex patterns for policy violations
   - Examples: unauthorized promises, data sharing

3. **Empathy** (+2)
   - Counts empathy indicators: "I understand", "I apologize"
   - Rewards customer-centric language

4. **Tone** (+1/-2)
   - Professional: "I'd be happy", "May I"
   - Inappropriate: "whatever", "lol"

5. **Escalation** (+1/-1)
   - Detects if customer wants escalation
   - Checks if agent appropriately offers it

6. **Safety** (-5)
   - Heavy penalty for dangerous advice
   - Examples: "ignore policy", "lie", "hack"

7. **Resolution** (+1-2)
   - Bonus for actionable next steps
   - Bonus for proactive suggestions

**Reward Propagation:**

```
Query + Response
      │
      v
┌─────────────────┐
│ Reward Function │
│   (7 checks)    │
└─────────────────┘
      │
      ├──> Reward Value (float)
      └──> Component Breakdown (dict)
            │
            └──> Used for:
                 - Training signal
                 - Debugging
                 - Analysis
```

### 3. Environment Layer (`support_env.py`)

**Purpose**: Gymnasium-compatible RL environment

```
┌────────────────────────────────────────────────┐
│       CustomerSupportEnv (gym.Env)             │
├────────────────────────────────────────────────┤
│                                                 │
│  Observation Space:                            │
│    - query: Text (customer message)            │
│    - intent: Discrete (intent category)        │
│    - context: Text (metadata)                  │
│                                                 │
│  Action Space:                                 │
│    - response: Text (agent message)            │
│                                                 │
│  Reward:                                       │
│    - Float from reward function                │
│                                                 │
└────────────────────────────────────────────────┘
```

**Episode Flow:**

```
reset()
  │
  ├──> Sample random example from dataset
  ├──> Extract customer query
  └──> Return observation: {query, intent, context}
       │
       v
step(response)
  │
  ├──> Calculate reward for response
  ├──> Check if episode done (single-turn)
  └──> Return (next_obs, reward, done, info)
       │
       └──> info contains:
            - reward_components
            - ground_truth_response
            - query and response
```

**Vectorized Environment:**

For parallel training, `CustomerSupportVectorEnv` manages multiple environments:

```
┌────────────────────────────────────────┐
│  CustomerSupportVectorEnv              │
│                                         │
│  Env 1 ──┐                             │
│  Env 2 ──┼──> batch_reset()            │
│  Env 3 ──┤                             │
│  Env 4 ──┘    batch_step(actions)      │
│                                         │
│           Returns lists of obs/rewards │
└────────────────────────────────────────┘
```

### 4. Training System (`train_grpo.py`)

**Purpose**: GRPO training loop with Unsloth optimization

```
┌─────────────────────────────────────────────────┐
│              GRPOTrainer                         │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. Load Model (with Unsloth)                   │
│     ├──> LLaMA 3.2-3B-Instruct                  │
│     ├──> 4-bit quantization                     │
│     └──> LoRA adapters (r=16)                   │
│                                                  │
│  2. Training Loop:                              │
│     For each epoch:                             │
│       For each batch:                           │
│         ├──> collect_rollouts()                 │
│         ├──> compute_grpo_advantages()          │
│         └──> train_step()                       │
│                                                  │
│  3. Evaluation:                                 │
│     Every N steps:                              │
│       ├──> Generate on validation set           │
│       ├──> Calculate rewards                    │
│       └──> Save if best                         │
│                                                  │
└─────────────────────────────────────────────────┘
```

**GRPO Algorithm:**

```
For each training iteration:

1. Collect Rollouts:
   ┌──────────────────────────┐
   │ Sample batch_size queries│
   │ Generate responses       │
   │ Calculate rewards        │
   └──────────────────────────┘
          │
          v
2. Compute Group Advantages:
   ┌──────────────────────────┐
   │ Group rewards (size=4)   │
   │ Advantage = reward - mean│
   │ Normalize advantages     │
   └──────────────────────────┘
          │
          v
3. Policy Gradient Update:
   ┌──────────────────────────┐
   │ Loss = -advantage * logP │
   │ Add KL regularization    │
   │ Backward + optimize      │
   └──────────────────────────┘
```

**Why GRPO?**

Traditional policy gradient:
```
Loss = -reward * log P(action|state)
```

GRPO (group-relative):
```
Loss = -(reward - group_mean) * log P(action|state)
```

Benefits:
- **Relative Comparison**: Compares within group, not global
- **Better Exploration**: Encourages diversity
- **More Stable**: Advantage normalization reduces variance

**Unsloth Integration:**

```python
# Standard (slow):
model = AutoModelForCausalLM.from_pretrained(...)

# With Unsloth (2x faster):
model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing="unsloth"  # Optimized
)
```

Optimizations:
- Custom CUDA kernels for attention
- Memory-efficient gradient checkpointing
- Optimized LoRA implementation
- Fast tokenization

### 5. Evaluation System (`evaluate.py`)

**Purpose**: Comprehensive model evaluation

```
┌─────────────────────────────────────────┐
│     CustomerSupportEvaluator            │
├─────────────────────────────────────────┤
│                                          │
│  1. Intent Classification Accuracy      │
│     ├──> Generate responses             │
│     └──> Check if addresses intent      │
│                                          │
│  2. Policy Compliance Rate              │
│     ├──> Check for violations           │
│     └──> % compliant                    │
│                                          │
│  3. Escalation Metrics                  │
│     ├──> Precision: correct escalations │
│     └──> Recall: missed escalations     │
│                                          │
│  4. Response Quality (Mean Reward)      │
│     └──> Average reward on test set     │
│                                          │
│  5. Edge Cases                          │
│     └──> Test on adversarial examples   │
│                                          │
└─────────────────────────────────────────┘
```

**Metrics Flow:**

```
Test Dataset
     │
     ├──> For each example:
     │      ├──> Generate response
     │      ├──> Calculate reward
     │      └──> Evaluate metrics
     │
     v
┌─────────────────────────┐
│   Evaluation Results    │
├─────────────────────────┤
│ Intent Accuracy: 85%    │
│ Policy Compliance: 96%  │
│ Mean Reward: +3.2       │
│ Escalation F1: 0.78     │
└─────────────────────────┘
```

### 6. Inference Layer (`inference.py`)

**Purpose**: Deploy trained model for inference

```
┌──────────────────────────────────────┐
│    CustomerSupportAgent              │
├──────────────────────────────────────┤
│                                       │
│  respond(query) -> response           │
│     │                                 │
│     ├──> Format prompt               │
│     ├──> Tokenize                    │
│     ├──> Generate (with sampling)    │
│     └──> Extract response            │
│                                       │
└──────────────────────────────────────┘
         │
         ├──> CLI: Interactive chat
         ├──> API: FastAPI server
         └──> Batch: Process multiple
```

**Deployment Modes:**

1. **Interactive Chat:**
   ```
   User: I want a refund
   Agent: [generates response]
   ```

2. **API Server:**
   ```
   POST /chat
   {
     "query": "I want a refund",
     "temperature": 0.7
   }

   Response:
   {
     "response": "...",
     "query": "..."
   }
   ```

3. **Batch Processing:**
   ```python
   queries = [...]
   responses = agent.batch_respond(queries)
   ```

## Data Flow End-to-End

```
┌──────────────┐
│ Raw Data     │
│ (Bitext)     │
└──────┬───────┘
       │
       v
┌──────────────┐
│ data_prep.py │ ──> train.json, val.json, test.json
└──────┬───────┘
       │
       v
┌──────────────┐
│ train_grpo.py│ <──> reward_function.py
│              │ <──> support_env.py
└──────┬───────┘
       │
       ├──> Checkpoints/
       │
       v
┌──────────────┐
│ evaluate.py  │ ──> evaluation_results.json
└──────────────┘
       │
       v
┌──────────────┐
│ inference.py │ ──> Production deployment
└──────────────┘
```

## Configuration System (`config.py`)

Centralized configuration using dataclasses:

```python
@dataclass
class ModelConfig:
    base_model: str
    lora_r: int
    lora_alpha: int
    # ...

@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    grpo_group_size: int
    # ...

@dataclass
class RewardConfig:
    intent_classification_correct: float
    policy_violation: float
    # ...
```

**Benefits:**
- Single source of truth
- Type checking
- Easy to modify
- Version control friendly

## Memory and Compute Optimization

### 1. Model Loading

```
4-bit Quantization
├──> Reduces model size 4x
├──> Maintains 95%+ quality
└──> Enables larger batch sizes

LoRA Adapters
├──> Only train 0.1% of parameters
├──> Faster training
└──> Less memory
```

### 2. Gradient Checkpointing

```
Standard:
├──> Store all activations
└──> High memory, fast backward

Unsloth Gradient Checkpointing:
├──> Recompute activations
├──> Low memory, slightly slower
└──> 40% memory savings
```

### 3. Batch Processing

```
Gradient Accumulation (4 steps)
├──> Effective batch size = 16
├──> Physical batch size = 4
└──> Fits in GPU memory
```

## Error Handling and Fallbacks

```
┌─────────────────────────┐
│ Try: Load with Unsloth  │
└───────┬─────────────────┘
        │
        ├─> Success: 2x faster training
        │
        └─> Failure:
            ├──> Fallback to transformers
            └──> Continue (slower)

┌─────────────────────────┐
│ Try: Load Bitext data   │
└───────┬─────────────────┘
        │
        ├─> Success: Use real data
        │
        └─> Failure:
            ├──> Use synthetic data
            └──> Warn user
```

## Monitoring and Logging

During training:
```
Every N steps:
├──> Log: loss, reward, KL divergence
├──> Evaluate: validation reward
└──> Save: best checkpoint

Tracked Metrics:
├──> Training reward (per batch)
├──> Validation reward (per epoch)
├──> Loss curve
└──> KL divergence (prevent drift)
```

## Scalability Considerations

**Current**: Single GPU, small batch

**To Scale**:

1. **Data Parallel**: Multi-GPU training
   ```python
   model = torch.nn.DataParallel(model)
   ```

2. **Larger Batches**: Increase batch size with more GPUs
   ```python
   batch_size = 4 * num_gpus
   ```

3. **Distributed Training**: Multiple machines
   ```python
   torch.distributed.launch(...)
   ```

4. **Serve at Scale**:
   - Load balancer in front of API
   - Multiple inference workers
   - Model caching
   - Batch inference requests

## Testing Strategy

1. **Unit Tests**: Each component independently
   - Reward function with known examples
   - Environment reset/step
   - Data loading

2. **Integration Tests**: Components together
   - Training loop (1 epoch)
   - Evaluation pipeline
   - End-to-end generation

3. **Edge Case Tests**: Adversarial examples
   - Angry customers
   - Policy violations
   - Ambiguous queries

## Summary

This architecture prioritizes:
- **Modularity**: Each component is independent
- **Simplicity**: KISS principle throughout
- **Debuggability**: Clear data flow, interpretable rewards
- **Production-Ready**: Error handling, monitoring, deployment modes

The design allows easy modification of any component without affecting others, making it maintainable and extensible.
