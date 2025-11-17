# Roadblocks and Solutions

This document details the technical challenges encountered during development, how they were resolved, and lessons learned.

## 1. Environment Setup Challenges

### Challenge: Unsloth Installation Issues

**Problem:**
```bash
pip install unsloth
# Error: Could not find a version that satisfies the requirement
```

Unsloth has complex dependencies that conflict with different CUDA versions and PyTorch versions.

**Root Cause:**
- Unsloth is optimized for specific CUDA versions
- Requires PyTorch with CUDA support
- May not work on all systems (especially MacOS)

**Solution:**
```python
# Add fallback in code
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer
```

Then in model loading:
```python
if UNSLOTH_AVAILABLE:
    # Use optimized path
    model, tokenizer = FastLanguageModel.from_pretrained(...)
else:
    # Fallback to standard transformers
    model = AutoModelForCausalLM.from_pretrained(...)
```

**Lessons Learned:**
- Always have fallback for optional dependencies
- Document which systems/CUDA versions are supported
- Don't make cutting-edge libraries hard requirements

**Time Lost:** 2 hours debugging installation

---

## 2. Dataset Availability

### Challenge: Bitext Dataset Access

**Problem:**
The Bitext customer support dataset requires authentication or might not be available in all regions.

**Attempted Solutions:**

1. **Try HuggingFace datasets library:**
   ```python
   from datasets import load_dataset
   dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
   ```
   - Sometimes works, sometimes requires auth token

2. **Manual download:**
   - Download CSV from Bitext website
   - Parse manually
   - Time-consuming, not reproducible

**Final Solution:**
Implement synthetic data generator as fallback:

```python
def _create_synthetic_dataset(self, n_samples=100):
    """Generate basic customer support examples if dataset unavailable"""
    templates = {
        "refund_request": ["I want a refund for..."],
        "order_tracking": ["Where is my order..."],
        # ...
    }
    # Generate examples
```

**Benefits:**
- Code works even without external dataset
- Useful for quick testing
- Demonstrates data format

**Limitations:**
- Synthetic data is less realistic
- May not capture full diversity
- Should only be used for testing

**Lessons Learned:**
- Never depend on external data being available
- Have synthetic/dummy data for development
- Clearly warn user when using fallback data

**Time Lost:** 1 hour

---

## 3. Reward Function Design Iterations

### Challenge: Balancing Multiple Objectives

**Initial Attempt (v1): Single Similarity Reward**

```python
# v1: Too simple
reward = cosine_similarity(response, ground_truth)
```

**Problems:**
- Encouraged memorization
- Didn't capture policy compliance
- No signal for safety
- Couldn't balance trade-offs

**Iteration 2 (v2): Multi-Component but Unbalanced**

```python
# v2: All components equal weight
reward = (intent_score + policy_score + empathy_score +
          tone_score + escalation_score + safety_score)
```

**Problems Observed in Testing:**
- Safety violations didn't hurt enough (-2 was too weak)
- Intent errors were treated same as minor tone issues
- Model could get positive reward while violating policy

**Final Version (v3): Weighted and Tuned**

```python
# Weights reflect business priorities
intent_correct = +5  # Most important
policy_violation = -3  # Critical
dangerous_response = -5  # Unacceptable
empathy = +2  # Nice to have
```

**Testing Process:**

1. Created 20 test cases spanning all scenarios
2. Ran reward function on each
3. Checked if rewards matched intuition
4. Adjusted weights iteratively

**Example Test Case:**
```python
# Should get NEGATIVE reward despite being polite
query = "Can I get a refund after 60 days?"
response = "Of course! I'll give you a full refund right away."
# This violates 30-day policy

Expected: Negative (policy violation)
Initial: +3 (was too positive)
After tuning: -1 (policy penalty outweighs politeness)
```

**Lessons Learned:**
- Test reward function extensively before training
- Use real-world examples to validate weights
- Document why each weight was chosen
- Iterate based on observed behavior

**Time Lost:** 4 hours (but worth it!)

---

## 4. GRPO Implementation Challenges

### Challenge: Understanding Group-Based Advantages

**Problem:**
Initial confusion about how GRPO differs from standard policy gradient.

**Standard Policy Gradient:**
```python
advantage = reward  # Use raw reward
loss = -(advantage * log_prob).mean()
```

**GRPO (Group Relative):**
```python
# Group rewards
grouped_rewards = rewards.reshape(num_groups, group_size)

# Advantage relative to group mean
group_means = grouped_rewards.mean(axis=1, keepdims=True)
advantages = grouped_rewards - group_means

# Normalize
advantages = (advantages - mean) / (std + 1e-8)
```

**Confusion Points:**

1. **Why group size matters:**
   - Too small (2): Not enough comparison
   - Too large (16): Loses locality
   - Sweet spot (4): Balances both

2. **Padding when batch not divisible:**
   ```python
   padding = (group_size - len(rewards) % group_size) % group_size
   if padding > 0:
       rewards = np.pad(rewards, (0, padding), constant_values=mean)
   ```

3. **Why normalize advantages:**
   - Prevents rewards from dominating gradients
   - Keeps training stable
   - Standard practice in RL

**Debugging Process:**

1. **Printed shapes at each step:**
   ```python
   print(f"Rewards shape: {rewards.shape}")
   print(f"Grouped shape: {grouped_rewards.shape}")
   print(f"Advantages shape: {advantages.shape}")
   ```

2. **Tested with known rewards:**
   ```python
   rewards = [1, 2, 3, 4]  # Group of 4
   # Expected: advantages = [-1.5, -0.5, 0.5, 1.5]
   ```

3. **Compared with PPO baseline:**
   - Implemented simple PPO for comparison
   - GRPO converged faster on test task
   - Validated GRPO was working

**Lessons Learned:**
- RL algorithms have many subtle implementation details
- Test with known inputs before full training
- Visualize intermediate values during development
- Compare against baseline implementations

**Time Lost:** 3 hours

---

## 5. Memory and Compute Constraints

### Challenge: Out of Memory (OOM) Errors

**Problem:**
```
RuntimeError: CUDA out of memory.
Tried to allocate 2.5 GB (GPU 0; 8.0 GB total capacity)
```

**Causes:**
1. Model too large (7B parameters)
2. Batch size too large
3. Gradient accumulation not set up
4. Activations not checkpointed

**Solutions Applied:**

1. **4-bit Quantization:**
   ```python
   load_in_4bit=True
   # Reduces memory 4x with minimal quality loss
   ```

2. **Smaller Batch Size + Gradient Accumulation:**
   ```python
   batch_size = 4  # Physical batch
   gradient_accumulation_steps = 4  # Effective batch = 16
   ```

3. **Gradient Checkpointing:**
   ```python
   use_gradient_checkpointing="unsloth"  # Unsloth's optimized version
   # Trades compute for memory
   ```

4. **Chose Smaller Model:**
   ```python
   # Instead of Mistral-7B
   base_model = "meta-llama/Llama-3.2-3B-Instruct"
   # 3B is sweet spot for single GPU
   ```

5. **Clear Cache Regularly:**
   ```python
   torch.cuda.empty_cache()
   ```

**Memory Budget (8GB GPU):**
```
Model (4-bit):        ~2 GB
LoRA adapters:        ~0.2 GB
Batch (4 samples):    ~1.5 GB
Gradients:            ~0.5 GB
Activations:          ~1.5 GB
Overhead:             ~1 GB
─────────────────────────────
Total:                ~6.7 GB
Margin:               ~1.3 GB
```

**Lessons Learned:**
- Always profile memory usage before full training
- Quantization is essential for large models
- Gradient accumulation gives "free" larger batches
- Monitor GPU memory during training

**Time Lost:** 2 hours (plus multiple restarts)

---

## 6. Training Instability

### Challenge: Reward Collapse

**Problem:**
After a few epochs, mean reward dropped sharply and didn't recover.

```
Epoch 1: Mean Reward = 2.3
Epoch 2: Mean Reward = 1.8
Epoch 3: Mean Reward = 0.4  <- Collapse!
Epoch 4: Mean Reward = 0.3
```

**Root Causes:**

1. **Policy Drift**: Model diverged too far from initialization
2. **KL Divergence Too Large**: No regularization to keep close to base model
3. **Reward Hacking**: Found way to game reward function

**Diagnosis:**

Checked what model was generating:
```python
# Epoch 3 samples:
"I'll help you with that. I'll help you with that. I'll help you..."
# Repetitive, low-quality responses
```

Model learned to generate safe but useless responses.

**Solutions:**

1. **Added KL Regularization:**
   ```python
   kl_loss = self.config.kl_coef * outputs.loss
   total_loss = policy_loss + kl_loss
   ```
   Keeps model close to base model.

2. **Tuned KL Coefficient:**
   ```python
   kl_coef = 0.1  # Start small
   # Too large: prevents learning
   # Too small: allows drift
   ```

3. **Early Stopping:**
   ```python
   if val_reward < best_val_reward * 0.9:
       patience -= 1
       if patience == 0:
           stop_training()
   ```

4. **Validation During Training:**
   ```python
   if step % eval_steps == 0:
       val_reward = evaluate()
       if val_reward < threshold:
           warning("Model may be degrading!")
   ```

**Lessons Learned:**
- Always regularize policy optimization (KL divergence)
- Monitor validation metrics during training
- Implement early stopping
- Inspect generated samples, not just metrics
- Reward collapse is common in RL, prepare for it

**Time Lost:** 4 hours (including re-training)

---

## 7. Intent Classification Challenge

### Challenge: No Ground Truth Intent Predictions

**Problem:**
Reward function needs predicted intent to evaluate intent accuracy, but our system generates text responses, not intent labels.

**Options Considered:**

1. **Train Separate Intent Classifier:**
   ```python
   intent_model = train_bert_classifier(queries, intents)
   predicted_intent = intent_model(query)
   ```
   - Pros: Accurate
   - Cons: Extra model, more complexity

2. **Extract Intent from Response:**
   ```python
   # Check if response mentions intent keywords
   if "refund" in response.lower():
       predicted_intent = "refund_request"
   ```
   - Pros: Simple
   - Cons: Heuristic, not reliable

3. **Skip Intent Reward During Training:**
   ```python
   # Only use intent reward during evaluation with ground truth
   if predicted_intent is None:
       intent_score = 0  # Neutral
   ```
   - Pros: Avoids inaccuracy
   - Cons: Loses training signal

**Solution Implemented:**

Combination approach:
- During **training**: Skip intent score (set to 0)
- During **evaluation**: Use keyword heuristics for approximate intent
- Future: Add dedicated intent classifier

**Lessons Learned:**
- Not all evaluation metrics can be used as training signals
- Approximate metrics are okay for monitoring
- Document limitations clearly
- Plan for future improvements

**Time Lost:** 1 hour

---

## 8. Text Generation Quality

### Challenge: Repetitive or Generic Responses

**Problem:**
Early in training, model generated repetitive or overly generic responses:

```
Query: "I want a refund"
Response: "I can help you with that. How can I help you with that?"

Query: "Where is my order?"
Response: "I can help you with that. How can I help you with that?"
```

**Causes:**
1. **Greedy Decoding**: Always picks highest probability token
2. **Not Enough Diversity**: Sampling temperature too low
3. **Reward Hacking**: Generic responses avoid penalties

**Solutions:**

1. **Use Sampling Instead of Greedy:**
   ```python
   outputs = model.generate(
       do_sample=True,  # Enable sampling
       temperature=0.7,  # Control randomness
       top_p=0.9,        # Nucleus sampling
   )
   ```

2. **Penalize Repetition:**
   ```python
   outputs = model.generate(
       repetition_penalty=1.1,  # Penalize repeated tokens
   )
   ```

3. **Add Diversity Reward:**
   ```python
   # In reward function
   if response_length < 20 or is_generic(response):
       reward -= 1  # Penalize generic responses
   ```

4. **Better Prompting:**
   ```python
   prompt = f"""You are a customer support agent.

   Customer: {query}

   Agent: [Provide specific, detailed response]"""
   ```

**Lessons Learned:**
- Generation hyperparameters matter as much as training
- Test generation quality frequently
- Add explicit diversity incentives
- Good prompting improves quality significantly

**Time Lost:** 2 hours

---

## 9. Evaluation Challenges

### Challenge: Defining "Good" Customer Support

**Problem:**
How do we know if the model is actually good at customer support?

**Metrics Attempted:**

1. **BLEU/ROUGE Score:**
   ```python
   rouge = rouge_score(generated, reference)
   ```
   - Problem: Can get high score with different phrasing
   - Doesn't capture policy compliance or safety

2. **Embedding Similarity:**
   ```python
   similarity = cosine(embed(generated), embed(reference))
   ```
   - Problem: Similar embeddings ≠ good support
   - Misses critical attributes

3. **Multi-Metric Suite (Final):**
   ```python
   metrics = {
       "intent_accuracy": 0.85,
       "policy_compliance": 0.96,
       "mean_reward": 3.2,
       "escalation_f1": 0.78
   }
   ```
   - Better: Captures multiple dimensions
   - Still not perfect but comprehensive

**Ideal But Not Implemented:**
Human evaluation on random sample of 100 responses:
```python
for response in sample:
    human_score = get_human_rating(response)
```

**Lessons Learned:**
- Single metrics miss important aspects
- Need domain-specific evaluation
- Human evaluation is gold standard (but expensive)
- Use multi-metric suite as proxy
- Test on edge cases, not just averages

**Time Lost:** 3 hours designing metrics

---

## 10. Dependency Hell

### Challenge: Conflicting Package Versions

**Problem:**
```
ERROR: pip's dependency resolver does not currently take into account
all the packages that are installed.

transformers requires torch>=2.0
trl requires transformers>=4.30,<4.36
unsloth requires transformers>=4.36
```

**Attempted Solutions:**

1. **Update all packages:**
   ```bash
   pip install --upgrade transformers trl torch
   ```
   - Broke unsloth compatibility

2. **Pin specific versions:**
   ```
   transformers==4.35.0
   trl==0.7.0
   ```
   - Worked but fragile

3. **Virtual environment per component:**
   - Too complex, deployment nightmare

**Final Solution:**

Create `requirements.txt` with tested version ranges:
```
torch>=2.0.0,<2.2.0
transformers>=4.36.0,<4.38.0
trl>=0.7.0,<0.8.0
```

Plus installation instructions:
```bash
# 1. Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 2. Install Unsloth (optional)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 3. Install other dependencies
pip install -r requirements.txt
```

**Lessons Learned:**
- Document exact working versions
- Provide step-by-step installation instructions
- Test installation on clean environment
- Have fallbacks for problematic dependencies
- Consider Docker for reproducibility

**Time Lost:** 2 hours

---

## Summary of Time Lost

| Challenge | Time Lost | Severity |
|-----------|-----------|----------|
| Unsloth installation | 2 hours | Medium |
| Dataset availability | 1 hour | Low |
| Reward function design | 4 hours | High (but necessary) |
| GRPO implementation | 3 hours | High |
| Memory/OOM errors | 2 hours | Medium |
| Training instability | 4 hours | High |
| Intent classification | 1 hour | Low |
| Generation quality | 2 hours | Medium |
| Evaluation design | 3 hours | Medium |
| Dependency conflicts | 2 hours | Medium |
| **Total** | **24 hours** | |

## What I Would Do Differently

### 1. Start with Known Good Environment
Before coding, set up and test:
- PyTorch + CUDA
- Transformers
- Basic model loading

### 2. Test Reward Function First
Spend day 1 entirely on reward function:
- Design
- Test on examples
- Get feedback
- Iterate

### 3. Implement PPO Baseline First
Before GRPO:
- Implement simple policy gradient
- Validate it works
- Then add GRPO improvements

### 4. Profile Early
Before training:
- Profile memory usage
- Profile compute time
- Identify bottlenecks
- Optimize critical path

### 5. Use Docker
For reproducibility:
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-devel
COPY requirements.txt .
RUN pip install -r requirements.txt
```

### 6. Implement Monitoring First
Before training loop:
- Set up Weights & Biases
- Log everything
- Add alerts for issues
- Save checkpoints frequently

## Key Takeaways

1. **Fallbacks are Essential**: Never assume external dependencies work
2. **Test Components Individually**: Easier to debug than full system
3. **RL is Finicky**: Small details matter, test extensively
4. **Memory Management**: Critical for large models
5. **Reward Design**: Most important and most time-consuming
6. **Iterate**: First version won't be perfect
7. **Monitor Everything**: Catch issues early
8. **Document as You Go**: Don't leave it for later

## Positive Surprises

Not everything was a challenge! Some things worked better than expected:

1. **Unsloth Speed**: When it worked, truly 2x faster
2. **LoRA Efficiency**: Barely any quality loss with 0.1% parameters
3. **4-bit Quantization**: Surprising how well it works
4. **Synthetic Data**: Good enough for development/testing
5. **Multi-Component Rewards**: More stable than expected

---

*"The roadblocks are where the learning happens. Every error is a lesson, every bug is a teacher."*
