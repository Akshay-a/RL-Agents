# Quick Start Guide - Learn GRPO in 30 Minutes

> **Goal**: Understand GRPO fundamentals and run your first training session

---

## What You'll Learn

In 30 minutes, you'll understand:
1. What GRPO/PPO does (reward-based learning)
2. Why it works for accounting (hard constraints)
3. How to train your own model

---

## 5-Minute Concept Overview

### The Problem

**Traditional fine-tuning (SFT)**:
```
Show model: Input → Perfect Output
Model learns to mimic
❌ Problem: Can't enforce "debits must equal credits"
```

**Result**: Model generates entries like:
- Debits: $1000
- Credits: $999 ← Close but WRONG!

### The Solution: Reward-Based Learning (GRPO/PPO)

```
1. Model generates multiple answers
   Answer A: Balanced ✓     → +2.0 points
   Answer B: Unbalanced ✗   → -1.0 points
   Answer C: Invalid codes  → +0.5 points
   Answer D: Balanced ✓     → +2.0 points

2. Model learns: "Balanced = good, unbalanced = bad"

3. Next time: Model generates balanced entries!
```

**Result**: >90% balance accuracy (vs 70% with SFT alone)

### Key Insight

**GRPO/PPO = Training with grades, not just examples**

- SFT: "Here's the answer, copy it"
- GRPO: "Here are 4 attempts, this one scored highest, learn why"

---

## 3-Minute Setup

### 1. Check Your Setup

```bash
cd journal-grpo
python validate_setup.py
```

**Expected**: All tests pass ✓ (except ML libraries if not installed)

### 2. Install Dependencies (if needed)

```bash
pip install torch transformers peft trl bitsandbytes pyyaml
```

Or for fastest training:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

---

## 10-Minute Code Walkthrough

### The Reward Function (Heart of GRPO)

```python
# src/reward_model.py (simplified)

def score_journal_entry(entry):
    # 1. Balance check (most important!)
    if debits == credits:
        balance_score = +1.0  # ✓ Correct
    else:
        balance_score = -1.0  # ✗ Wrong

    # 2. Valid account codes
    valid_codes = count_valid_codes(entry)
    account_score = valid_codes * 0.5

    # 3. Schema compliance
    schema_score = 0.3 if valid_json else 0.0

    # 4. Reasonable amounts
    amount_score = 0.2 if no_negatives else 0.0

    # Total: -1.0 to +2.0
    return balance_score + account_score + schema_score + amount_score
```

**This function teaches the model what "good" means!**

### The Training Loop (Simplified)

```python
# What happens during training

for prompt in training_data:
    # Generate answer
    generated_entry = model.generate(prompt)

    # Score it
    reward = reward_function(generated_entry)

    # Update model
    if reward > 0:
        model.learn("do more like this!")
    else:
        model.learn("avoid this!")
```

**Real code** in `train_grpo.py` has more details, but concept is the same.

---

## 12-Minute Hands-On Training

### Option 1: Quick Demo (10 minutes)

Generate tiny dataset and train:

```bash
# 1. Generate 50 examples (instead of 500)
python -c "
import sys
sys.path.insert(0, 'src')
from reward_model import JournalEntryRewardModel

sys.path.insert(0, 'data')
from generate_sft import SFTDataGenerator

gen = SFTDataGenerator(JournalEntryRewardModel())
data = gen.generate_dataset(num_examples=50)

import json
with open('data/sft_train_tiny.jsonl', 'w') as f:
    for ex in data:
        f.write(json.dumps(ex) + '\n')
print('Generated 50 examples!')
"

# 2. Edit train_with_unsloth.py
# Change line 40: TRAIN_DATA_PATH = "../data/sft_train_tiny.jsonl"
# Change line 45: NUM_EPOCHS = 1

# 3. Train (takes ~10 min on GPU, ~30 min on CPU)
python src/train_with_unsloth.py

# 4. Test the model (check generated entry)
# Output will show a sample journal entry
```

### Option 2: Full Training (2-6 hours)

```bash
# 1. Generate full dataset (~5 min)
python data/generate_sft.py          # 500 examples
python data/generate_grpo_candidates.py  # 2000 prompts

# 2. Train Stage 1: SFT (~2 hours)
python src/train_sft.py

# 3. Train Stage 2: GRPO/PPO (~4 hours)
python src/train_grpo.py

# 4. Evaluate
python src/evaluate.py
```

**Fastest**: Use Unsloth (Step 1 done in ~1 hour)
```bash
python src/train_with_unsloth.py
```

---

## Understanding the Results

### What to Look For

**During Training**:
```
Epoch 1: Loss = 2.5
Epoch 2: Loss = 1.2
Epoch 3: Loss = 0.8  ← Getting better!
```

**After Training**:
```
Test on: "Sold services for $5,000 cash"

Model generates:
{
  "date": "2024-01-15",
  "entries": [
    {"account": "Cash", "debit": 5000, "credit": 0},
    {"account": "Revenue", "debit": 0, "credit": 5000}
  ]
}

Reward Score: 2.0 / 2.0 ✓ Perfect!
```

### Metrics Explained

| Metric | What It Means | Target |
|--------|---------------|--------|
| **Balance Accuracy** | % with debits == credits | >90% |
| **Avg Reward** | Average score across test set | >1.8 |
| **Schema Compliance** | % valid JSON structure | >95% |

---

## Next Steps

### Learn More

**If you have 30 more minutes**:
1. Read `TRAINING_GUIDE.md` (GRPO Fundamentals section)
2. Study `train_with_unsloth.py` code (well-commented)

**If you have 2 hours**:
1. Read `PROJECT_JOURNAL.md` (deep dive into GRPO)
2. Experiment with hyperparameters in `configs/grpo_config.yaml`

**If you want to go deep**:
1. Read `VALIDATION.md` (what's actually implemented)
2. Read PPO paper (Schulman et al., 2017)
3. Implement custom reward functions

### Experiment Ideas

1. **Change reward weights**:
   ```python
   # In reward_model.py
   balance_score = +2.0  # Increase from +1.0
   # Train again, see if balance improves
   ```

2. **Add new transaction types**:
   ```python
   # In generate_sft.py
   def generate_loan_payment(self):
       # New transaction type
   ```

3. **Try different models**:
   ```yaml
   # In grpo_config.yaml
   model_name: "Qwen/Qwen2.5-3B-Instruct"  # Larger model
   ```

---

## Common Questions

### Q: What's the difference between GRPO and PPO?

**A**: In this codebase, they're essentially the same. We use PPO (standard library) but the concept is identical:
- Both use reward functions
- Both update policy to maximize reward
- Both work for constraint enforcement

Read `VALIDATION.md` for details.

### Q: Do I need a GPU?

**A**: Recommended but not required
- **With GPU**: 1-6 hours training
- **Without GPU**: 10-50 hours training (slow but works!)
- **Google Colab**: Free GPU, runs fine

### Q: Can I use this for other tasks?

**A**: YES! The pattern works for:
- Code generation (must compile = reward)
- Math problems (correct answer = reward)
- SQL queries (valid syntax = reward)
- Any task with verifiable constraints

Just change:
1. Reward function (what makes output "good")
2. Training data (examples for your domain)
3. Model (appropriate size for task)

### Q: How do I know if it's working?

**Watch for**:
1. ✅ Reward scores increasing during training
2. ✅ Balance accuracy >90% on test set
3. ✅ Generated entries look correct to human
4. ✅ Loss decreasing over time

**Red flags**:
1. ❌ Reward scores stay constant
2. ❌ All generations look identical
3. ❌ Loss increases or oscillates wildly

---

## Troubleshooting 30-Second Fixes

### "CUDA out of memory"
```bash
# Reduce batch size
# Edit grpo_config.yaml:
per_device_batch_size: 2  # Down from 4
```

### "Training too slow"
```bash
# Use Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python src/train_with_unsloth.py
```

### "Model generates garbage"
```bash
# Train longer or increase LoRA rank
# Edit grpo_config.yaml:
num_epochs: 5    # Up from 3
lora_r: 32      # Up from 16
```

### "ModuleNotFoundError"
```bash
# Install dependencies
pip install -r requirements.txt
```

---

## Summary: 30-Minute Takeaways

### Core Concepts ✓

1. **GRPO/PPO**: Train models using reward signals (not just examples)
2. **Reward Function**: Defines what "good" means (balance = +1, unbalanced = -1)
3. **Two-Stage**: SFT gets "close", GRPO/PPO enforces constraints
4. **LoRA + Quantization**: Train efficiently on consumer GPUs

### Code Structure ✓

```
reward_model.py       ← Grades outputs
   ↓
generate_sft.py       ← Creates training data
   ↓
train_with_unsloth.py ← Trains model
   ↓
evaluate.py           ← Tests results
```

### Running Training ✓

```bash
# Quick (10 min): Use tiny dataset
python validate_setup.py
# Edit train_with_unsloth.py for tiny data
python src/train_with_unsloth.py

# Full (2-6 hours): Complete pipeline
python data/generate_sft.py
python src/train_sft.py
python src/train_grpo.py
python src/evaluate.py
```

### Expected Results ✓

- SFT only: ~70% balance accuracy
- SFT + GRPO: **>90% balance accuracy**
- Proof that reward-based learning works!

---

## You're Ready! 🚀

**Next action**:
```bash
# Start with validation
python validate_setup.py

# Then choose:
# A) Quick demo (10 min)
# B) Full training (2-6 hours)
# C) Read more docs first
```

**Need help?**
- Code questions → Read inline comments in `train_with_unsloth.py`
- Concepts → Read `TRAINING_GUIDE.md`
- Validation → Read `VALIDATION.md`
- Theory → Read `PROJECT_JOURNAL.md`

---

**You now understand GRPO fundamentals!** 🎉

Go train a model and see reward-based learning in action.
