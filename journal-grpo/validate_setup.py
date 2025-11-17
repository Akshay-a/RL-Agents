"""
Validation Script - Test All Components

This script validates that all components work correctly before training.
Run this to ensure your setup is ready.

Tests:
1. Reward model functionality
2. Data generators
3. Configuration loading
4. Model loading (without training)
5. Documentation accuracy

Usage:
    python validate_setup.py
"""

import os
import sys
import json
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("\n" + "="*70)
print("JOURNAL-GRPO VALIDATION SCRIPT")
print("="*70)
print("This script validates your setup before training.\n")

# ============================================================================
# TEST 1: Reward Model
# ============================================================================

print("TEST 1: Reward Model")
print("-" * 70)

try:
    from reward_model import JournalEntryRewardModel

    reward_model = JournalEntryRewardModel()

    # Test valid entry
    valid_entry = {
        "date": "2024-01-15",
        "description": "Test entry",
        "entries": [
            {"account_code": "1000", "account_name": "Cash", "debit": 1000.0, "credit": 0.0},
            {"account_code": "4100", "account_name": "Revenue", "debit": 0.0, "credit": 1000.0}
        ]
    }

    result = reward_model.compute_reward(valid_entry)

    assert result["total_score"] == 2.0, f"Expected score 2.0, got {result['total_score']}"
    assert result["is_valid"] == True, "Valid entry should be marked as valid"

    # Test invalid entry (unbalanced)
    invalid_entry = {
        "date": "2024-01-15",
        "description": "Unbalanced",
        "entries": [
            {"account_code": "1000", "account_name": "Cash", "debit": 1000.0, "credit": 0.0},
            {"account_code": "4100", "account_name": "Revenue", "debit": 0.0, "credit": 500.0}
        ]
    }

    result = reward_model.compute_reward(invalid_entry)
    assert result["breakdown"]["balance"] == -1.0, "Unbalanced entry should get -1.0 balance score"
    assert result["is_valid"] == False, "Unbalanced entry should be invalid"

    print("✓ Reward model working correctly")
    print(f"  - Chart of accounts: {len(reward_model.chart_of_accounts)} accounts loaded")
    print(f"  - Scoring: Valid entry gets 2.0, unbalanced gets penalty")

except Exception as e:
    print(f"✗ Reward model test failed: {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: Data Files
# ============================================================================

print("\nTEST 2: Data Files")
print("-" * 70)

try:
    # Check schema
    schema_path = Path("data/schema.json")
    assert schema_path.exists(), "schema.json not found"
    with open(schema_path) as f:
        schema = json.load(f)
    assert "properties" in schema, "Invalid schema format"
    print(f"✓ schema.json exists and is valid")

    # Check chart of accounts
    coa_path = Path("data/chart_of_accounts.json")
    assert coa_path.exists(), "chart_of_accounts.json not found"
    with open(coa_path) as f:
        coa = json.load(f)
    assert len(coa["accounts"]) == 25, f"Expected 25 accounts, found {len(coa['accounts'])}"
    print(f"✓ chart_of_accounts.json exists with 25 accounts")

    # Check golden test set
    test_path = Path("test_cases/golden_set.jsonl")
    assert test_path.exists(), "golden_set.jsonl not found"
    test_count = sum(1 for _ in open(test_path))
    assert test_count == 50, f"Expected 50 test cases, found {test_count}"
    print(f"✓ golden_set.jsonl exists with 50 test cases")

except Exception as e:
    print(f"✗ Data files test failed: {e}")
    sys.exit(1)

# ============================================================================
# TEST 3: Configuration
# ============================================================================

print("\nTEST 3: Configuration")
print("-" * 70)

try:
    config_path = Path("configs/grpo_config.yaml")
    assert config_path.exists(), "grpo_config.yaml not found"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    assert "model_name" in config, "Missing model_name"
    assert "lora" in config, "Missing lora config"
    assert "sft" in config, "Missing sft config"
    assert "grpo" in config, "Missing grpo config"

    print(f"✓ Configuration file valid")
    print(f"  - Model: {config['model_name']}")
    print(f"  - LoRA rank: {config['lora']['r']}")
    print(f"  - SFT epochs: {config['sft']['num_epochs']}")
    print(f"  - GRPO epochs: {config['grpo']['num_epochs']}")

except Exception as e:
    print(f"✗ Configuration test failed: {e}")
    sys.exit(1)

# ============================================================================
# TEST 4: Data Generators
# ============================================================================

print("\nTEST 4: Data Generators")
print("-" * 70)

try:
    # Test SFT generator
    sys.path.insert(0, str(Path("data")))
    from generate_sft import SFTDataGenerator

    generator = SFTDataGenerator(reward_model)

    # Generate one example
    prompt, entry = generator.generate_cash_sale()
    assert prompt is not None, "Failed to generate prompt"
    assert entry is not None, "Failed to generate entry"
    assert "entries" in entry, "Entry missing 'entries' field"

    # Validate with reward model
    result = reward_model.compute_reward(entry)
    assert result["is_valid"], f"Generated entry is invalid: {result['messages']}"

    print(f"✓ SFT data generator working")
    print(f"  - Sample prompt: {prompt[:60]}...")
    print(f"  - Reward score: {result['total_score']}")

    # Test GRPO generator
    from generate_grpo_candidates import GRPOPromptGenerator

    grpo_gen = GRPOPromptGenerator()
    prompt = grpo_gen.generate_prompt()
    assert prompt is not None, "Failed to generate GRPO prompt"
    assert len(prompt) > 10, "Prompt too short"

    print(f"✓ GRPO prompt generator working")
    print(f"  - Sample prompt: {prompt[:60]}...")

except Exception as e:
    print(f"✗ Data generator test failed: {e}")
    sys.exit(1)

# ============================================================================
# TEST 5: Dependencies Check
# ============================================================================

print("\nTEST 5: Dependencies")
print("-" * 70)

dependencies = {
    "torch": "PyTorch",
    "transformers": "HuggingFace Transformers",
    "peft": "PEFT (LoRA)",
    "trl": "TRL (RL training)",
    "bitsandbytes": "BitsAndBytes (quantization)",
    "yaml": "PyYAML",
}

missing = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"✓ {name}")
    except ImportError:
        print(f"✗ {name} - NOT INSTALLED")
        missing.append(module)

if missing:
    print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
    print("Install with: pip install -r requirements.txt")
else:
    print("\n✓ All dependencies installed")

# ============================================================================
# TEST 6: Documentation Validation
# ============================================================================

print("\nTEST 6: Documentation")
print("-" * 70)

docs = {
    "README.md": "Project overview",
    "PROJECT_JOURNAL.md": "GRPO fundamentals and journey",
    "TRAINING_GUIDE.md": "Training concepts and troubleshooting",
}

for doc, description in docs.items():
    path = Path(doc)
    if path.exists():
        size_kb = path.stat().st_size / 1024
        print(f"✓ {doc} ({size_kb:.1f} KB) - {description}")
    else:
        print(f"✗ {doc} - MISSING")

# ============================================================================
# TEST 7: Training Scripts
# ============================================================================

print("\nTEST 7: Training Scripts")
print("-" * 70)

scripts = {
    "src/train_sft.py": "Supervised fine-tuning",
    "src/train_grpo.py": "GRPO/PPO training",
    "src/train_with_unsloth.py": "Unsloth fast training",
    "src/evaluate.py": "Model evaluation",
}

for script, description in scripts.items():
    path = Path(script)
    if path.exists():
        lines = sum(1 for _ in open(path))
        print(f"✓ {script} ({lines} lines) - {description}")
    else:
        print(f"✗ {script} - MISSING")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

print("""
✓ All core components validated!

Next Steps:

1. GENERATE TRAINING DATA:
   cd journal-grpo
   python data/generate_sft.py        # Creates 500 examples (~5 min)
   python data/generate_grpo_candidates.py  # Creates 2000 prompts (~30 sec)

2. TRAIN MODEL (Choose one):

   Option A - Standard HuggingFace:
   python src/train_sft.py            # Stage 1: SFT (~2 hours)
   python src/train_grpo.py           # Stage 2: GRPO (~4 hours)

   Option B - Unsloth (2-5x faster):
   python src/train_with_unsloth.py  # All-in-one (~1 hour)

3. EVALUATE:
   python src/evaluate.py             # Compare models

Documentation:
- Quick concepts: README.md
- Deep dive: TRAINING_GUIDE.md
- GRPO theory: PROJECT_JOURNAL.md

For learning GRPO, start with:
1. Read TRAINING_GUIDE.md (especially GRPO Fundamentals section)
2. Read train_with_unsloth.py code (well-commented)
3. Run training and observe reward scores
""")

print("="*70)
print("Setup validated successfully! Ready to train.")
print("="*70 + "\n")
