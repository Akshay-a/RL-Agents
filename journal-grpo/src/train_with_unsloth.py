"""
Standalone Unsloth Fine-Tuning Script

Unsloth is a library that makes fine-tuning 2-5x faster with lower memory usage.
It's perfect for consumer GPUs and quick experimentation.

Key Benefits:
- 2-5x faster training
- 80% less VRAM usage
- Works with HuggingFace transformers
- Easy to use - just replace model loading

This script shows how to fine-tune Qwen with Unsloth for journal entry generation.

Installation:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes

Usage:
python train_with_unsloth.py
"""

import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model settings
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"  # Unsloth's optimized version
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True

# LoRA settings
LORA_R = 16  # Rank
LORA_ALPHA = 32  # Alpha (typically 2x rank)
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training settings
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
OUTPUT_DIR = "./checkpoints/unsloth_sft"

# Data paths
TRAIN_DATA_PATH = "../data/sft_train.jsonl"


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_training_data(data_path: str) -> Dataset:
    """
    Load training data from JSONL file.

    Each line should have:
    - prompt: The instruction/question
    - completion: The expected output

    We'll format this into a conversation template that Qwen understands.
    """
    print(f"Loading training data from {data_path}...")

    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            example = json.loads(line)

            # Format: Instruction + Response
            # Qwen uses a chat template, but for simplicity we'll use a basic format
            text = f"""### Instruction:
{example['prompt']}

### Response:
{example['completion']}"""

            examples.append({"text": text})

    print(f"Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)


# ============================================================================
# STEP 2: LOAD MODEL WITH UNSLOTH
# ============================================================================

def load_model_with_unsloth():
    """
    Load model and tokenizer using Unsloth.

    Why Unsloth?
    - Optimized attention kernels (Flash Attention 2)
    - Gradient checkpointing
    - Optimized embeddings and loss functions
    - 4-bit quantization with minimal quality loss

    Result: 2-5x faster training with same or better quality!
    """
    print("\n" + "="*70)
    print("LOADING MODEL WITH UNSLOTH")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"4-bit quantization: {LOAD_IN_4BIT}")
    print("="*70 + "\n")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (float16 for GPU, float32 for CPU)
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Add LoRA adapters
    # Unsloth's version is optimized for speed
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=42,
    )

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}\n")

    return model, tokenizer


# ============================================================================
# STEP 3: TRAIN
# ============================================================================

def train_model(model, tokenizer, train_dataset):
    """
    Train the model using Unsloth + TRL.

    Training Strategy:
    - SFTTrainer (Supervised Fine-Tuning Trainer) from TRL
    - LoRA for parameter efficiency
    - Gradient accumulation for larger effective batch size
    - Mixed precision (fp16) for speed
    """
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Training examples: {len(train_dataset)}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70 + "\n")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),  # Use bf16 if available
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        optim="adamw_8bit",  # 8-bit Adam optimizer (saves memory)
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",  # Change to "wandb" for experiment tracking
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    # Disable cache for training (saves memory)
    model.config.use_cache = False

    # Train!
    print("Starting training...\n")
    trainer.train()

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70 + "\n")

    return trainer


# ============================================================================
# STEP 4: SAVE MODEL
# ============================================================================

def save_model(trainer, tokenizer):
    """
    Save the fine-tuned model.

    Unsloth models can be saved in two ways:
    1. LoRA adapters only (small, ~50MB)
    2. Merged model (full model with LoRA merged in)
    """
    print("Saving model...\n")

    # Save LoRA adapters (small, fast to save/load)
    lora_output_dir = f"{OUTPUT_DIR}/lora_adapters"
    trainer.model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    print(f"✓ Saved LoRA adapters to: {lora_output_dir}")

    # Optionally: Save merged model (full model with adapters merged)
    # This creates a standalone model that doesn't need PEFT
    merged_output_dir = f"{OUTPUT_DIR}/merged_model"
    trainer.model.save_pretrained_merged(
        merged_output_dir,
        tokenizer,
        save_method="merged_16bit",  # Options: "merged_16bit", "merged_4bit", "lora"
    )
    print(f"✓ Saved merged model to: {merged_output_dir}")

    print("\nModel saved successfully!")


# ============================================================================
# STEP 5: TEST INFERENCE
# ============================================================================

def test_inference(model, tokenizer):
    """
    Test the trained model with a sample prompt.

    This shows how to generate journal entries with your fine-tuned model.
    """
    print("\n" + "="*70)
    print("TESTING INFERENCE")
    print("="*70 + "\n")

    # Sample prompt
    test_prompt = "Record the sale of consulting services for $5,000 cash on 2024-01-15"

    # Format prompt
    formatted_prompt = f"""### Instruction:
{test_prompt}

### Response:
"""

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Enable inference mode (faster)
    FastLanguageModel.for_inference(model)

    # Generate
    print(f"Prompt: {test_prompt}\n")
    print("Generating...\n")

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    print("Generated Journal Entry:")
    print("-" * 70)
    print(generated_text)
    print("-" * 70 + "\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main training pipeline.

    Steps:
    1. Load training data
    2. Load model with Unsloth (optimized)
    3. Train with SFTTrainer
    4. Save model (both LoRA and merged)
    5. Test inference
    """
    print("\n" + "="*70)
    print("UNSLOTH FINE-TUNING PIPELINE")
    print("="*70)
    print("This script fine-tunes Qwen 2.5B for journal entry generation")
    print("using Unsloth for 2-5x faster training!")
    print("="*70 + "\n")

    # Step 1: Load data
    train_dataset = load_training_data(TRAIN_DATA_PATH)

    # Step 2: Load model
    model, tokenizer = load_model_with_unsloth()

    # Step 3: Train
    trainer = train_model(model, tokenizer, train_dataset)

    # Step 4: Save
    save_model(trainer, tokenizer)

    # Step 5: Test
    test_inference(model, tokenizer)

    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"Your fine-tuned model is ready at: {OUTPUT_DIR}")
    print("\nTo use it:")
    print("1. Load LoRA adapters: model = PeftModel.from_pretrained(...)")
    print("2. Or load merged model: model = AutoModelForCausalLM.from_pretrained(...)")
    print("\nSee inference.py for example usage.")
    print("="*70 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check if data exists
    import os
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"ERROR: Training data not found at {TRAIN_DATA_PATH}")
        print("Please run: python data/generate_sft.py first")
    else:
        main()


# ============================================================================
# ADDITIONAL NOTES
# ============================================================================

"""
Understanding the Code:

1. WHY UNSLOTH?
   - Standard fine-tuning: ~12GB VRAM, 3 hours
   - With Unsloth: ~6GB VRAM, 1 hour
   - Same quality, much faster!

2. WHY LORA?
   - Full fine-tuning: Update all 1.5B parameters
   - LoRA: Update only ~1% (still great results!)
   - Much faster and less memory

3. KEY PARAMETERS:
   - LORA_R (16): Higher = more capacity, slower training
   - LORA_ALPHA (32): Scaling factor, typically 2x R
   - LEARNING_RATE (2e-4): Standard for LoRA fine-tuning
   - BATCH_SIZE (4): Adjust based on your GPU memory

4. GRADIENT ACCUMULATION:
   - Updates weights every N batches
   - Simulates larger batch size
   - Example: batch=4, accum=4 → effective batch=16

5. MIXED PRECISION (FP16/BF16):
   - Uses 16-bit floats instead of 32-bit
   - 2x faster, 2x less memory
   - Minimal accuracy loss

6. 4-BIT QUANTIZATION:
   - Stores weights in 4 bits instead of 16
   - 4x less memory!
   - Uses NF4 (normalized float 4-bit) for minimal quality loss

COMMON ISSUES:

Q: CUDA out of memory?
A: Reduce BATCH_SIZE or MAX_SEQ_LENGTH

Q: Training too slow?
A: Ensure you have GPU with CUDA available

Q: Model generates garbage?
A: Try higher LORA_R (32) or more EPOCHS

Q: How to resume training?
A: Add resume_from_checkpoint=True to TrainingArguments

FURTHER OPTIMIZATION:

- Enable Flash Attention 2 (auto-enabled in Unsloth)
- Use bf16 instead of fp16 (if GPU supports)
- Increase gradient accumulation for stability
- Use DeepSpeed for multi-GPU training
"""
