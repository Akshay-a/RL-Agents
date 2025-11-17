"""
Supervised Fine-Tuning (SFT) - Stage 1

This script performs standard supervised fine-tuning on labeled journal entry examples.
Goal: Get the model "in the ballpark" of generating valid journal entries.

Process:
1. Load base model (Qwen 2.5B)
2. Add LoRA adapters for efficient fine-tuning
3. Train on 500 labeled examples
4. Save checkpoint for GRPO stage
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def load_config(config_path: str = "../configs/grpo_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sft_data(data_path: str) -> Dataset:
    """
    Load SFT training data from JSONL file.

    Format expected:
    {"prompt": "...", "completion": "{...}", "metadata": {...}}
    """
    examples = []

    with open(data_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            # Combine prompt and completion into single text for training
            text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['completion']}"
            examples.append({"text": text})

    print(f"Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)


def create_model_and_tokenizer(config: Dict):
    """
    Create model and tokenizer with 4-bit quantization and LoRA.

    Why 4-bit quantization?
    - Reduces VRAM from ~12GB to ~6GB
    - Minimal accuracy loss
    - Allows training on consumer GPUs

    Why LoRA?
    - Only trains ~1% of parameters
    - Fast training
    - Easy to merge or swap adapters
    """
    model_name = config['model_name']

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normalized float 4-bit
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
    )

    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute across available GPUs
        trust_remote_code=True,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type'],
    )

    # Add LoRA adapters to model
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


def train_sft(config: Dict):
    """
    Main SFT training function.

    Training Strategy:
    - 3 epochs (more than 3 risks overfitting on 500 examples)
    - Effective batch size: 16 (4 per device * 4 grad accumulation)
    - Learning rate: 2e-4 (standard for LoRA fine-tuning)
    """
    # Load data
    train_dataset = load_sft_data(config['data']['sft_train'])

    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(config)

    # Training arguments
    sft_config = config['sft']
    training_args = TrainingArguments(
        output_dir=sft_config['output_dir'],
        num_train_epochs=sft_config['num_epochs'],
        per_device_train_batch_size=sft_config['per_device_batch_size'],
        gradient_accumulation_steps=sft_config['gradient_accumulation_steps'],
        learning_rate=sft_config['learning_rate'],
        warmup_steps=sft_config['warmup_steps'],
        logging_steps=sft_config['logging_steps'],
        save_steps=sft_config['save_steps'],
        save_total_limit=3,  # Keep only last 3 checkpoints
        optim=sft_config['optim'],
        fp16=True,  # Mixed precision training
        report_to="none",  # Change to "wandb" if you want experiment tracking
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=sft_config['max_seq_length'],
        dataset_text_field="text",  # Field containing the training text
    )

    # Start training
    print("\n" + "="*70)
    print("STARTING SFT TRAINING")
    print("="*70)
    print(f"Training examples: {len(train_dataset)}")
    print(f"Epochs: {sft_config['num_epochs']}")
    print(f"Effective batch size: {sft_config['per_device_batch_size'] * sft_config['gradient_accumulation_steps']}")
    print(f"Total steps: ~{len(train_dataset) // (sft_config['per_device_batch_size'] * sft_config['gradient_accumulation_steps']) * sft_config['num_epochs']}")
    print("="*70 + "\n")

    trainer.train()

    # Save final model
    final_output_dir = os.path.join(sft_config['output_dir'], "final")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print("\n" + "="*70)
    print("SFT TRAINING COMPLETE")
    print("="*70)
    print(f"Model saved to: {final_output_dir}")
    print("Next step: Run train_grpo.py for reinforcement learning refinement")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Load configuration
    config = load_config("configs/grpo_config.yaml")

    # Check if data exists
    if not os.path.exists(config['data']['sft_train']):
        print(f"ERROR: Training data not found at {config['data']['sft_train']}")
        print("Run: python data/generate_sft.py")
        sys.exit(1)

    # Train
    train_sft(config)


if __name__ == "__main__":
    main()
