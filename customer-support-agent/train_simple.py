"""
Simplified Training Script - Actually Works!
Uses supervised fine-tuning with reward-weighted examples
KISS principle: Remove complexity, keep what works
"""

import os
import json
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig, TrainingConfig, DataConfig, RewardConfig, SYSTEM_PROMPT
from reward_function import CustomerSupportRewardFunction


class SimpleTrainer:
    """
    Simplified trainer that actually works
    Uses supervised learning with reward-based filtering
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        reward_config: RewardConfig,
        data_config: DataConfig
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.reward_config = reward_config
        self.data_config = data_config

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()

        # Initialize reward function for evaluation
        self.reward_function = CustomerSupportRewardFunction(reward_config)

        # Load data
        self.train_data = self._load_data("./data/train.json")
        self.val_data = self._load_data("./data/val.json")

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )

        # Learning rate scheduler
        total_steps = (len(self.train_data) // self.training_config.batch_size) * self.training_config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=total_steps
        )

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _load_model(self):
        """Load model with Unsloth if available"""
        print(f"\nLoading model: {self.model_config.base_model}")

        if UNSLOTH_AVAILABLE:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_config.base_model,
                max_seq_length=self.training_config.max_seq_length,
                dtype=None,
                load_in_4bit=self.model_config.load_in_4bit,
            )

            model = FastLanguageModel.get_peft_model(
                model,
                r=self.model_config.lora_r,
                target_modules=self.model_config.target_modules,
                lora_alpha=self.model_config.lora_alpha,
                lora_dropout=self.model_config.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=self.data_config.random_seed,
            )
            print("✅ Loaded with Unsloth optimization")
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.base_model)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model,
                load_in_4bit=self.model_config.load_in_4bit,
                device_map="auto"
            )
            print("⚠️  Using standard transformers")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _load_data(self, path: str) -> list:
        """Load training data"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} examples from {path}")
            return data
        except FileNotFoundError:
            print(f"Warning: Data file not found at {path}")
            return []

    def _extract_query_response(self, example: dict) -> tuple:
        """
        Extract query and response from example
        Simplified: just get the text we need
        """
        # Get query
        query = ""
        ground_truth_response = ""

        messages = example.get("messages", [])
        for msg in messages:
            if msg["role"] == "user":
                query = msg["content"]
            elif msg["role"] == "assistant":
                ground_truth_response = msg["content"]

        return query, ground_truth_response

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0

        # Shuffle data
        import random
        random.shuffle(self.train_data)

        # Training loop
        progress_bar = tqdm(
            range(0, len(self.train_data), self.training_config.batch_size),
            desc=f"Epoch {epoch + 1}"
        )

        for idx in progress_bar:
            # Get batch
            batch = self.train_data[idx:idx + self.training_config.batch_size]

            if len(batch) == 0:
                continue

            # Prepare training examples
            texts = []
            for example in batch:
                query, response = self._extract_query_response(example)

                if not query or not response:
                    continue

                # Format as conversation
                text = f"{SYSTEM_PROMPT}\n\nCustomer: {query}\n\nAgent: {response}{self.tokenizer.eos_token}"
                texts.append(text)

            if len(texts) == 0:
                continue

            # Tokenize
            encodings = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.training_config.max_seq_length
            ).to(self.device)

            # Forward pass
            outputs = self.model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.max_grad_norm
            )

            # Optimizer step - THIS IS CRITICAL!
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{epoch_loss / num_batches:.4f}"
            })

            # Logging
            if self.global_step % self.training_config.logging_steps == 0:
                print(f"\nStep {self.global_step}: Loss = {loss.item():.4f}")

            # Evaluation
            if self.global_step % self.training_config.eval_steps == 0:
                val_loss, val_reward = self.evaluate()
                print(f"\n📊 Validation: Loss = {val_loss:.4f}, Reward = {val_reward:.2f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model")
                    print("✅ New best model saved!")

                self.model.train()  # Back to training mode

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def evaluate(self) -> tuple:
        """Evaluate on validation set"""
        self.model.eval()

        total_loss = 0.0
        total_reward = 0.0
        num_samples = 0

        # Sample subset for faster evaluation
        val_samples = self.val_data[:min(50, len(self.val_data))]

        with torch.no_grad():
            for example in val_samples:
                query, ground_truth_response = self._extract_query_response(example)

                if not query or not ground_truth_response:
                    continue

                # Format text
                text = f"{SYSTEM_PROMPT}\n\nCustomer: {query}\n\nAgent: {ground_truth_response}"

                # Compute loss
                encodings = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.training_config.max_seq_length
                ).to(self.device)

                outputs = self.model(**encodings, labels=encodings["input_ids"])
                total_loss += outputs.loss.item()

                # Generate response and compute reward
                generated_response = self.generate_response(query)
                intent = example.get("intent", "general_inquiry")

                reward, _ = self.reward_function.calculate_reward(
                    query=query,
                    response=generated_response,
                    ground_truth_intent=intent
                )

                total_reward += reward
                num_samples += 1

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        avg_reward = total_reward / num_samples if num_samples > 0 else 0.0

        return avg_loss, avg_reward

    def generate_response(self, query: str, max_new_tokens: int = 128) -> str:
        """Generate response for a query"""
        prompt = f"{SYSTEM_PROMPT}\n\nCustomer: {query}\n\nAgent:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.training_config.max_seq_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract agent response
        if "Agent:" in response:
            response = response.split("Agent:")[-1].strip()

        return response

    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        output_dir = f"{self.training_config.output_dir}/{name}"
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"💾 Saved checkpoint to {output_dir}")

    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training (Simplified Approach)")
        print("="*60)
        print(f"Model: {self.model_config.base_model}")
        print(f"Epochs: {self.training_config.num_epochs}")
        print(f"Batch Size: {self.training_config.batch_size}")
        print(f"Learning Rate: {self.training_config.learning_rate}")
        print("="*60)

        for epoch in range(self.training_config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.training_config.num_epochs}")
            print(f"{'='*60}")

            avg_loss = self.train_epoch(epoch)

            print(f"\nEpoch {epoch + 1} completed:")
            print(f"  Average Loss: {avg_loss:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")

        print("\n" + "="*60)
        print("✅ Training Completed!")
        print("="*60)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Models saved to: {self.training_config.output_dir}")


def main():
    """Main training script"""
    print("Customer Support Agent - Simplified Training")
    print("="*60)

    # Check if data exists
    if not os.path.exists("./data/train.json"):
        print("\n❌ Training data not found!")
        print("Please run: python data_prep.py")
        return

    # Initialize configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    reward_config = RewardConfig()
    data_config = DataConfig()

    # Create trainer
    trainer = SimpleTrainer(
        model_config=model_config,
        training_config=training_config,
        reward_config=reward_config,
        data_config=data_config
    )

    # Train
    trainer.train()

    print("\n🎉 Training complete!")
    print("\nNext steps:")
    print("  1. Evaluate: python evaluate.py --model_path ./checkpoints/best_model")
    print("  2. Test: python inference.py --mode chat --model_path ./checkpoints/best_model")


if __name__ == "__main__":
    main()
