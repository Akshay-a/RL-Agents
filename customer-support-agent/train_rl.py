"""
End-to-End RL Training for Customer Support Agent
Uses TRL (Transformer Reinforcement Learning) library with PPO
Proper RL implementation that actually works!
"""

import os
import json
import torch
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from tqdm import tqdm

# TRL for RL training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# Transformers
from transformers import AutoTokenizer
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM

# Local imports
from config import ModelConfig, TrainingConfig, DataConfig, RewardConfig, SYSTEM_PROMPT
from reward_function import CustomerSupportRewardFunction


class RLCustomerSupportTrainer:
    """
    End-to-end RL training using PPO (similar to GRPO concept)
    Uses TRL library for proven RL implementation
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
        print(f"🖥️  Using device: {self.device}")

        # Initialize reward function
        self.reward_function = CustomerSupportRewardFunction(reward_config)

        # Load data
        self.train_data = self._load_data("./data/train.json")
        self.val_data = self._load_data("./data/val.json")

        # Build datasets
        self.train_queries = self._extract_queries(self.train_data)

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self._setup_model()

        # Initialize PPO trainer
        self.ppo_trainer = None
        self._setup_ppo_trainer()

        print("✅ Trainer initialized successfully!")

    def _load_data(self, path: str) -> List[Dict]:
        """Load training data"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"📚 Loaded {len(data)} examples from {path}")
            return data
        except FileNotFoundError:
            print(f"⚠️  Data file not found at {path}")
            return []

    def _extract_queries(self, data: List[Dict]) -> List[Dict]:
        """Extract queries with metadata for RL training"""
        queries = []
        for example in data:
            messages = example.get("messages", [])
            query = ""
            for msg in messages:
                if msg["role"] == "user":
                    query = msg["content"]
                    break

            if query:
                queries.append({
                    "query": query,
                    "intent": example.get("intent", "general_inquiry"),
                    "query_tensor": None  # Will be filled during training
                })

        return queries

    def _setup_model(self):
        """Setup model for RL training"""
        print(f"\n🔧 Loading model: {self.model_config.base_model}")

        if UNSLOTH_AVAILABLE:
            # Load with Unsloth for speed
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_config.base_model,
                max_seq_length=self.training_config.max_seq_length,
                dtype=None,
                load_in_4bit=self.model_config.load_in_4bit,
            )

            # Apply LoRA
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

            print("✅ Loaded with Unsloth (2x faster!)")

            # Wrap for PPO (TRL needs value head)
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        else:
            # Standard transformers
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.base_model)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model,
                load_in_4bit=self.model_config.load_in_4bit,
                device_map="auto"
            )
            print("⚠️  Using standard transformers (slower)")

            # Wrap for PPO
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer

        # Reference model for KL penalty (frozen copy)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model.pretrained_model
        )
        self.ref_model.eval()

    def _setup_ppo_trainer(self):
        """Setup PPO trainer from TRL"""
        # PPO configuration
        ppo_config = PPOConfig(
            model_name=self.model_config.base_model,
            learning_rate=self.training_config.learning_rate,
            batch_size=self.training_config.batch_size,
            mini_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            optimize_cuda_cache=True,
            early_stopping=True,
            target_kl=self.training_config.kl_coef,
            ppo_epochs=4,
            seed=self.data_config.random_seed,
            remove_unused_columns=False,
        )

        # Create PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

        print("✅ PPO Trainer initialized")

    def generate_response(self, query_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Generate responses using current policy"""
        response_tensors = []

        for query_tensor in query_tensors:
            # Generate
            with torch.no_grad():
                response = self.ppo_trainer.generate(
                    query_tensor.unsqueeze(0),
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Extract only the generated part (remove query)
            response_only = response[0][len(query_tensor):]
            response_tensors.append(response_only)

        return response_tensors

    def compute_rewards(
        self,
        queries: List[str],
        responses: List[str],
        intents: List[str]
    ) -> List[torch.Tensor]:
        """Compute rewards using reward function"""
        rewards = []

        for query, response, intent in zip(queries, responses, intents):
            # Calculate reward
            reward_value, components = self.reward_function.calculate_reward(
                query=query,
                response=response,
                ground_truth_intent=intent,
                predicted_intent=None
            )

            # Convert to tensor
            reward_tensor = torch.tensor([reward_value], dtype=torch.float32)
            rewards.append(reward_tensor)

        return rewards

    def train_epoch(self, epoch: int):
        """Train one epoch with PPO"""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{self.training_config.num_epochs}")
        print(f"{'='*60}")

        # Shuffle queries
        import random
        random.shuffle(self.train_queries)

        # Training loop
        epoch_rewards = []
        batch_size = self.training_config.batch_size

        for batch_start in tqdm(
            range(0, len(self.train_queries), batch_size),
            desc=f"Epoch {epoch + 1}"
        ):
            batch = self.train_queries[batch_start:batch_start + batch_size]

            if len(batch) == 0:
                continue

            # Prepare queries
            query_texts = []
            query_tensors = []
            intents = []

            for item in batch:
                # Format query with system prompt
                query_text = f"{SYSTEM_PROMPT}\n\nCustomer: {item['query']}\n\nAgent:"

                # Tokenize
                query_tensor = self.tokenizer.encode(
                    query_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.training_config.max_seq_length // 2
                )[0]

                query_texts.append(item['query'])
                query_tensors.append(query_tensor)
                intents.append(item['intent'])

            # Generate responses
            response_tensors = self.generate_response(query_tensors)

            # Decode responses
            responses = [
                self.tokenizer.decode(r, skip_special_tokens=True)
                for r in response_tensors
            ]

            # Compute rewards
            rewards = self.compute_rewards(query_texts, responses, intents)

            # Track metrics
            mean_reward = np.mean([r.item() for r in rewards])
            epoch_rewards.append(mean_reward)

            # PPO step - THIS IS WHERE RL HAPPENS!
            stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)

            # Log
            if batch_start % (batch_size * 10) == 0:
                print(f"\n  Batch {batch_start//batch_size}: Reward = {mean_reward:.2f}")

        # Epoch summary
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Average Reward: {avg_reward:.2f}")

        return avg_reward

    def evaluate(self) -> float:
        """Evaluate on validation set"""
        print("\n🔍 Evaluating...")

        self.model.eval()
        eval_rewards = []

        # Sample validation set
        val_samples = self.val_data[:min(50, len(self.val_data))]

        for example in val_samples:
            messages = example.get("messages", [])
            query = ""
            for msg in messages:
                if msg["role"] == "user":
                    query = msg["content"]
                    break

            if not query:
                continue

            # Generate response
            query_text = f"{SYSTEM_PROMPT}\n\nCustomer: {query}\n\nAgent:"
            query_tensor = self.tokenizer.encode(
                query_text,
                return_tensors="pt",
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                response = self.ppo_trainer.generate(
                    query_tensor,
                    max_new_tokens=128,
                    temperature=0.7,
                )

            response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)

            # Extract agent response
            if "Agent:" in response_text:
                response_text = response_text.split("Agent:")[-1].strip()

            # Calculate reward
            intent = example.get("intent", "general_inquiry")
            reward, _ = self.reward_function.calculate_reward(
                query=query,
                response=response_text,
                ground_truth_intent=intent
            )

            eval_rewards.append(reward)

        avg_reward = np.mean(eval_rewards) if eval_rewards else 0.0
        print(f"   Validation Reward: {avg_reward:.2f}")

        self.model.train()
        return avg_reward

    def save_model(self, path: str):
        """Save trained model"""
        os.makedirs(path, exist_ok=True)

        # Save model and tokenizer
        self.ppo_trainer.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        print(f"💾 Model saved to {path}")

    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("🚀 Starting RL Training with PPO")
        print("="*60)
        print(f"Model: {self.model_config.base_model}")
        print(f"Epochs: {self.training_config.num_epochs}")
        print(f"Batch Size: {self.training_config.batch_size}")
        print(f"Training Samples: {len(self.train_queries)}")
        print("="*60)

        best_reward = float('-inf')

        for epoch in range(self.training_config.num_epochs):
            # Train epoch
            avg_reward = self.train_epoch(epoch)

            # Evaluate
            val_reward = self.evaluate()

            # Save best model
            if val_reward > best_reward:
                best_reward = val_reward
                self.save_model(f"{self.training_config.output_dir}/best_model")
                print("✅ New best model saved!")

            # Save epoch checkpoint
            self.save_model(f"{self.training_config.output_dir}/epoch_{epoch+1}")

        print("\n" + "="*60)
        print("🎉 RL Training Completed!")
        print("="*60)
        print(f"Best Validation Reward: {best_reward:.2f}")
        print(f"Models saved to: {self.training_config.output_dir}")


def main():
    """Main training script"""
    print("="*60)
    print("Customer Support Agent - RL Training with PPO")
    print("="*60)

    # Check data
    if not os.path.exists("./data/train.json"):
        print("\n❌ Training data not found!")
        print("Please run: python data_prep_simple.py")
        return

    # Initialize configs
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # Adjust for RL (smaller batches, more epochs)
    training_config.batch_size = 4  # Smaller for RL
    training_config.num_epochs = 3

    reward_config = RewardConfig()
    data_config = DataConfig()

    # Create trainer
    trainer = RLCustomerSupportTrainer(
        model_config=model_config,
        training_config=training_config,
        reward_config=reward_config,
        data_config=data_config
    )

    # Train with RL!
    trainer.train()

    print("\n🎉 Training complete!")
    print("\nNext steps:")
    print("  1. Evaluate: python evaluate.py --model_path ./checkpoints/best_model")
    print("  2. Deploy: python inference.py --mode chat --model_path ./checkpoints/best_model")


if __name__ == "__main__":
    main()
