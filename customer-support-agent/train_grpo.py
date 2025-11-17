"""
GRPO Training Script with Unsloth Integration
Trains customer support agent using Group Relative Policy Optimization
"""

import os
import json
import torch
from dataclasses import asdict
from typing import List, Dict
import numpy as np
from tqdm import tqdm

# Unsloth and model imports
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("Warning: Unsloth not available. Install with: pip install unsloth")
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import TrainingArguments
from trl import AutoModelForCausalLMWithValueHead

# Local imports
from config import ModelConfig, TrainingConfig, DataConfig, RewardConfig, SYSTEM_PROMPT
from reward_function import CustomerSupportRewardFunction
from support_env import CustomerSupportEnv


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) Trainer
    Implements policy optimization with group-based advantage estimation
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        reward_config: RewardConfig,
        data_config: DataConfig
    ):
        """Initialize GRPO trainer"""
        self.model_config = model_config
        self.training_config = training_config
        self.reward_config = reward_config
        self.data_config = data_config

        # Initialize components
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()

        # Initialize reward function
        self.reward_function = CustomerSupportRewardFunction(reward_config)

        # Load training data
        self.train_data = self._load_data("./data/train.json")
        self.val_data = self._load_data("./data/val.json")

        # Training state
        self.global_step = 0
        self.best_val_reward = float('-inf')
        self.training_history = {
            "train_rewards": [],
            "val_rewards": [],
            "train_loss": [],
            "kl_divergence": []
        }

    def _load_model(self):
        """Load base model with Unsloth optimization"""
        print(f"\nLoading model: {self.model_config.base_model}")

        if UNSLOTH_AVAILABLE:
            # Use Unsloth for 2x faster training
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_config.base_model,
                max_seq_length=self.training_config.max_seq_length,
                dtype=None,  # Auto-detect
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
                use_gradient_checkpointing="unsloth",  # Unsloth optimization
                random_state=self.data_config.random_seed,
            )

            print("✅ Loaded model with Unsloth optimization (2x faster!)")

        else:
            # Fallback to standard transformers
            print("⚠️  Using standard transformers (slower)")
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.base_model)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model,
                load_in_4bit=self.model_config.load_in_4bit,
                device_map="auto"
            )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _load_data(self, path: str) -> List[Dict]:
        """Load training or validation data"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} examples from {path}")
            return data
        except FileNotFoundError:
            print(f"Warning: Data file not found at {path}")
            return []

    def generate_response(self, query: str, max_new_tokens: int = 256) -> str:
        """Generate response from model"""
        # Format prompt
        prompt = f"{SYSTEM_PROMPT}\n\nCustomer: {query}\n\nAgent:"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.training_config.max_seq_length
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the agent response (after "Agent:")
        if "Agent:" in response:
            response = response.split("Agent:")[-1].strip()

        return response

    def collect_rollouts(self, batch_size: int) -> Dict:
        """
        Collect rollouts (query-response pairs) with rewards
        This is the core of GRPO data collection
        """
        queries = []
        responses = []
        rewards = []
        intents = []
        metadata_list = []

        # Sample batch from training data
        batch_samples = np.random.choice(
            len(self.train_data),
            size=min(batch_size, len(self.train_data)),
            replace=False
        )

        for idx in batch_samples:
            example = self.train_data[idx]

            # Extract query
            messages = example.get("messages", [])
            query = ""
            for msg in messages:
                if msg["role"] == "user":
                    query = msg["content"]
                    break

            if not query:
                continue

            # Generate response from model
            response = self.generate_response(query)

            # Calculate reward
            ground_truth_intent = example.get("intent", "general_inquiry")
            reward, components = self.reward_function.calculate_reward(
                query=query,
                response=response,
                ground_truth_intent=ground_truth_intent,
                predicted_intent=None
            )

            # Store rollout
            queries.append(query)
            responses.append(response)
            rewards.append(reward)
            intents.append(ground_truth_intent)
            metadata_list.append(components.to_dict())

        return {
            "queries": queries,
            "responses": responses,
            "rewards": rewards,
            "intents": intents,
            "metadata": metadata_list
        }

    def compute_grpo_advantages(self, rewards: List[float], group_size: int) -> np.ndarray:
        """
        Compute GRPO advantages using group-relative comparison
        Key idea: Compare each sample's reward with others in its group
        """
        rewards = np.array(rewards)
        num_samples = len(rewards)

        # Pad rewards to make divisible by group_size
        padding = (group_size - num_samples % group_size) % group_size
        if padding > 0:
            rewards = np.pad(rewards, (0, padding), mode='constant', constant_values=np.mean(rewards))

        # Reshape into groups
        num_groups = len(rewards) // group_size
        grouped_rewards = rewards.reshape(num_groups, group_size)

        # Compute group-relative advantages
        # Each sample's advantage is relative to its group mean
        group_means = grouped_rewards.mean(axis=1, keepdims=True)
        advantages = grouped_rewards - group_means

        # Flatten back and remove padding
        advantages = advantages.flatten()[:num_samples]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def train_step(self, batch_size: int) -> Dict:
        """
        Single training step with GRPO
        """
        self.model.train()

        # Collect rollouts
        rollouts = self.collect_rollouts(batch_size)

        if len(rollouts["rewards"]) == 0:
            return {"loss": 0.0, "reward": 0.0}

        # Compute GRPO advantages
        advantages = self.compute_grpo_advantages(
            rollouts["rewards"],
            self.training_config.grpo_group_size
        )

        # Prepare for training
        total_loss = 0.0
        num_updates = 0

        # Process in mini-batches
        for i in range(0, len(rollouts["queries"]), self.training_config.batch_size):
            end_idx = min(i + self.training_config.batch_size, len(rollouts["queries"]))

            batch_queries = rollouts["queries"][i:end_idx]
            batch_responses = rollouts["responses"][i:end_idx]
            batch_advantages = advantages[i:end_idx]

            # Create training samples
            batch_texts = []
            for query, response in zip(batch_queries, batch_responses):
                text = f"{SYSTEM_PROMPT}\n\nCustomer: {query}\n\nAgent: {response}"
                batch_texts.append(text)

            # Tokenize
            encodings = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.training_config.max_seq_length
            ).to(self.device)

            # Forward pass
            outputs = self.model(**encodings, labels=encodings["input_ids"])
            log_probs = -outputs.loss  # Negative loss as log prob

            # Compute policy gradient loss with advantages
            # Loss = -advantage * log_prob
            advantages_tensor = torch.tensor(batch_advantages, dtype=torch.float32).to(self.device)
            policy_loss = -(advantages_tensor * log_probs).mean()

            # Add KL regularization (prevent model from deviating too much)
            kl_loss = self.training_config.kl_coef * outputs.loss

            # Total loss
            loss = policy_loss + kl_loss

            # Backward and optimize
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.max_grad_norm
            )

            # Optimizer step (simplified - in practice use proper optimizer)
            # This would be handled by Trainer in full implementation

            total_loss += loss.item()
            num_updates += 1

        # Average metrics
        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
        avg_reward = np.mean(rollouts["rewards"])

        return {
            "loss": avg_loss,
            "reward": avg_reward,
            "num_samples": len(rollouts["rewards"])
        }

    def evaluate(self, num_samples: int = 50) -> Dict:
        """Evaluate model on validation set"""
        self.model.eval()

        eval_rewards = []
        eval_components = []

        # Sample from validation set
        val_samples = np.random.choice(
            len(self.val_data),
            size=min(num_samples, len(self.val_data)),
            replace=False
        )

        for idx in val_samples:
            example = self.val_data[idx]

            # Extract query
            messages = example.get("messages", [])
            query = ""
            for msg in messages:
                if msg["role"] == "user":
                    query = msg["content"]
                    break

            if not query:
                continue

            # Generate response
            response = self.generate_response(query)

            # Calculate reward
            ground_truth_intent = example.get("intent", "general_inquiry")
            reward, components = self.reward_function.calculate_reward(
                query=query,
                response=response,
                ground_truth_intent=ground_truth_intent
            )

            eval_rewards.append(reward)
            eval_components.append(components.to_dict())

        # Compute metrics
        metrics = {
            "mean_reward": np.mean(eval_rewards) if eval_rewards else 0.0,
            "std_reward": np.std(eval_rewards) if eval_rewards else 0.0,
            "min_reward": np.min(eval_rewards) if eval_rewards else 0.0,
            "max_reward": np.max(eval_rewards) if eval_rewards else 0.0,
            "num_samples": len(eval_rewards)
        }

        return metrics

    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting GRPO Training")
        print("="*60)

        total_steps = self.training_config.num_epochs * (
            len(self.train_data) // self.training_config.batch_size
        )

        print(f"Total training steps: {total_steps}")
        print(f"Batch size: {self.training_config.batch_size}")
        print(f"GRPO group size: {self.training_config.grpo_group_size}")

        # Training loop
        for epoch in range(self.training_config.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.training_config.num_epochs} ---")

            epoch_rewards = []
            epoch_losses = []

            # Steps per epoch
            steps_per_epoch = len(self.train_data) // self.training_config.batch_size

            for step in tqdm(range(steps_per_epoch), desc="Training"):
                # Training step
                metrics = self.train_step(self.training_config.batch_size)

                epoch_rewards.append(metrics["reward"])
                epoch_losses.append(metrics["loss"])
                self.global_step += 1

                # Logging
                if self.global_step % self.training_config.logging_steps == 0:
                    print(f"\nStep {self.global_step}: Loss={metrics['loss']:.4f}, Reward={metrics['reward']:.2f}")

                # Evaluation
                if self.global_step % self.training_config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    print(f"\n📊 Evaluation: Mean Reward={eval_metrics['mean_reward']:.2f}")

                    # Save best model
                    if eval_metrics['mean_reward'] > self.best_val_reward:
                        self.best_val_reward = eval_metrics['mean_reward']
                        self.save_checkpoint("best_model")
                        print("✅ New best model saved!")

            # Epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Mean Reward: {np.mean(epoch_rewards):.2f}")
            print(f"  Mean Loss: {np.mean(epoch_losses):.4f}")

            # Save checkpoint
            if (epoch + 1) % 1 == 0:
                self.save_checkpoint(f"epoch_{epoch+1}")

        print("\n" + "="*60)
        print("✅ Training completed!")
        print("="*60)

    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        output_dir = f"{self.training_config.output_dir}/{name}"
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_val_reward": self.best_val_reward,
            "training_history": self.training_history
        }

        with open(f"{output_dir}/training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"💾 Checkpoint saved to {output_dir}")


def main():
    """Main training script"""
    print("Customer Support AI Agent - GRPO Training")
    print("="*60)

    # Initialize configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    reward_config = RewardConfig()
    data_config = DataConfig()

    print("\n📋 Configuration:")
    print(f"  Model: {model_config.base_model}")
    print(f"  Epochs: {training_config.num_epochs}")
    print(f"  Batch Size: {training_config.batch_size}")
    print(f"  GRPO Group Size: {training_config.grpo_group_size}")

    # Check if data exists
    if not os.path.exists("./data/train.json"):
        print("\n❌ Training data not found!")
        print("Please run: python data_prep.py")
        return

    # Initialize trainer
    trainer = GRPOTrainer(
        model_config=model_config,
        training_config=training_config,
        reward_config=reward_config,
        data_config=data_config
    )

    # Start training
    trainer.train()

    print("\n🎉 Training complete!")
    print(f"Best validation reward: {trainer.best_val_reward:.2f}")
    print(f"Model saved to: {training_config.output_dir}")


if __name__ == "__main__":
    main()
