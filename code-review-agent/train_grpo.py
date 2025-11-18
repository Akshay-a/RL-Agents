"""
GRPO Training for Code Review Agent
Group Relative Policy Optimization implementation
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from typing import List, Dict

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoTokenizer

from config import ModelConfig, GRPOConfig, RewardConfig, DataConfig, SYSTEM_PROMPT
from reward_function import CodeReviewRewardFunction


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) Trainer

    Key idea: Compute advantages relative to group mean instead of global mean
    This provides better comparison and more stable training
    """

    def __init__(
        self,
        model_config: ModelConfig,
        grpo_config: GRPOConfig,
        reward_config: RewardConfig,
        data_config: DataConfig
    ):
        self.model_config = model_config
        self.grpo_config = grpo_config
        self.reward_config = reward_config
        self.data_config = data_config

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Device: {self.device}")

        # Initialize reward function
        self.reward_function = CodeReviewRewardFunction(reward_config)

        # Load data
        self.train_data = self._load_data("./data/train.json")
        self.val_data = self._load_data("./data/val.json")

        # Setup model
        self.model = None
        self.tokenizer = None
        self.ref_model = None  # Frozen reference for KL penalty
        self._setup_model()

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.grpo_config.learning_rate
        )

        # Scheduler
        total_steps = (len(self.train_data) // self.grpo_config.batch_size) * self.grpo_config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.grpo_config.warmup_steps,
            num_training_steps=total_steps
        )

        # Training state
        self.global_step = 0
        self.best_val_reward = float('-inf')

        print("✅ GRPO Trainer initialized!")

    def _load_data(self, path: str) -> List[Dict]:
        """Load training data"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"📚 Loaded {len(data)} examples from {path}")
            return data
        except FileNotFoundError:
            print(f"⚠️  File not found: {path}")
            return []

    def _setup_model(self):
        """Setup model for GRPO training"""
        print(f"\n🔧 Loading model: {self.model_config.base_model}")

        if UNSLOTH_AVAILABLE:
            # Use Unsloth for 2x speedup
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_config.base_model,
                max_seq_length=self.grpo_config.max_seq_length,
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

            print("✅ Loaded with Unsloth (2x faster)")
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

        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        # Create frozen reference model for KL penalty
        if UNSLOTH_AVAILABLE:
            ref_model, _ = FastLanguageModel.from_pretrained(
                model_name=self.model_config.base_model,
                max_seq_length=self.grpo_config.max_seq_length,
                dtype=None,
                load_in_4bit=self.model_config.load_in_4bit,
            )
        else:
            ref_model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model,
                load_in_4bit=self.model_config.load_in_4bit,
                device_map="auto"
            )

        self.ref_model = ref_model.to(self.device)
        self.ref_model.eval()  # Frozen
        for param in self.ref_model.parameters():
            param.requires_grad = False

        print("✅ Reference model created (frozen)")

    def generate_response(self, code: str) -> str:
        """Generate review for code snippet"""
        prompt = f"{SYSTEM_PROMPT}\n\nReview this code:\n\n```python\n{code}\n```\n\nReview:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.grpo_config.max_seq_length - self.grpo_config.max_new_tokens
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.grpo_config.max_new_tokens,
                do_sample=True,
                temperature=self.grpo_config.temperature,
                top_p=self.grpo_config.top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract review (after "Review:")
        if "Review:" in response:
            response = response.split("Review:")[-1].strip()

        return response

    def compute_grpo_advantages(self, rewards: List[float]) -> np.ndarray:
        """
        Compute GRPO advantages using group-relative comparison

        Key idea:
        - Standard RL: advantage = reward - mean(all_rewards)
        - GRPO: advantage = reward - mean(group_rewards)

        This gives better comparison within similar samples
        """
        rewards = np.array(rewards)
        num_samples = len(rewards)
        group_size = self.grpo_config.group_size

        # Pad if needed
        padding = (group_size - num_samples % group_size) % group_size
        if padding > 0:
            # Pad with mean to avoid skewing
            pad_value = np.mean(rewards)
            rewards = np.pad(rewards, (0, padding), constant_values=pad_value)

        # Reshape into groups
        num_groups = len(rewards) // group_size
        grouped_rewards = rewards.reshape(num_groups, group_size)

        # Compute group-relative advantages
        group_means = grouped_rewards.mean(axis=1, keepdims=True)
        advantages = grouped_rewards - group_means  # Relative to group!

        # Flatten and remove padding
        advantages = advantages.flatten()[:num_samples]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def compute_policy_loss(
        self,
        batch_texts: List[str],
        advantages: np.ndarray
    ) -> torch.Tensor:
        """
        Compute policy gradient loss with PPO-style clipping

        Loss = -advantage * log P(action|state) with clipping
        """
        # Tokenize
        encodings = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.grpo_config.max_seq_length
        ).to(self.device)

        # Forward pass (current policy)
        outputs = self.model(**encodings, labels=encodings["input_ids"])
        current_logprobs = -outputs.loss  # Approximate log prob

        # Forward pass (reference policy for KL)
        with torch.no_grad():
            ref_outputs = self.ref_model(**encodings, labels=encodings["input_ids"])
            ref_logprobs = -ref_outputs.loss

        # Convert advantages to tensor
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        # Policy gradient loss
        ratio = torch.exp(current_logprobs - ref_logprobs)  # π_new / π_old

        # PPO-style clipping
        clip_ratio = self.grpo_config.clip_ratio
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

        # Take minimum (conservative policy update)
        policy_loss = -torch.min(
            ratio * advantages_tensor.mean(),
            clipped_ratio * advantages_tensor.mean()
        )

        # KL divergence penalty (prevent drift)
        kl_penalty = self.grpo_config.kl_coef * (current_logprobs - ref_logprobs).pow(2).mean()

        total_loss = policy_loss + kl_penalty

        return total_loss

    def train_epoch(self, epoch: int) -> float:
        """Train one epoch with GRPO"""
        self.model.train()

        epoch_rewards = []
        batch_size = self.grpo_config.batch_size

        # Shuffle data
        import random
        random.shuffle(self.train_data)

        progress_bar = tqdm(
            range(0, len(self.train_data), batch_size),
            desc=f"Epoch {epoch + 1}"
        )

        for batch_start in progress_bar:
            batch = self.train_data[batch_start:batch_start + batch_size]

            if len(batch) == 0:
                continue

            # Collect rollouts
            codes = []
            reviews = []
            categories = []
            issues = []

            for example in batch:
                code = example.get("code", "")
                category = example.get("category", "bug")
                issue = example.get("issue", "")

                if not code:
                    continue

                # Generate review
                review = self.generate_response(code)

                codes.append(code)
                reviews.append(review)
                categories.append(category)
                issues.append(issue)

            if len(reviews) == 0:
                continue

            # Compute rewards
            rewards = []
            for code, review, category, issue in zip(codes, reviews, categories, issues):
                reward, _ = self.reward_function.calculate_reward(
                    code=code,
                    review=review,
                    ground_truth_category=category,
                    ground_truth_issue=issue
                )
                rewards.append(reward)

            # Compute GRPO advantages (group-relative!)
            advantages = self.compute_grpo_advantages(rewards)

            # Prepare training texts
            batch_texts = []
            for code, review in zip(codes, reviews):
                text = f"{SYSTEM_PROMPT}\n\nReview this code:\n\n```python\n{code}\n```\n\nReview: {review}"
                batch_texts.append(text)

            # Compute loss
            loss = self.compute_policy_loss(batch_texts, advantages)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grpo_config.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            # Track metrics
            mean_reward = np.mean(rewards)
            epoch_rewards.append(mean_reward)
            self.global_step += 1

            # Update progress
            progress_bar.set_postfix({
                "reward": f"{mean_reward:.2f}",
                "loss": f"{loss.item():.4f}"
            })

            # Evaluation
            if self.global_step % self.grpo_config.eval_steps == 0:
                val_reward = self.evaluate()
                print(f"\n📊 Step {self.global_step}: Val Reward = {val_reward:.2f}")

                if val_reward > self.best_val_reward:
                    self.best_val_reward = val_reward
                    self.save_model("best_model")
                    print("✅ New best model!")

                self.model.train()

        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        return avg_reward

    def evaluate(self) -> float:
        """Evaluate on validation set"""
        self.model.eval()

        eval_rewards = []
        val_samples = self.val_data[:min(30, len(self.val_data))]

        for example in val_samples:
            code = example.get("code", "")
            category = example.get("category", "bug")
            issue = example.get("issue", "")

            if not code:
                continue

            # Generate review
            review = self.generate_response(code)

            # Compute reward
            reward, _ = self.reward_function.calculate_reward(
                code=code,
                review=review,
                ground_truth_category=category,
                ground_truth_issue=issue
            )

            eval_rewards.append(reward)

        avg_reward = np.mean(eval_rewards) if eval_rewards else 0.0
        return avg_reward

    def save_model(self, name: str):
        """Save model checkpoint"""
        output_dir = f"{self.grpo_config.output_dir}/{name}"
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"💾 Saved to {output_dir}")

    def train(self):
        """Main GRPO training loop"""
        print("\n" + "="*60)
        print("🚀 Starting GRPO Training")
        print("="*60)
        print(f"Model: {self.model_config.base_model}")
        print(f"Epochs: {self.grpo_config.num_epochs}")
        print(f"Batch Size: {self.grpo_config.batch_size}")
        print(f"Group Size: {self.grpo_config.group_size}")
        print(f"Training Samples: {len(self.train_data)}")
        print("="*60)

        for epoch in range(self.grpo_config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.grpo_config.num_epochs}")
            print(f"{'='*60}")

            avg_reward = self.train_epoch(epoch)

            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   Average Reward: {avg_reward:.2f}")

            # Save epoch checkpoint
            self.save_model(f"epoch_{epoch + 1}")

        print("\n" + "="*60)
        print("🎉 GRPO Training Completed!")
        print("="*60)
        print(f"Best Validation Reward: {self.best_val_reward:.2f}")


def main():
    """Main training script"""
    print("="*60)
    print("Code Review Agent - GRPO Training")
    print("="*60)

    # Check data
    if not os.path.exists("./data/train.json"):
        print("\n❌ Training data not found!")
        print("Run: python data_prep.py")
        return

    # Initialize configs
    model_config = ModelConfig()
    grpo_config = GRPOConfig()
    reward_config = RewardConfig()
    data_config = DataConfig()

    # Create trainer
    trainer = GRPOTrainer(
        model_config=model_config,
        grpo_config=grpo_config,
        reward_config=reward_config,
        data_config=data_config
    )

    # Train!
    trainer.train()

    print("\n✅ Training complete!")
    print("\nNext steps:")
    print("  1. Evaluate: python evaluate.py")
    print("  2. Deploy: python inference.py")


if __name__ == "__main__":
    main()
