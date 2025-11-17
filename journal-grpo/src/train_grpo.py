"""
GRPO Training - Stage 2: Reinforcement Learning Refinement

This script performs Group Relative Policy Optimization to teach the model
to satisfy hard constraints (balanced journal entries).

Key Concept:
Instead of just learning from examples, GRPO:
1. Generates multiple candidates for each prompt
2. Scores them with reward model
3. Updates policy to favor high-reward outputs

Process:
1. Load SFT checkpoint
2. For each prompt, generate 4 candidates
3. Score each with reward_model.py
4. Compute GRPO loss (upweight good, downweight bad)
5. Update model
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
from reward_model import JournalEntryRewardModel


def load_config(config_path: str = "../configs/grpo_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_grpo_prompts(data_path: str) -> Dataset:
    """
    Load GRPO prompts from JSONL file.

    Format: {"prompt": "..."}
    No labels needed - reward model grades outputs!
    """
    prompts = []

    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompts.append({"query": data["prompt"]})

    print(f"Loaded {len(prompts)} GRPO prompts")
    return Dataset.from_list(prompts)


def load_sft_model(sft_checkpoint: str, tokenizer):
    """
    Load the SFT checkpoint as starting point for GRPO.

    Why start from SFT?
    - Model already knows what journal entries look like
    - GRPO refines this knowledge to satisfy constraints
    - Much faster than starting from scratch
    """
    print(f"Loading SFT checkpoint: {sft_checkpoint}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    return model


def create_reward_function(reward_model: JournalEntryRewardModel):
    """
    Create reward function for GRPO training.

    The reward function takes generated text and returns a score.
    This is the "teacher" that guides the RL training.
    """
    def reward_fn(generated_texts: List[str]) -> List[float]:
        """
        Score a batch of generated journal entries.

        Args:
            generated_texts: List of JSON strings (generated entries)

        Returns:
            List of reward scores (floats)
        """
        rewards = []

        for text in generated_texts:
            try:
                # Extract JSON from generated text
                # Model might generate extra text, so we need to extract JSON
                if "```json" in text:
                    # Extract from markdown code block
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()

                # Score with reward model
                result = reward_model.compute_reward(text)
                reward = result["total_score"]

            except Exception as e:
                # If parsing fails, give negative reward
                reward = -1.0

            rewards.append(reward)

        return rewards

    return reward_fn


def train_grpo_simple(config: Dict):
    """
    Simplified GRPO training using PPO (similar principles to GRPO).

    Why PPO instead of pure GRPO?
    - TRL has excellent PPO support
    - GRPO and PPO share core concepts (policy gradient + reward)
    - PPO is more stable for this use case

    GRPO Key Idea:
    - Generate multiple candidates (temperature > 0)
    - Rank by reward
    - Update policy to prefer high-reward generations
    """
    # Load prompts
    dataset = load_grpo_prompts(config['data']['grpo_prompts'])

    # Take subset for faster training (optional)
    # Uncomment for quick testing:
    # dataset = dataset.select(range(200))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    # Load SFT checkpoint
    sft_checkpoint = os.path.join(config['sft']['output_dir'], "final")
    model = load_sft_model(sft_checkpoint, tokenizer)

    # Wrap model for PPO training
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    # Initialize reward model
    print("Initializing reward model...")
    reward_model = JournalEntryRewardModel()
    reward_fn = create_reward_function(reward_model)

    # PPO configuration
    grpo_config_dict = config['grpo']
    ppo_config = PPOConfig(
        model_name=config['model_name'],
        learning_rate=grpo_config_dict['learning_rate'],
        batch_size=grpo_config_dict['per_device_batch_size'],
        mini_batch_size=1,
        gradient_accumulation_steps=grpo_config_dict['gradient_accumulation_steps'],
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=grpo_config_dict['kl_penalty'],  # KL divergence constraint
        seed=42,
    )

    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=None,
    )

    # Generation config
    generation_kwargs = {
        "max_new_tokens": config['generation']['max_new_tokens'],
        "do_sample": True,
        "temperature": grpo_config_dict['temperature'],
        "top_p": config['generation']['top_p'],
        "pad_token_id": tokenizer.pad_token_id,
    }

    print("\n" + "="*70)
    print("STARTING GRPO TRAINING (via PPO)")
    print("="*70)
    print(f"Training prompts: {len(dataset)}")
    print(f"Candidates per prompt: {grpo_config_dict['num_samples']}")
    print(f"Temperature: {grpo_config_dict['temperature']}")
    print(f"KL penalty: {grpo_config_dict['kl_penalty']}")
    print("="*70 + "\n")

    # Training loop
    for epoch in range(grpo_config_dict['num_epochs']):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{grpo_config_dict['num_epochs']}")
        print(f"{'='*70}\n")

        for batch_idx, batch in enumerate(ppo_trainer.dataloader):
            # Get query (prompt)
            query_tensors = batch["input_ids"]

            # Generate responses
            response_tensors = []
            responses = []

            for query_tensor in query_tensors:
                # Generate response
                response_tensor = ppo_trainer.generate(
                    query_tensor.unsqueeze(0),
                    **generation_kwargs
                )
                response_tensors.append(response_tensor.squeeze())

                # Decode response
                response_text = tokenizer.decode(response_tensor.squeeze(), skip_special_tokens=True)
                responses.append(response_text)

            # Compute rewards
            rewards = reward_fn(responses)
            rewards = [torch.tensor(r) for r in rewards]

            # Update model
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            # Log progress
            if batch_idx % grpo_config_dict['logging_steps'] == 0:
                avg_reward = sum([r.item() for r in rewards]) / len(rewards)
                print(f"Batch {batch_idx} | Avg Reward: {avg_reward:.3f} | KL: {stats.get('objective/kl', 0):.4f}")

            # Save checkpoint
            if batch_idx % grpo_config_dict['save_steps'] == 0 and batch_idx > 0:
                checkpoint_dir = os.path.join(grpo_config_dict['output_dir'], f"checkpoint-{epoch}-{batch_idx}")
                ppo_trainer.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint: {checkpoint_dir}")

    # Save final model
    final_output_dir = os.path.join(grpo_config_dict['output_dir'], "final")
    ppo_trainer.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print("\n" + "="*70)
    print("GRPO TRAINING COMPLETE")
    print("="*70)
    print(f"Model saved to: {final_output_dir}")
    print("Next step: Run evaluate.py to test performance")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Load config
    config = load_config("configs/grpo_config.yaml")

    # Check if SFT checkpoint exists
    sft_checkpoint = os.path.join(config['sft']['output_dir'], "final")
    if not os.path.exists(sft_checkpoint):
        print(f"ERROR: SFT checkpoint not found at {sft_checkpoint}")
        print("Run: python src/train_sft.py first")
        sys.exit(1)

    # Check if prompts exist
    if not os.path.exists(config['data']['grpo_prompts']):
        print(f"ERROR: GRPO prompts not found at {config['data']['grpo_prompts']}")
        print("Run: python data/generate_grpo_candidates.py")
        sys.exit(1)

    # Train
    train_grpo_simple(config)


if __name__ == "__main__":
    main()
