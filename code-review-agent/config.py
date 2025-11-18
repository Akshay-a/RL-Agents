"""
Configuration for Code Review AI Agent with GRPO
All hyperparameters and settings in one place
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration"""
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Smaller, code-focused model

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    # Quantization
    load_in_4bit: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class GRPOConfig:
    """GRPO training configuration"""
    num_epochs: int = 2
    batch_size: int = 4
    group_size: int = 4  # GRPO group size for advantage calculation
    learning_rate: float = 5e-5
    max_seq_length: int = 1024

    # GRPO specific
    kl_coef: float = 0.05  # KL divergence penalty
    clip_ratio: float = 0.2  # PPO-style clipping
    value_coef: float = 0.5  # Value loss coefficient

    # Optimization
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 50

    # Checkpointing
    save_steps: int = 100
    eval_steps: int = 50
    output_dir: str = "./checkpoints"

    # Training
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class RewardConfig:
    """Reward function weights"""
    # Code understanding
    identifies_issues: float = 5.0  # Correctly identifies bugs/issues

    # Review quality
    provides_solution: float = 3.0  # Suggests fixes
    explains_reasoning: float = 2.0  # Explains why it's an issue

    # Tone
    constructive_tone: float = 2.0  # Helpful, not harsh
    specific_feedback: float = 1.0  # Concrete, not vague

    # Penalties
    wrong_issue: float = -3.0  # Identifies non-existent problem
    harsh_tone: float = -2.0  # Overly critical
    vague_feedback: float = -1.0  # Too general

    # Safety
    suggests_vulnerable_code: float = -5.0  # Recommends insecure patterns

    # Normalization
    clip_range: tuple = (-10.0, 10.0)


@dataclass
class DataConfig:
    """Data configuration"""
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    max_code_length: int = 512  # Max tokens for code snippet


# System prompt for code review
SYSTEM_PROMPT = """You are an expert code reviewer. Your role is to:
1. Identify bugs, security issues, and code smells
2. Provide constructive, specific feedback
3. Suggest concrete improvements
4. Explain your reasoning clearly

Be helpful and professional. Focus on actionable feedback."""


# Code review categories
REVIEW_CATEGORIES = [
    "bug",
    "security",
    "performance",
    "style",
    "best_practice",
    "documentation",
    "testing",
    "clean_code"
]
