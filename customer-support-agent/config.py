"""
Configuration file for Customer Support AI Agent
Centralizes all hyperparameters and settings
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration"""
    # Choose base model: "meta-llama/Llama-3.2-3B" or "mistralai/Mistral-7B-v0.1"
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"

    # LoRA configuration for efficient fine-tuning
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = None  # Will use Unsloth defaults

    # Quantization for memory efficiency
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainingConfig:
    """GRPO training configuration"""
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # GRPO specific
    grpo_group_size: int = 4  # Number of samples per group for GRPO
    kl_coef: float = 0.1  # KL divergence coefficient
    gamma: float = 0.99  # Discount factor

    # Optimization
    optimizer: str = "adamw_8bit"
    weight_decay: float = 0.01
    max_seq_length: int = 512

    # Checkpointing
    save_steps: int = 100
    eval_steps: int = 50
    logging_steps: int = 10
    output_dir: str = "./checkpoints"

    # Early stopping
    early_stopping_patience: int = 3


@dataclass
class RewardConfig:
    """Reward function configuration"""
    # Core rewards
    intent_classification_correct: float = 5.0
    policy_compliance: float = 3.0
    empathy_tone: float = 2.0
    proper_escalation: float = 1.0

    # Penalties
    policy_violation: float = -3.0
    wrong_intent: float = -2.0
    inappropriate_tone: float = -2.0
    dangerous_response: float = -5.0

    # Bonuses
    complete_resolution: float = 2.0
    proactive_suggestions: float = 1.0

    # Normalization
    normalize_rewards: bool = True
    clip_rewards: bool = True
    reward_clip_range: tuple = (-10.0, 10.0)


@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    max_samples: Optional[int] = None  # Set to limit dataset size for testing
    random_seed: int = 42


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: list = None
    test_batch_size: int = 8
    num_test_samples: int = 100

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "intent_accuracy",
                "policy_compliance_rate",
                "response_quality",
                "escalation_precision"
            ]


# Intent categories for customer support
INTENT_CATEGORIES = [
    "refund_request",
    "order_tracking",
    "product_inquiry",
    "complaint",
    "technical_support",
    "account_management",
    "billing_issue",
    "shipping_inquiry",
    "return_exchange",
    "general_inquiry"
]

# Company policies (example - customize based on your business)
COMPANY_POLICIES = {
    "refund_window": "30 days from purchase",
    "refund_conditions": "Product must be unused and in original packaging",
    "unauthorized_promises": [
        "Cannot promise immediate refunds without verification",
        "Cannot share customer personal data",
        "Cannot override return policy without manager approval"
    ],
    "escalation_triggers": [
        "Customer requests manager",
        "Complaint about service quality",
        "Request outside standard policy",
        "Customer extremely upset or threatening legal action"
    ],
    "tone_guidelines": {
        "empathetic": "Always acknowledge customer frustration",
        "professional": "Maintain courteous language",
        "helpful": "Provide actionable next steps"
    }
}

# System prompt template for customer support
SYSTEM_PROMPT = """You are a helpful customer support agent. Your role is to:
1. Understand customer intent accurately
2. Follow company policies strictly
3. Respond with empathy and professionalism
4. Escalate when appropriate
5. Provide clear, actionable solutions

Remember:
- Never promise what you can't deliver
- Never share private customer data
- Always follow the refund/return policies
- Escalate complex issues to human agents
"""
