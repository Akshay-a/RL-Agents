"""
Evaluation Suite for Code Review Agent
Tests model performance on code review quality
"""

import json
import torch
from typing import Dict, List
import numpy as np

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

from config import SYSTEM_PROMPT
from reward_function import CodeReviewRewardFunction, RewardConfig


class CodeReviewEvaluator:
    """Evaluate trained code review model"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = self._load_model()

        # Initialize reward function
        self.reward_function = CodeReviewRewardFunction(RewardConfig())

        print("✅ Evaluator initialized!")

    def _load_model(self):
        """Load trained model"""
        if UNSLOTH_AVAILABLE:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=1024,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto"
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def generate_review(self, code: str) -> str:
        """Generate code review"""
        prompt = f"{SYSTEM_PROMPT}\n\nReview this code:\n\n```python\n{code}\n```\n\nReview:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Review:" in response:
            response = response.split("Review:")[-1].strip()

        return response

    def evaluate_dataset(self, test_data_path: str = "./data/test.json") -> Dict:
        """Evaluate on test dataset"""
        print(f"\n📊 Evaluating on {test_data_path}")

        try:
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ Test data not found at {test_data_path}")
            return {}

        rewards = []
        category_rewards = {}

        for example in test_data:
            code = example.get("code", "")
            category = example.get("category", "bug")
            issue = example.get("issue", "")

            if not code:
                continue

            # Generate review
            review = self.generate_review(code)

            # Compute reward
            reward, components = self.reward_function.calculate_reward(
                code=code,
                review=review,
                ground_truth_category=category,
                ground_truth_issue=issue
            )

            rewards.append(reward)

            # Track by category
            if category not in category_rewards:
                category_rewards[category] = []
            category_rewards[category].append(reward)

        # Compute metrics
        metrics = {
            "mean_reward": np.mean(rewards) if rewards else 0.0,
            "std_reward": np.std(rewards) if rewards else 0.0,
            "min_reward": np.min(rewards) if rewards else 0.0,
            "max_reward": np.max(rewards) if rewards else 0.0,
            "num_samples": len(rewards)
        }

        # Category breakdown
        for category, cat_rewards in category_rewards.items():
            metrics[f"{category}_mean"] = np.mean(cat_rewards)

        return metrics

    def print_report(self, metrics: Dict):
        """Print evaluation report"""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)

        print(f"\n📊 Overall Metrics:")
        print(f"   Test Samples: {metrics.get('num_samples', 0)}")
        print(f"   Mean Reward: {metrics.get('mean_reward', 0):.2f}")
        print(f"   Std Reward: {metrics.get('std_reward', 0):.2f}")
        print(f"   Min Reward: {metrics.get('min_reward', 0):.2f}")
        print(f"   Max Reward: {metrics.get('max_reward', 0):.2f}")

        print(f"\n📁 By Category:")
        categories = ["bug", "security", "performance", "style", "best_practice"]
        for cat in categories:
            key = f"{cat}_mean"
            if key in metrics:
                print(f"   {cat.capitalize()}: {metrics[key]:.2f}")

        print("\n" + "="*60)

    def test_examples(self):
        """Test on example code snippets"""
        print("\n🧪 Testing Example Reviews")
        print("="*60)

        examples = [
            {
                "code": '''def divide(a, b):
    return a / b''',
                "expected": "Division by zero"
            },
            {
                "code": '''password = input("Enter password: ")
print(f"Your password is {password}")''',
                "expected": "Security issue - don't print passwords"
            },
            {
                "code": '''for i in range(len(items)):
    print(items[i])''',
                "expected": "Use direct iteration"
            }
        ]

        for i, ex in enumerate(examples, 1):
            print(f"\n--- Example {i} ---")
            print(f"Code:\n{ex['code']}")
            print(f"\nExpected Issue: {ex['expected']}")

            review = self.generate_review(ex['code'])
            print(f"\n🤖 Generated Review:\n{review}")
            print("-" * 60)


def main():
    """Main evaluation script"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./checkpoints/best_model")
    parser.add_argument("--test_data", default="./data/test.json")
    parser.add_argument("--examples", action="store_true", help="Test on example code")
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = CodeReviewEvaluator(args.model_path)

    # Run evaluation
    metrics = evaluator.evaluate_dataset(args.test_data)

    # Print report
    evaluator.print_report(metrics)

    # Test examples if requested
    if args.examples:
        evaluator.test_examples()

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
