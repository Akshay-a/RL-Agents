"""
Comprehensive Evaluation Suite for Customer Support Agent
Tests model performance on multiple metrics and edge cases
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass
import torch

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ModelConfig, RewardConfig, DataConfig,
    SYSTEM_PROMPT, INTENT_CATEGORIES, COMPANY_POLICIES
)
from reward_function import CustomerSupportRewardFunction


@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    intent_accuracy: float = 0.0
    policy_compliance_rate: float = 0.0
    mean_reward: float = 0.0
    escalation_precision: float = 0.0
    escalation_recall: float = 0.0
    response_quality_score: float = 0.0
    total_examples: int = 0

    def to_dict(self) -> Dict:
        return {
            "intent_accuracy": self.intent_accuracy,
            "policy_compliance_rate": self.policy_compliance_rate,
            "mean_reward": self.mean_reward,
            "escalation_precision": self.escalation_precision,
            "escalation_recall": self.escalation_recall,
            "response_quality_score": self.response_quality_score,
            "total_examples": self.total_examples
        }


class CustomerSupportEvaluator:
    """Comprehensive evaluator for customer support agent"""

    def __init__(
        self,
        model_path: str,
        test_data_path: str = "./data/test.json",
        reward_config: RewardConfig = None
    ):
        """Initialize evaluator"""
        self.model_path = model_path
        self.test_data_path = test_data_path

        # Load model and tokenizer
        print(f"Loading model from {model_path}")
        self.model, self.tokenizer = self._load_model()

        # Initialize reward function
        if reward_config is None:
            reward_config = RewardConfig()
        self.reward_function = CustomerSupportRewardFunction(reward_config)

        # Load test data
        self.test_data = self._load_test_data()

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Load trained model"""
        if UNSLOTH_AVAILABLE:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=512,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)  # Enable inference mode
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto"
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _load_test_data(self) -> List[Dict]:
        """Load test dataset"""
        try:
            with open(self.test_data_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} test examples")
            return data
        except FileNotFoundError:
            print(f"Warning: Test data not found at {self.test_data_path}")
            return []

    def generate_response(self, query: str, max_new_tokens: int = 256) -> str:
        """Generate response from model"""
        prompt = f"{SYSTEM_PROMPT}\n\nCustomer: {query}\n\nAgent:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
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

        if "Agent:" in response:
            response = response.split("Agent:")[-1].strip()

        return response

    def evaluate_intent_classification(self) -> float:
        """
        Evaluate intent classification accuracy
        Note: This is simplified - in practice would need intent classifier
        """
        # For now, use reward function's intent scoring
        correct = 0
        total = 0

        for example in self.test_data:
            messages = example.get("messages", [])
            query = ""
            for msg in messages:
                if msg["role"] == "user":
                    query = msg["content"]
                    break

            if not query:
                continue

            ground_truth_intent = example.get("intent", "general_inquiry")

            # Generate response
            response = self.generate_response(query)

            # Simple heuristic: check if response mentions relevant keywords
            # In production, would use a proper intent classifier
            intent_keywords = {
                "refund_request": ["refund", "return", "money back"],
                "order_tracking": ["track", "shipping", "delivery", "order status"],
                "product_inquiry": ["product", "specs", "features", "available"],
                "complaint": ["sorry", "apologize", "understand", "frustration"],
                "technical_support": ["troubleshoot", "fix", "technical", "support"]
            }

            # Check if response contains relevant keywords for ground truth intent
            response_lower = response.lower()
            if ground_truth_intent in intent_keywords:
                keywords = intent_keywords[ground_truth_intent]
                if any(kw in response_lower for kw in keywords):
                    correct += 1

            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def evaluate_policy_compliance(self) -> float:
        """Evaluate policy compliance rate"""
        compliant = 0
        total = 0

        for example in self.test_data:
            messages = example.get("messages", [])
            query = ""
            for msg in messages:
                if msg["role"] == "user":
                    query = msg["content"]
                    break

            if not query:
                continue

            ground_truth_intent = example.get("intent", "general_inquiry")
            response = self.generate_response(query)

            # Calculate reward (includes policy compliance)
            reward, components = self.reward_function.calculate_reward(
                query=query,
                response=response,
                ground_truth_intent=ground_truth_intent
            )

            # If policy score is positive, it's compliant
            if components.policy_score > 0:
                compliant += 1

            total += 1

        compliance_rate = compliant / total if total > 0 else 0.0
        return compliance_rate

    def evaluate_escalation(self) -> Tuple[float, float]:
        """
        Evaluate escalation handling (precision and recall)
        Precision: Of cases where model escalates, how many needed it?
        Recall: Of cases that needed escalation, how many did model escalate?
        """
        escalation_triggers = ["manager", "supervisor", "legal", "unacceptable", "complaint"]

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for example in self.test_data:
            messages = example.get("messages", [])
            query = ""
            for msg in messages:
                if msg["role"] == "user":
                    query = msg["content"]
                    break

            if not query:
                continue

            response = self.generate_response(query)

            # Check if query needs escalation
            needs_escalation = any(trigger in query.lower() for trigger in escalation_triggers)

            # Check if model escalates
            escalation_keywords = ["transfer", "connect", "manager", "supervisor", "escalate"]
            model_escalates = any(kw in response.lower() for kw in escalation_keywords)

            if needs_escalation and model_escalates:
                true_positives += 1
            elif not needs_escalation and model_escalates:
                false_positives += 1
            elif needs_escalation and not model_escalates:
                false_negatives += 1

        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        return precision, recall

    def evaluate_response_quality(self) -> float:
        """Evaluate overall response quality using reward function"""
        rewards = []

        for example in self.test_data:
            messages = example.get("messages", [])
            query = ""
            for msg in messages:
                if msg["role"] == "user":
                    query = msg["content"]
                    break

            if not query:
                continue

            ground_truth_intent = example.get("intent", "general_inquiry")
            response = self.generate_response(query)

            reward, _ = self.reward_function.calculate_reward(
                query=query,
                response=response,
                ground_truth_intent=ground_truth_intent
            )

            rewards.append(reward)

        mean_reward = np.mean(rewards) if rewards else 0.0
        return mean_reward

    def run_full_evaluation(self) -> EvaluationResults:
        """Run comprehensive evaluation"""
        print("\n" + "="*60)
        print("Running Comprehensive Evaluation")
        print("="*60)

        results = EvaluationResults()
        results.total_examples = len(self.test_data)

        # 1. Intent Classification Accuracy
        print("\n1️⃣  Evaluating Intent Classification...")
        results.intent_accuracy = self.evaluate_intent_classification()
        print(f"   Intent Accuracy: {results.intent_accuracy*100:.1f}%")

        # 2. Policy Compliance Rate
        print("\n2️⃣  Evaluating Policy Compliance...")
        results.policy_compliance_rate = self.evaluate_policy_compliance()
        print(f"   Policy Compliance Rate: {results.policy_compliance_rate*100:.1f}%")

        # 3. Escalation Metrics
        print("\n3️⃣  Evaluating Escalation Handling...")
        precision, recall = self.evaluate_escalation()
        results.escalation_precision = precision
        results.escalation_recall = recall
        print(f"   Escalation Precision: {precision*100:.1f}%")
        print(f"   Escalation Recall: {recall*100:.1f}%")

        # 4. Response Quality
        print("\n4️⃣  Evaluating Response Quality...")
        results.response_quality_score = self.evaluate_response_quality()
        results.mean_reward = results.response_quality_score
        print(f"   Mean Reward: {results.mean_reward:.2f}")

        return results

    def test_edge_cases(self):
        """Test model on edge cases"""
        print("\n" + "="*60)
        print("Testing Edge Cases")
        print("="*60)

        edge_cases = [
            {
                "name": "Angry Customer",
                "query": "This is absolutely unacceptable! I want a refund NOW and I want to speak to your manager!",
                "expected_behavior": "Should acknowledge frustration, offer escalation"
            },
            {
                "name": "Ambiguous Query",
                "query": "I have a problem with my order",
                "expected_behavior": "Should ask clarifying questions"
            },
            {
                "name": "Multi-Intent Request",
                "query": "I want to track my order and also need to know how to return it if I don't like it",
                "expected_behavior": "Should address both tracking and return policy"
            },
            {
                "name": "Policy Edge Case",
                "query": "Can I get a refund after 45 days? I really need it",
                "expected_behavior": "Should explain policy without violating it"
            },
            {
                "name": "Inappropriate Customer",
                "query": "You're all idiots and I hate your company",
                "expected_behavior": "Should remain professional and offer help"
            }
        ]

        for case in edge_cases:
            print(f"\n🔍 {case['name']}")
            print(f"Query: {case['query']}")
            print(f"Expected: {case['expected_behavior']}")

            response = self.generate_response(case['query'])
            print(f"\n🤖 Response: {response}")

            # Evaluate reward
            reward, components = self.reward_function.calculate_reward(
                query=case['query'],
                response=response,
                ground_truth_intent="general_inquiry"
            )

            print(f"\n💰 Reward: {reward:.2f}")
            print("Components:")
            for key, value in components.to_dict().items():
                if value != 0.0:
                    print(f"  {key}: {value:+.2f}")

            print("-" * 60)

    def save_results(self, results: EvaluationResults, output_path: str = "./evaluation_results.json"):
        """Save evaluation results"""
        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)

        print(f"\n💾 Results saved to {output_path}")

    def generate_report(self, results: EvaluationResults):
        """Generate detailed evaluation report"""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)

        print(f"\n📊 Overall Metrics:")
        print(f"   Total Test Examples: {results.total_examples}")
        print(f"   Mean Reward: {results.mean_reward:.2f}")
        print(f"\n🎯 Accuracy Metrics:")
        print(f"   Intent Classification: {results.intent_accuracy*100:.1f}%")
        print(f"   Policy Compliance: {results.policy_compliance_rate*100:.1f}%")

        print(f"\n🚀 Escalation Metrics:")
        print(f"   Precision: {results.escalation_precision*100:.1f}%")
        print(f"   Recall: {results.escalation_recall*100:.1f}%")

        print(f"\n⭐ Quality Score: {results.response_quality_score:.2f}")

        # Pass/Fail criteria
        print(f"\n✅ Success Criteria:")
        print(f"   Policy Compliance > 95%: {'✅ PASS' if results.policy_compliance_rate > 0.95 else '❌ FAIL'}")
        print(f"   Mean Reward > 0.0: {'✅ PASS' if results.mean_reward > 0.0 else '❌ FAIL'}")

        print("\n" + "="*60)


def main():
    """Main evaluation script"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Customer Support Agent")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/best_model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/test.json",
        help="Path to test data"
    )
    parser.add_argument(
        "--edge_cases",
        action="store_true",
        help="Test edge cases"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found at {args.model_path}")
        print("Please train the model first: python train_grpo.py")
        return

    # Initialize evaluator
    evaluator = CustomerSupportEvaluator(
        model_path=args.model_path,
        test_data_path=args.test_data
    )

    # Run full evaluation
    results = evaluator.run_full_evaluation()

    # Generate report
    evaluator.generate_report(results)

    # Save results
    evaluator.save_results(results)

    # Test edge cases if requested
    if args.edge_cases:
        evaluator.test_edge_cases()

    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
