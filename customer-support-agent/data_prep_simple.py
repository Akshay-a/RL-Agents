"""
Simplified Data Preparation
KISS principle: Just get query-response pairs, nothing more
"""

import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import random

from config import DataConfig, SYSTEM_PROMPT


class SimpleDataPrep:
    """Simplified data preparation - no over-engineering"""

    def __init__(self, config: DataConfig):
        self.config = config
        random.seed(config.random_seed)

    def load_dataset(self) -> list:
        """Load Bitext dataset or create synthetic data"""
        print(f"Loading dataset: {self.config.dataset_name}")

        try:
            # Try to load Bitext dataset
            dataset = load_dataset(self.config.dataset_name)
            df = pd.DataFrame(dataset['train'])

            # Limit size if specified
            if self.config.max_samples and len(df) > self.config.max_samples:
                df = df.sample(n=self.config.max_samples, random_state=self.config.random_seed)

            print(f"✅ Loaded {len(df)} examples from Bitext")

            # Convert to simple format
            data = []
            for _, row in df.iterrows():
                # Extract fields (Bitext has: instruction, response, category/intent)
                query = row.get('instruction', row.get('query', ''))
                response = row.get('response', '')
                intent = row.get('intent', row.get('category', 'general_inquiry'))

                if query and response:
                    data.append({
                        "query": query,
                        "response": response,
                        "intent": intent,
                        # Add conversation format for compatibility
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": query},
                            {"role": "assistant", "content": response}
                        ]
                    })

            return data

        except Exception as e:
            print(f"⚠️  Could not load Bitext: {e}")
            print("Creating synthetic dataset...")
            return self._create_synthetic_dataset()

    def _create_synthetic_dataset(self, n_samples: int = 200) -> list:
        """Create synthetic customer support examples"""
        data = []

        examples = [
            {
                "query": "I want a refund for my order #12345",
                "response": "I understand you'd like a refund. I'll be happy to help with that. Our refund policy allows returns within 30 days for unused items in original packaging. Let me check your order #12345 and process this for you.",
                "intent": "refund_request"
            },
            {
                "query": "Where is my package? It's been 5 days",
                "response": "I apologize for the delay. Let me track your package for you. Could you please provide your order number so I can check the current shipping status and estimated delivery date?",
                "intent": "order_tracking"
            },
            {
                "query": "Your product broke after one day! This is unacceptable!",
                "response": "I sincerely apologize for this experience. That's definitely not the quality we expect. I'd like to make this right for you. Let me arrange a replacement or refund immediately. Could you provide your order number?",
                "intent": "complaint"
            },
            {
                "query": "Do you have this in blue?",
                "response": "I'd be happy to check product availability for you. Could you please let me know which specific product you're interested in? I'll verify if we have it in blue.",
                "intent": "product_inquiry"
            },
            {
                "query": "I need to speak to a manager NOW",
                "response": "I understand you'd like to speak with a manager. I'll be happy to connect you right away. Let me transfer you to a supervisor who can assist you further.",
                "intent": "complaint"
            },
            {
                "query": "How do I reset my password?",
                "response": "I can help you reset your password. Please click on 'Forgot Password' on the login page, enter your email address, and you'll receive a reset link within a few minutes. Let me know if you need any help with this process.",
                "intent": "account_management"
            },
            {
                "query": "I was charged twice for the same order",
                "response": "I apologize for this billing error. Let me look into this immediately. Could you please provide your order number? I'll verify the charges and process a refund for any duplicate payment right away.",
                "intent": "billing_issue"
            },
            {
                "query": "Can I change my shipping address?",
                "response": "I'll help you update the shipping address. Could you provide your order number? If the order hasn't shipped yet, I can update it immediately. If it's already in transit, I'll contact the carrier to redirect it.",
                "intent": "shipping_inquiry"
            },
            {
                "query": "What's your return policy?",
                "response": "Our return policy allows returns within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days after we receive the return. Would you like me to help you initiate a return?",
                "intent": "general_inquiry"
            },
            {
                "query": "The item I received is damaged",
                "response": "I'm very sorry the item arrived damaged. That's not acceptable. I'll arrange a replacement to be sent out immediately, or process a full refund if you prefer. You don't need to return the damaged item. May I have your order number to process this?",
                "intent": "complaint"
            }
        ]

        # Replicate examples to reach n_samples
        while len(data) < n_samples:
            for ex in examples:
                if len(data) >= n_samples:
                    break

                # Add some variation
                item = ex.copy()
                item["messages"] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["query"]},
                    {"role": "assistant", "content": item["response"]}
                ]
                data.append(item)

        print(f"✅ Created {len(data)} synthetic examples")
        return data

    def create_splits(self, data: list) -> tuple:
        """Split into train/val/test"""
        # First split: train+val vs test
        train_val, test = train_test_split(
            data,
            test_size=self.config.test_split,
            random_state=self.config.random_seed
        )

        # Second split: train vs val
        relative_val = self.config.val_split / (1 - self.config.test_split)
        train, val = train_test_split(
            train_val,
            test_size=relative_val,
            random_state=self.config.random_seed
        )

        print(f"\nDataset splits:")
        print(f"  Train: {len(train)} ({len(train)/len(data)*100:.1f}%)")
        print(f"  Val:   {len(val)} ({len(val)/len(data)*100:.1f}%)")
        print(f"  Test:  {len(test)} ({len(test)/len(data)*100:.1f}%)")

        return train, val, test

    def save_datasets(self, train, val, test, output_dir="./data"):
        """Save datasets to JSON files"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/train.json", "w") as f:
            json.dump(train, f, indent=2)

        with open(f"{output_dir}/val.json", "w") as f:
            json.dump(val, f, indent=2)

        with open(f"{output_dir}/test.json", "w") as f:
            json.dump(test, f, indent=2)

        print(f"\n✅ Datasets saved to {output_dir}/")


def main():
    """Main data preparation"""
    print("="*60)
    print("Simplified Data Preparation")
    print("="*60)

    # Initialize
    config = DataConfig()
    prep = SimpleDataPrep(config)

    # Load dataset
    data = prep.load_dataset()

    # Create splits
    train, val, test = prep.create_splits(data)

    # Save
    prep.save_datasets(train, val, test)

    print("\n" + "="*60)
    print("✅ Data preparation complete!")
    print("="*60)
    print("\nNext step: python train_simple.py")


if __name__ == "__main__":
    main()
