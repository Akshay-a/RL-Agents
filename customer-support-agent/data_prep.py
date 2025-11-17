"""
Data Preparation Script for Customer Support Dataset
Downloads, cleans, and formats the Bitext customer support dataset
"""

import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import random
from config import DataConfig, SYSTEM_PROMPT, INTENT_CATEGORIES

class CustomerSupportDataPrep:
    """Handles data loading and preprocessing for customer support training"""

    def __init__(self, config: DataConfig):
        self.config = config
        random.seed(config.random_seed)

    def load_bitext_dataset(self) -> pd.DataFrame:
        """Load Bitext customer support dataset from HuggingFace"""
        print(f"Loading dataset: {self.config.dataset_name}")

        try:
            dataset = load_dataset(self.config.dataset_name)
            df = pd.DataFrame(dataset['train'])
            print(f"Loaded {len(df)} examples from Bitext dataset")

            # Limit dataset size if specified
            if self.config.max_samples and len(df) > self.config.max_samples:
                df = df.sample(n=self.config.max_samples, random_state=self.config.random_seed)
                print(f"Limited to {len(df)} samples")

            return df

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating synthetic dataset for testing...")
            return self._create_synthetic_dataset()

    def _create_synthetic_dataset(self, n_samples: int = 100) -> pd.DataFrame:
        """Create synthetic customer support data for testing"""
        synthetic_data = []

        # Sample queries for different intents
        templates = {
            "refund_request": [
                "I want a refund for order {order_id}",
                "Can I get my money back? I'm not satisfied",
                "How do I request a refund for my recent purchase?"
            ],
            "order_tracking": [
                "Where is my order {order_id}?",
                "Can you track my package?",
                "I haven't received my order yet"
            ],
            "product_inquiry": [
                "Tell me about product {product_name}",
                "What are the specs of {product_name}?",
                "Is {product_name} available in different colors?"
            ],
            "complaint": [
                "I'm very unhappy with the service",
                "This is unacceptable, I want to speak to a manager",
                "Your product broke after one day!"
            ],
            "technical_support": [
                "My {product_name} isn't working",
                "I need help setting up my device",
                "There's an error message on my screen"
            ]
        }

        # Sample responses
        responses = {
            "refund_request": "I understand you'd like a refund. I'll be happy to help you with that. Our refund policy allows returns within 30 days of purchase for unused items in original packaging. May I have your order number to check the details?",
            "order_tracking": "I'd be glad to help you track your order. Could you please provide your order number? I'll look up the current status and estimated delivery date for you.",
            "product_inquiry": "I'd be happy to provide information about that product. Let me share the key features and specifications with you. Is there anything specific you'd like to know?",
            "complaint": "I sincerely apologize for the frustration you've experienced. Your satisfaction is very important to us. I'd like to understand the issue better so I can help resolve it. Could you please share more details about what happened?",
            "technical_support": "I'm sorry you're having trouble with your device. I'll do my best to help you resolve this. Let me walk you through some troubleshooting steps. First, have you tried restarting the device?"
        }

        for i in range(n_samples):
            intent = random.choice(list(templates.keys()))
            query_template = random.choice(templates[intent])

            # Fill in placeholders
            query = query_template.format(
                order_id=f"ORD{random.randint(10000, 99999)}",
                product_name=random.choice(["smartphone", "laptop", "headphones", "tablet"])
            )

            synthetic_data.append({
                "instruction": query,
                "response": responses[intent],
                "category": intent,
                "intent": intent
            })

        return pd.DataFrame(synthetic_data)

    def format_for_training(self, df: pd.DataFrame) -> List[Dict]:
        """Format dataset for training with proper prompt structure"""
        formatted_data = []

        for idx, row in df.iterrows():
            # Extract relevant fields
            instruction = row.get('instruction', row.get('query', row.get('customer_message', '')))
            response = row.get('response', row.get('agent_response', ''))
            intent = row.get('intent', row.get('category', 'general_inquiry'))

            # Create conversation format
            formatted_example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ],
                "intent": intent,
                "metadata": {
                    "example_id": idx,
                    "requires_escalation": self._check_escalation_needed(instruction, response),
                    "complexity": self._assess_complexity(instruction)
                }
            }

            formatted_data.append(formatted_example)

        return formatted_data

    def _check_escalation_needed(self, instruction: str, response: str) -> bool:
        """Check if query requires escalation to human agent"""
        escalation_keywords = [
            "manager", "supervisor", "legal", "lawyer", "sue",
            "terrible", "worst", "unacceptable", "disgusting"
        ]

        text = (instruction + " " + response).lower()
        return any(keyword in text for keyword in escalation_keywords)

    def _assess_complexity(self, instruction: str) -> str:
        """Assess query complexity: simple, medium, complex"""
        word_count = len(instruction.split())

        if word_count < 10:
            return "simple"
        elif word_count < 25:
            return "medium"
        else:
            return "complex"

    def create_train_val_test_split(
        self,
        formatted_data: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train, validation, and test sets"""

        # First split: train + val vs test
        train_val, test = train_test_split(
            formatted_data,
            test_size=self.config.test_split,
            random_state=self.config.random_seed
        )

        # Second split: train vs val
        relative_val_size = self.config.val_split / (1 - self.config.test_split)
        train, val = train_test_split(
            train_val,
            test_size=relative_val_size,
            random_state=self.config.random_seed
        )

        print(f"\nDataset split:")
        print(f"  Train: {len(train)} examples ({len(train)/len(formatted_data)*100:.1f}%)")
        print(f"  Val:   {len(val)} examples ({len(val)/len(formatted_data)*100:.1f}%)")
        print(f"  Test:  {len(test)} examples ({len(test)/len(formatted_data)*100:.1f}%)")

        return train, val, test

    def save_datasets(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        test_data: List[Dict],
        output_dir: str = "./data"
    ):
        """Save processed datasets to JSON files"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/train.json", "w") as f:
            json.dump(train_data, f, indent=2)

        with open(f"{output_dir}/val.json", "w") as f:
            json.dump(val_data, f, indent=2)

        with open(f"{output_dir}/test.json", "w") as f:
            json.dump(test_data, f, indent=2)

        print(f"\nDatasets saved to {output_dir}/")

    def analyze_dataset(self, df: pd.DataFrame):
        """Print dataset statistics"""
        print("\n" + "="*50)
        print("Dataset Analysis")
        print("="*50)

        print(f"\nTotal examples: {len(df)}")

        # Intent distribution
        if 'intent' in df.columns or 'category' in df.columns:
            intent_col = 'intent' if 'intent' in df.columns else 'category'
            print(f"\nIntent distribution:")
            print(df[intent_col].value_counts())

        # Text length statistics
        if 'instruction' in df.columns:
            df['query_length'] = df['instruction'].str.split().str.len()
            print(f"\nQuery length statistics:")
            print(f"  Mean: {df['query_length'].mean():.1f} words")
            print(f"  Median: {df['query_length'].median():.1f} words")
            print(f"  Min: {df['query_length'].min()} words")
            print(f"  Max: {df['query_length'].max()} words")


def main():
    """Main data preparation pipeline"""
    print("Starting data preparation pipeline...")

    # Initialize config and data prep
    config = DataConfig()
    data_prep = CustomerSupportDataPrep(config)

    # Load dataset
    df = data_prep.load_bitext_dataset()

    # Analyze dataset
    data_prep.analyze_dataset(df)

    # Format for training
    formatted_data = data_prep.format_for_training(df)

    # Create splits
    train, val, test = data_prep.create_train_val_test_split(formatted_data)

    # Save datasets
    data_prep.save_datasets(train, val, test)

    print("\n✅ Data preparation completed successfully!")
    print("\nNext steps:")
    print("  1. Review the data in ./data/ directory")
    print("  2. Run reward function tests: python reward_function.py")
    print("  3. Start training: python train_grpo.py")


if __name__ == "__main__":
    main()
