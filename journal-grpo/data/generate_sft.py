"""
Generate Supervised Fine-Tuning (SFT) Training Data

This script creates high-quality labeled examples for the initial supervised
fine-tuning phase. All examples are validated through the reward model to
ensure they meet quality standards (score >= 1.8).

Strategy:
- Template-based generation for consistency
- Variety across transaction types
- All examples validated before saving
"""

import json
import random
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reward_model import JournalEntryRewardModel


class SFTDataGenerator:
    """Generate supervised fine-tuning examples for journal entries."""

    def __init__(self, reward_model: JournalEntryRewardModel):
        """
        Initialize the data generator.

        Args:
            reward_model: Reward model for validating generated examples
        """
        self.reward_model = reward_model
        self.valid_accounts = reward_model.chart_of_accounts

        # Load account categories for easier template creation
        self._categorize_accounts()

    def _categorize_accounts(self):
        """Organize accounts by category for template generation."""
        self.accounts_by_category = {
            "Asset": [],
            "Liability": [],
            "Equity": [],
            "Revenue": [],
            "Expense": []
        }

        for code, details in self.valid_accounts.items():
            category = details["category"]
            if category in self.accounts_by_category:
                self.accounts_by_category[category].append({
                    "code": code,
                    "name": details["name"],
                    "type": details["type"]
                })

    def _random_date(self, start_year: int = 2023, end_year: int = 2024) -> str:
        """
        Generate a random date in ISO format.

        Args:
            start_year: Starting year
            end_year: Ending year

        Returns:
            Date string in YYYY-MM-DD format
        """
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 12, 31)
        random_date = start + timedelta(days=random.randint(0, (end - start).days))
        return random_date.strftime("%Y-%m-%d")

    def _random_amount(self, min_val: float = 100, max_val: float = 50000) -> float:
        """Generate a random monetary amount."""
        # Generate amounts that are "round" numbers more often
        if random.random() < 0.7:
            # Round to nearest 100
            return round(random.uniform(min_val, max_val) / 100) * 100
        else:
            # Exact amounts
            return round(random.uniform(min_val, max_val), 2)

    def generate_cash_sale(self) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a cash sale transaction.

        Returns:
            Tuple of (prompt, journal_entry)
        """
        amount = self._random_amount(500, 20000)
        date = self._random_date()

        services = [
            "consulting services",
            "software licenses",
            "products",
            "professional services",
            "training services",
            "maintenance services"
        ]
        service = random.choice(services)

        prompt = f"Record the sale of {service} for ${amount:,.2f} cash on {date}"

        entry = {
            "date": date,
            "description": f"Sold {service} for cash",
            "entries": [
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": amount,
                    "credit": 0.00
                },
                {
                    "account_code": "4100" if "service" in service else "4000",
                    "account_name": "Service Revenue" if "service" in service else "Sales Revenue",
                    "debit": 0.00,
                    "credit": amount
                }
            ]
        }

        return prompt, entry

    def generate_credit_sale(self) -> Tuple[str, Dict[str, Any]]:
        """Generate a sale on account (accounts receivable)."""
        amount = self._random_amount(1000, 30000)
        date = self._random_date()

        prompt = f"Record a sale on account for ${amount:,.2f} on {date}"

        entry = {
            "date": date,
            "description": "Sale on account",
            "entries": [
                {
                    "account_code": "1100",
                    "account_name": "Accounts Receivable",
                    "debit": amount,
                    "credit": 0.00
                },
                {
                    "account_code": "4000",
                    "account_name": "Sales Revenue",
                    "debit": 0.00,
                    "credit": amount
                }
            ]
        }

        return prompt, entry

    def generate_expense_payment(self) -> Tuple[str, Dict[str, Any]]:
        """Generate an expense payment transaction."""
        expenses = [
            ("5200", "Rent Expense", "rent", 1500, 5000),
            ("5300", "Utilities Expense", "utilities", 200, 1000),
            ("5100", "Salaries Expense", "salaries", 3000, 15000),
        ]

        expense_code, expense_name, expense_desc, min_amt, max_amt = random.choice(expenses)
        amount = self._random_amount(min_amt, max_amt)
        date = self._random_date()

        prompt = f"Record payment of ${amount:,.2f} for {expense_desc} on {date}"

        entry = {
            "date": date,
            "description": f"Paid {expense_desc}",
            "entries": [
                {
                    "account_code": expense_code,
                    "account_name": expense_name,
                    "debit": amount,
                    "credit": 0.00
                },
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": 0.00,
                    "credit": amount
                }
            ]
        }

        return prompt, entry

    def generate_asset_purchase_cash(self) -> Tuple[str, Dict[str, Any]]:
        """Generate an asset purchase with cash."""
        assets = [
            ("1500", "Equipment", "equipment", 5000, 50000),
            ("1700", "Vehicles", "a vehicle", 15000, 60000),
            ("1200", "Inventory", "inventory", 2000, 30000),
        ]

        asset_code, asset_name, asset_desc, min_amt, max_amt = random.choice(assets)
        amount = self._random_amount(min_amt, max_amt)
        date = self._random_date()

        prompt = f"Record purchase of {asset_desc} for ${amount:,.2f} cash on {date}"

        entry = {
            "date": date,
            "description": f"Purchased {asset_desc}",
            "entries": [
                {
                    "account_code": asset_code,
                    "account_name": asset_name,
                    "debit": amount,
                    "credit": 0.00
                },
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": 0.00,
                    "credit": amount
                }
            ]
        }

        return prompt, entry

    def generate_asset_purchase_credit(self) -> Tuple[str, Dict[str, Any]]:
        """Generate an asset purchase on credit."""
        amount = self._random_amount(10000, 80000)
        date = self._random_date()

        prompt = f"Record purchase of equipment for ${amount:,.2f} on account on {date}"

        entry = {
            "date": date,
            "description": "Purchased equipment on account",
            "entries": [
                {
                    "account_code": "1500",
                    "account_name": "Equipment",
                    "debit": amount,
                    "credit": 0.00
                },
                {
                    "account_code": "2000",
                    "account_name": "Accounts Payable",
                    "debit": 0.00,
                    "credit": amount
                }
            ]
        }

        return prompt, entry

    def generate_loan_received(self) -> Tuple[str, Dict[str, Any]]:
        """Generate a loan receipt transaction."""
        amount = self._random_amount(10000, 100000)
        date = self._random_date()

        prompt = f"Record receipt of a ${amount:,.2f} bank loan on {date}"

        entry = {
            "date": date,
            "description": "Received bank loan",
            "entries": [
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": amount,
                    "credit": 0.00
                },
                {
                    "account_code": "2600",
                    "account_name": "Loans Payable",
                    "debit": 0.00,
                    "credit": amount
                }
            ]
        }

        return prompt, entry

    def generate_owner_investment(self) -> Tuple[str, Dict[str, Any]]:
        """Generate an owner investment transaction."""
        amount = self._random_amount(5000, 50000)
        date = self._random_date()

        prompt = f"Record owner investment of ${amount:,.2f} cash on {date}"

        entry = {
            "date": date,
            "description": "Owner investment",
            "entries": [
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": amount,
                    "credit": 0.00
                },
                {
                    "account_code": "3000",
                    "account_name": "Owner's Capital",
                    "debit": 0.00,
                    "credit": amount
                }
            ]
        }

        return prompt, entry

    def generate_depreciation(self) -> Tuple[str, Dict[str, Any]]:
        """Generate a depreciation expense transaction."""
        amount = self._random_amount(500, 3000)
        date = self._random_date()

        prompt = f"Record depreciation expense of ${amount:,.2f} for equipment on {date}"

        entry = {
            "date": date,
            "description": "Monthly depreciation expense",
            "entries": [
                {
                    "account_code": "5400",
                    "account_name": "Depreciation Expense",
                    "debit": amount,
                    "credit": 0.00
                },
                {
                    "account_code": "1600",
                    "account_name": "Accumulated Depreciation - Equipment",
                    "debit": 0.00,
                    "credit": amount
                }
            ]
        }

        return prompt, entry

    def generate_payment_on_account(self) -> Tuple[str, Dict[str, Any]]:
        """Generate a payment to reduce accounts payable."""
        amount = self._random_amount(1000, 20000)
        date = self._random_date()

        prompt = f"Record payment of ${amount:,.2f} to settle accounts payable on {date}"

        entry = {
            "date": date,
            "description": "Payment on account",
            "entries": [
                {
                    "account_code": "2000",
                    "account_name": "Accounts Payable",
                    "debit": amount,
                    "credit": 0.00
                },
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": 0.00,
                    "credit": amount
                }
            ]
        }

        return prompt, entry

    def generate_collection_on_account(self) -> Tuple[str, Dict[str, Any]]:
        """Generate collection from accounts receivable."""
        amount = self._random_amount(1000, 25000)
        date = self._random_date()

        prompt = f"Record collection of ${amount:,.2f} from customer on account on {date}"

        entry = {
            "date": date,
            "description": "Collection from customer",
            "entries": [
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": amount,
                    "credit": 0.00
                },
                {
                    "account_code": "1100",
                    "account_name": "Accounts Receivable",
                    "debit": 0.00,
                    "credit": amount
                }
            ]
        }

        return prompt, entry

    def generate_complex_transaction(self) -> Tuple[str, Dict[str, Any]]:
        """Generate a more complex transaction with multiple line items."""
        equipment_cost = self._random_amount(30000, 100000)
        cash_paid = equipment_cost * random.uniform(0.2, 0.4)
        loan_amount = equipment_cost - cash_paid
        date = self._random_date()

        prompt = f"Record purchase of equipment for ${equipment_cost:,.2f} - paid ${cash_paid:,.2f} cash and financed the rest with a loan on {date}"

        entry = {
            "date": date,
            "description": "Purchased equipment with cash and loan",
            "entries": [
                {
                    "account_code": "1500",
                    "account_name": "Equipment",
                    "debit": round(equipment_cost, 2),
                    "credit": 0.00
                },
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": 0.00,
                    "credit": round(cash_paid, 2)
                },
                {
                    "account_code": "2500",
                    "account_name": "Notes Payable",
                    "debit": 0.00,
                    "credit": round(loan_amount, 2)
                }
            ]
        }

        return prompt, entry

    def generate_dataset(self, num_examples: int = 500, min_score: float = 1.8) -> List[Dict[str, Any]]:
        """
        Generate a complete SFT dataset.

        Args:
            num_examples: Number of examples to generate
            min_score: Minimum reward score to accept (default: 1.8)

        Returns:
            List of training examples
        """
        generators = [
            (self.generate_cash_sale, 0.20),           # 20%
            (self.generate_credit_sale, 0.10),         # 10%
            (self.generate_expense_payment, 0.20),     # 20%
            (self.generate_asset_purchase_cash, 0.10), # 10%
            (self.generate_asset_purchase_credit, 0.05), # 5%
            (self.generate_loan_received, 0.05),       # 5%
            (self.generate_owner_investment, 0.05),    # 5%
            (self.generate_depreciation, 0.05),        # 5%
            (self.generate_payment_on_account, 0.08),  # 8%
            (self.generate_collection_on_account, 0.08), # 8%
            (self.generate_complex_transaction, 0.04), # 4%
        ]

        dataset = []
        rejected_count = 0

        print(f"Generating {num_examples} SFT examples...")
        print(f"Minimum reward score: {min_score}")

        for i in range(num_examples):
            # Select generator based on weights
            generator = random.choices(
                [g[0] for g in generators],
                weights=[g[1] for g in generators]
            )[0]

            # Generate example
            prompt, entry = generator()

            # Validate through reward model
            score_result = self.reward_model.compute_reward(entry)

            if score_result["total_score"] >= min_score and score_result["is_valid"]:
                # Format for training
                training_example = {
                    "prompt": prompt,
                    "completion": json.dumps(entry, indent=2),
                    "metadata": {
                        "reward_score": score_result["total_score"],
                        "breakdown": score_result["breakdown"]
                    }
                }
                dataset.append(training_example)

                if (i + 1) % 50 == 0:
                    print(f"Generated {i + 1}/{num_examples} examples (rejected: {rejected_count})")
            else:
                rejected_count += 1
                print(f"Rejected example {i+1} (score: {score_result['total_score']:.2f})")

        print(f"\nDataset generation complete!")
        print(f"Total examples: {len(dataset)}")
        print(f"Rejected: {rejected_count}")

        # Calculate statistics
        avg_score = sum(ex["metadata"]["reward_score"] for ex in dataset) / len(dataset)
        print(f"Average reward score: {avg_score:.3f}")

        return dataset


def main():
    """Generate and save SFT training data."""
    # Initialize reward model
    print("Initializing reward model...")
    reward_model = JournalEntryRewardModel()

    # Initialize generator
    generator = SFTDataGenerator(reward_model)

    # Generate dataset
    dataset = generator.generate_dataset(num_examples=500, min_score=1.8)

    # Save to JSONL file
    output_path = Path(__file__).parent / "sft_train.jsonl"
    print(f"\nSaving to {output_path}...")

    with open(output_path, 'w') as f:
        for example in dataset:
            f.write(json.dumps(example) + '\n')

    print(f"Successfully saved {len(dataset)} examples to {output_path}")

    # Show a sample
    print("\n" + "="*70)
    print("SAMPLE TRAINING EXAMPLE")
    print("="*70)
    sample = random.choice(dataset)
    print(f"\nPrompt:\n{sample['prompt']}")
    print(f"\nCompletion:\n{sample['completion']}")
    print(f"\nReward Score: {sample['metadata']['reward_score']:.3f}")


if __name__ == "__main__":
    main()
