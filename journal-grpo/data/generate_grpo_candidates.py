"""
Generate GRPO Training Prompts

This script creates diverse prompts for GRPO training. Unlike SFT, we don't
need labeled outputs - the reward model will grade whatever the model generates.

Strategy:
- Maximum diversity in phrasing and transaction types
- Include edge cases and complex scenarios
- Mix of simple and multi-line transactions
- Various amount ranges and date formats in prompts
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List


class GRPOPromptGenerator:
    """Generate diverse prompts for GRPO training."""

    def __init__(self):
        """Initialize the prompt generator."""
        # Transaction templates with variations
        self.templates = self._load_templates()

    def _load_templates(self) -> List[dict]:
        """Load prompt templates for different transaction types."""
        return [
            # Cash sales
            {
                "type": "cash_sale",
                "templates": [
                    "Record the sale of {product} for ${amount:,.2f} cash on {date}",
                    "We sold {product} for ${amount:,.2f} in cash on {date}",
                    "Cash sale: {product} for ${amount:,.2f} ({date})",
                    "On {date}, sold {product} and received ${amount:,.2f} cash",
                    "Received ${amount:,.2f} cash for {product} on {date}",
                ],
                "products": [
                    "consulting services", "software licenses", "products",
                    "professional services", "training programs", "maintenance services",
                    "subscription services", "digital products", "merchandise",
                    "computer equipment", "office supplies"
                ],
                "amounts": (500, 25000)
            },

            # Credit sales
            {
                "type": "credit_sale",
                "templates": [
                    "Record a sale on account for ${amount:,.2f} on {date}",
                    "We sold products on credit for ${amount:,.2f} on {date}",
                    "On {date}, made a credit sale of ${amount:,.2f}",
                    "Invoice customer ${amount:,.2f} for services rendered on {date}",
                    "Sold merchandise on account: ${amount:,.2f} ({date})",
                ],
                "amounts": (1000, 40000)
            },

            # Expense payments
            {
                "type": "expense",
                "templates": [
                    "Paid ${amount:,.2f} for {expense} on {date}",
                    "Record payment of {expense}: ${amount:,.2f} on {date}",
                    "We paid {expense} of ${amount:,.2f} on {date}",
                    "On {date}, paid ${amount:,.2f} for {expense}",
                    "{expense_cap} payment: ${amount:,.2f} ({date})",
                ],
                "expenses": [
                    ("rent", 1500, 6000),
                    ("utilities", 200, 1500),
                    ("salaries", 3000, 20000),
                    ("insurance", 500, 3000),
                    ("advertising", 500, 5000),
                    ("office supplies", 100, 1000),
                    ("telephone and internet", 150, 800),
                    ("professional fees", 1000, 8000),
                ]
            },

            # Asset purchases - cash
            {
                "type": "asset_purchase_cash",
                "templates": [
                    "Purchased {asset} for ${amount:,.2f} cash on {date}",
                    "Bought {asset} paying ${amount:,.2f} cash on {date}",
                    "On {date}, acquired {asset} for ${amount:,.2f} cash",
                    "Record purchase of {asset}: ${amount:,.2f} cash ({date})",
                ],
                "assets": [
                    ("equipment", 5000, 60000),
                    ("a company vehicle", 15000, 70000),
                    ("inventory", 2000, 40000),
                    ("furniture", 2000, 15000),
                    ("computer systems", 3000, 25000),
                ]
            },

            # Asset purchases - credit
            {
                "type": "asset_purchase_credit",
                "templates": [
                    "Purchased {asset} on account for ${amount:,.2f} on {date}",
                    "Bought {asset} on credit: ${amount:,.2f} ({date})",
                    "On {date}, acquired {asset} on account for ${amount:,.2f}",
                    "Record credit purchase of {asset} for ${amount:,.2f} on {date}",
                ],
                "assets": [
                    ("equipment", 10000, 80000),
                    ("machinery", 20000, 100000),
                    ("inventory", 5000, 50000),
                ]
            },

            # Loan transactions
            {
                "type": "loan",
                "templates": [
                    "Received a ${amount:,.2f} bank loan on {date}",
                    "Took out a loan for ${amount:,.2f} on {date}",
                    "On {date}, received loan proceeds of ${amount:,.2f}",
                    "Bank deposited ${amount:,.2f} loan into our account on {date}",
                ],
                "amounts": (10000, 150000)
            },

            # Owner investments
            {
                "type": "investment",
                "templates": [
                    "Owner invested ${amount:,.2f} cash on {date}",
                    "Record owner contribution of ${amount:,.2f} on {date}",
                    "On {date}, owner added ${amount:,.2f} capital to the business",
                    "Owner deposited ${amount:,.2f} into business account on {date}",
                ],
                "amounts": (5000, 60000)
            },

            # Collections
            {
                "type": "collection",
                "templates": [
                    "Collected ${amount:,.2f} from customer on account on {date}",
                    "Received ${amount:,.2f} payment from customer on {date}",
                    "Customer paid ${amount:,.2f} on their account on {date}",
                    "On {date}, received ${amount:,.2f} to settle customer account",
                ],
                "amounts": (1000, 30000)
            },

            # Payments to suppliers
            {
                "type": "payment_payable",
                "templates": [
                    "Paid ${amount:,.2f} to supplier to reduce accounts payable on {date}",
                    "Made payment of ${amount:,.2f} on account on {date}",
                    "On {date}, paid ${amount:,.2f} to settle supplier invoice",
                    "Reduced accounts payable by paying ${amount:,.2f} on {date}",
                ],
                "amounts": (1000, 25000)
            },

            # Depreciation
            {
                "type": "depreciation",
                "templates": [
                    "Record monthly depreciation expense of ${amount:,.2f} on {date}",
                    "Depreciation for the month: ${amount:,.2f} ({date})",
                    "On {date}, record ${amount:,.2f} depreciation on equipment",
                    "Monthly depreciation entry: ${amount:,.2f} for {date}",
                ],
                "amounts": (300, 4000)
            },

            # Complex transactions
            {
                "type": "complex",
                "templates": [
                    "Purchased {asset} for ${total:,.2f} - paid ${cash:,.2f} cash and financed ${loan:,.2f} on {date}",
                    "On {date}, bought {asset} for ${total:,.2f}: ${cash:,.2f} down payment, rest financed",
                    "Acquired {asset} on {date} for ${total:,.2f} ({cash_pct}% cash, rest on loan)",
                ],
                "assets": ["equipment", "a vehicle", "machinery"],
                "amounts": (30000, 120000),
                "cash_percent": (20, 40)
            },
        ]

    def _random_date(self, year: int = None) -> str:
        """Generate a random date."""
        if year is None:
            year = random.choice([2023, 2024])

        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)
        random_date = start + timedelta(days=random.randint(0, (end - start).days))
        return random_date.strftime("%Y-%m-%d")

    def _random_amount(self, min_val: float, max_val: float) -> float:
        """Generate a random amount."""
        if random.random() < 0.6:
            # Round numbers
            return round(random.uniform(min_val, max_val) / 100) * 100
        else:
            # Exact amounts
            return round(random.uniform(min_val, max_val), 2)

    def generate_prompt(self) -> str:
        """Generate a single prompt."""
        template_group = random.choice(self.templates)
        template = random.choice(template_group["templates"])

        # Generate date
        date = self._random_date()

        # Generate based on template type
        if template_group["type"] == "cash_sale":
            product = random.choice(template_group["products"])
            amount = self._random_amount(*template_group["amounts"])
            return template.format(product=product, amount=amount, date=date)

        elif template_group["type"] == "credit_sale":
            amount = self._random_amount(*template_group["amounts"])
            return template.format(amount=amount, date=date)

        elif template_group["type"] == "expense":
            expense_info = random.choice(template_group["expenses"])
            expense_name = expense_info[0]
            amount = self._random_amount(expense_info[1], expense_info[2])
            return template.format(expense=expense_name, expense_cap=expense_name.capitalize(), amount=amount, date=date)

        elif template_group["type"] in ["asset_purchase_cash", "asset_purchase_credit"]:
            asset_info = random.choice(template_group["assets"])
            asset_name = asset_info[0]
            amount = self._random_amount(asset_info[1], asset_info[2])
            return template.format(asset=asset_name, amount=amount, date=date)

        elif template_group["type"] in ["loan", "investment", "collection", "payment_payable", "depreciation"]:
            amount = self._random_amount(*template_group["amounts"])
            return template.format(amount=amount, date=date)

        elif template_group["type"] == "complex":
            asset = random.choice(template_group["assets"])
            total = self._random_amount(*template_group["amounts"])
            cash_pct = random.randint(*template_group["cash_percent"])
            cash = round(total * (cash_pct / 100), 2)
            loan = round(total - cash, 2)
            return template.format(
                asset=asset,
                total=total,
                cash=cash,
                loan=loan,
                cash_pct=cash_pct,
                date=date
            )

        return template

    def generate_dataset(self, num_prompts: int = 2000) -> List[str]:
        """
        Generate a dataset of prompts.

        Args:
            num_prompts: Number of prompts to generate

        Returns:
            List of prompt strings
        """
        prompts = []

        print(f"Generating {num_prompts} GRPO prompts...")

        for i in range(num_prompts):
            prompt = self.generate_prompt()
            prompts.append(prompt)

            if (i + 1) % 200 == 0:
                print(f"Generated {i + 1}/{num_prompts} prompts")

        # Remove duplicates while preserving order
        unique_prompts = list(dict.fromkeys(prompts))

        # If we lost too many to duplicates, generate more
        while len(unique_prompts) < num_prompts:
            additional_needed = num_prompts - len(unique_prompts)
            print(f"Removed {len(prompts) - len(unique_prompts)} duplicates, generating {additional_needed} more...")

            for _ in range(additional_needed):
                unique_prompts.append(self.generate_prompt())

            unique_prompts = list(dict.fromkeys(unique_prompts))

        print(f"\nDataset generation complete!")
        print(f"Total unique prompts: {len(unique_prompts[:num_prompts])}")

        return unique_prompts[:num_prompts]


def main():
    """Generate and save GRPO prompts."""
    # Initialize generator
    print("Initializing GRPO prompt generator...")
    generator = GRPOPromptGenerator()

    # Generate prompts
    prompts = generator.generate_dataset(num_prompts=2000)

    # Save to JSONL file
    output_path = Path(__file__).parent / "grpo_prompts.jsonl"
    print(f"\nSaving to {output_path}...")

    with open(output_path, 'w') as f:
        for prompt in prompts:
            f.write(json.dumps({"prompt": prompt}) + '\n')

    print(f"Successfully saved {len(prompts)} prompts to {output_path}")

    # Show samples
    print("\n" + "="*70)
    print("SAMPLE PROMPTS")
    print("="*70)
    for i, prompt in enumerate(random.sample(prompts, 10), 1):
        print(f"{i}. {prompt}")


if __name__ == "__main__":
    main()
