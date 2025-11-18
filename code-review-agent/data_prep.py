"""
Data Preparation for Code Review Agent
Creates synthetic code review examples
"""

import json
import random
from sklearn.model_selection import train_test_split
from config import DataConfig, SYSTEM_PROMPT


class CodeReviewDataPrep:
    """Prepare code review training data"""

    def __init__(self, config: DataConfig):
        self.config = config
        random.seed(config.random_seed)

    def create_dataset(self, n_samples: int = 300) -> list:
        """Create synthetic code review dataset"""
        print(f"Creating {n_samples} code review examples...")

        # Real code review examples with issues
        examples = [
            # Bug examples
            {
                "code": '''def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)''',
                "issue": "Division by zero error if empty list",
                "review": "This function will raise a ZeroDivisionError if an empty list is passed. Add a check: `if not numbers: return 0` or raise a ValueError.",
                "category": "bug"
            },
            {
                "code": '''def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)''',
                "issue": "SQL injection vulnerability",
                "review": "This code is vulnerable to SQL injection. Use parameterized queries instead: `db.execute(\"SELECT * FROM users WHERE id = ?\", (user_id,))`",
                "category": "security"
            },
            {
                "code": '''def find_item(items, target):
    for item in items:
        if item == target:
            return True
    return False''',
                "issue": "Inefficient search pattern",
                "review": "This can be simplified to `return target in items`, which is more Pythonic and efficient for lists. For large datasets, consider using a set.",
                "category": "performance"
            },
            {
                "code": '''def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result''',
                "issue": "Non-Pythonic iteration",
                "review": "Use list comprehension or direct iteration: `return [x * 2 for x in data]`. This is more readable and Pythonic than index-based iteration.",
                "category": "style"
            },
            {
                "code": '''def save_password(username, password):
    with open('passwords.txt', 'a') as f:
        f.write(f"{username}:{password}\\n")''',
                "issue": "Storing passwords in plaintext",
                "review": "Never store passwords in plaintext! Use a secure hashing algorithm like bcrypt: `import bcrypt; hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())`",
                "category": "security"
            },
            {
                "code": '''def fetch_data():
    try:
        response = requests.get(url)
        return response.json()
    except:
        pass''',
                "issue": "Bare except clause and silent failure",
                "review": "Avoid bare except clauses. Catch specific exceptions and handle errors properly: `except requests.RequestException as e: logger.error(f'Failed to fetch: {e}'); raise`",
                "category": "best_practice"
            },
            {
                "code": '''class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email''',
                "issue": "Missing validation and documentation",
                "review": "Add input validation (email format) and a docstring explaining the class purpose and parameters. Consider using @dataclass for simpler data classes.",
                "category": "documentation"
            },
            {
                "code": '''def multiply(a, b):
    return a * b

result = multiply(5, 3)''',
                "issue": "No tests",
                "review": "Add unit tests to verify edge cases: `assert multiply(5, 3) == 15`, `assert multiply(0, 10) == 0`, `assert multiply(-2, 3) == -6`",
                "category": "testing"
            },
            {
                "code": '''def do_stuff(x, y, z, flag1, flag2, flag3):
    if flag1:
        if flag2:
            if flag3:
                return x + y + z
    return 0''',
                "issue": "Deep nesting and unclear logic",
                "review": "Refactor to reduce nesting: `if not (flag1 and flag2 and flag3): return 0` or `if all([flag1, flag2, flag3]): return sum([x, y, z])`",
                "category": "clean_code"
            },
            {
                "code": '''def send_email(to, subject, body):
    # TODO: implement email sending
    print(f"Sending to {to}")''',
                "issue": "Incomplete implementation",
                "review": "This function has a TODO comment indicating incomplete implementation. Either complete it or raise NotImplementedError to make it explicit.",
                "category": "bug"
            },
        ]

        # Generate dataset by replicating and varying examples
        dataset = []
        while len(dataset) < n_samples:
            for ex in examples:
                if len(dataset) >= n_samples:
                    break

                # Create training example
                example = {
                    "code": ex["code"],
                    "category": ex["category"],
                    "issue": ex["issue"],
                    "review": ex["review"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Review this code:\n\n```python\n{ex['code']}\n```"},
                        {"role": "assistant", "content": ex["review"]}
                    ]
                }
                dataset.append(example)

        print(f"✅ Created {len(dataset)} examples")
        return dataset

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
    print("Code Review Agent - Data Preparation")
    print("="*60)

    config = DataConfig()
    prep = CodeReviewDataPrep(config)

    # Create dataset
    data = prep.create_dataset(n_samples=300)

    # Create splits
    train, val, test = prep.create_splits(data)

    # Save
    prep.save_datasets(train, val, test)

    print("\n" + "="*60)
    print("✅ Data preparation complete!")
    print("="*60)
    print("\nNext step: python train_grpo.py")


if __name__ == "__main__":
    main()
