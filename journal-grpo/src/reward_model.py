"""
Reward Model for Accounting Journal Entry Generation

This module implements a rule-based reward function that scores generated
journal entries based on accounting principles and schema compliance.

Scoring Components:
- Balance check (debits == credits): +1.0 / -1.0 (hard constraint)
- Valid account codes: +0.5
- Schema compliance: +0.3
- Reasonable amounts: +0.2

Total possible score: 2.0
Minimum score: -1.0
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import re

import jsonschema
from jsonschema import validate, ValidationError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JournalEntryRewardModel:
    """
    Reward model for scoring accounting journal entries.

    This class implements a rule-based scoring system that evaluates
    journal entries based on accounting principles and structural validity.
    """

    def __init__(
        self,
        schema_path: Optional[str] = None,
        chart_of_accounts_path: Optional[str] = None,
        tolerance: float = 0.01
    ):
        """
        Initialize the reward model.

        Args:
            schema_path: Path to JSON schema file. If None, uses default path.
            chart_of_accounts_path: Path to chart of accounts file. If None, uses default.
            tolerance: Tolerance for floating-point comparisons (default: 0.01)
        """
        self.tolerance = tolerance

        # Set default paths relative to this file
        base_path = Path(__file__).parent.parent / "data"
        self.schema_path = schema_path or str(base_path / "schema.json")
        self.coa_path = chart_of_accounts_path or str(base_path / "chart_of_accounts.json")

        # Load schema and chart of accounts
        self.schema = self._load_json(self.schema_path)
        self.chart_of_accounts = self._load_chart_of_accounts()

        logger.info(
            f"Initialized RewardModel with {len(self.chart_of_accounts)} accounts"
        )

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Load JSON file and return parsed content."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

    def _load_chart_of_accounts(self) -> Dict[str, Dict[str, str]]:
        """
        Load chart of accounts and create lookup dictionary.

        Returns:
            Dictionary mapping account codes to account details
        """
        coa_data = self._load_json(self.coa_path)
        accounts = {}

        for account in coa_data.get("accounts", []):
            code = account.get("code")
            if code:
                accounts[code] = {
                    "name": account.get("name", ""),
                    "category": account.get("category", ""),
                    "type": account.get("type", ""),
                    "description": account.get("description", "")
                }

        return accounts

    def check_balance(self, entry: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Check if debits equal credits (fundamental accounting equation).

        Args:
            entry: Journal entry dictionary

        Returns:
            Tuple of (is_balanced, score, message)
            Score: +1.0 if balanced, -1.0 if not balanced
        """
        try:
            entries = entry.get("entries", [])

            if not entries:
                return False, -1.0, "No entries found"

            total_debits = sum(float(e.get("debit", 0)) for e in entries)
            total_credits = sum(float(e.get("credit", 0)) for e in entries)

            difference = abs(total_debits - total_credits)
            is_balanced = difference <= self.tolerance

            if is_balanced:
                return True, 1.0, f"Balanced: debits={total_debits:.2f}, credits={total_credits:.2f}"
            else:
                return False, -1.0, f"Unbalanced: debits={total_debits:.2f}, credits={total_credits:.2f}, diff={difference:.2f}"

        except (ValueError, TypeError) as e:
            logger.warning(f"Balance check error: {e}")
            return False, -1.0, f"Error calculating balance: {e}"

    def check_account_codes(self, entry: Dict[str, Any]) -> Tuple[float, str]:
        """
        Check if all account codes are valid.

        Args:
            entry: Journal entry dictionary

        Returns:
            Tuple of (score, message)
            Score: +0.5 if all valid, proportional if partially valid
        """
        entries = entry.get("entries", [])

        if not entries:
            return 0.0, "No entries to validate"

        valid_count = 0
        invalid_codes = []

        for e in entries:
            account_code = e.get("account_code", "")

            # Check if code exists in chart of accounts
            if account_code in self.chart_of_accounts:
                valid_count += 1
            else:
                invalid_codes.append(account_code)

        # Proportional score based on valid codes
        score = (valid_count / len(entries)) * 0.5

        if invalid_codes:
            message = f"Invalid codes: {invalid_codes} ({valid_count}/{len(entries)} valid)"
        else:
            message = f"All {len(entries)} account codes are valid"

        return score, message

    def check_schema_compliance(self, entry: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Check if entry complies with JSON schema.

        Args:
            entry: Journal entry dictionary

        Returns:
            Tuple of (is_valid, score, message)
            Score: +0.3 if compliant, 0.0 if not
        """
        try:
            validate(instance=entry, schema=self.schema)
            return True, 0.3, "Schema validation passed"
        except ValidationError as e:
            # Get the first validation error message
            error_path = " -> ".join(str(p) for p in e.path) if e.path else "root"
            message = f"Schema violation at {error_path}: {e.message[:100]}"
            logger.debug(f"Schema validation failed: {message}")
            return False, 0.0, message
        except Exception as e:
            logger.warning(f"Schema validation error: {e}")
            return False, 0.0, f"Validation error: {str(e)[:100]}"

    def check_reasonable_amounts(self, entry: Dict[str, Any]) -> Tuple[float, str]:
        """
        Check if amounts are reasonable (no negative values, at least one non-zero).

        Args:
            entry: Journal entry dictionary

        Returns:
            Tuple of (score, message)
            Score: +0.2 if all checks pass, proportional otherwise
        """
        entries = entry.get("entries", [])

        if not entries:
            return 0.0, "No entries to check"

        issues = []
        score = 0.2

        # Check 1: No negative debits or credits
        for i, e in enumerate(entries):
            try:
                debit = float(e.get("debit", 0))
                credit = float(e.get("credit", 0))

                if debit < 0:
                    issues.append(f"Entry {i}: negative debit ({debit})")
                    score -= 0.05

                if credit < 0:
                    issues.append(f"Entry {i}: negative credit ({credit})")
                    score -= 0.05

            except (ValueError, TypeError) as e:
                issues.append(f"Entry {i}: invalid amount format")
                score -= 0.05

        # Check 2: At least one non-zero amount
        total_amount = sum(
            float(e.get("debit", 0)) + float(e.get("credit", 0))
            for e in entries
        )

        if total_amount <= self.tolerance:
            issues.append("All amounts are zero")
            score -= 0.05

        # Check 3: Each entry should have either debit OR credit (not both)
        for i, e in enumerate(entries):
            try:
                debit = float(e.get("debit", 0))
                credit = float(e.get("credit", 0))

                if debit > self.tolerance and credit > self.tolerance:
                    issues.append(f"Entry {i}: has both debit and credit")
                    score -= 0.03

            except (ValueError, TypeError):
                pass  # Already caught above

        # Ensure score doesn't go below 0
        score = max(0.0, score)

        if issues:
            message = f"Issues found: {'; '.join(issues[:3])}"
        else:
            message = "All amounts are reasonable"

        return score, message

    def compute_reward(
        self,
        entry: Dict[str, Any],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Compute total reward score for a journal entry.

        Args:
            entry: Journal entry dictionary (either parsed JSON or string)
            verbose: If True, include detailed breakdown in return value

        Returns:
            Dictionary containing:
                - total_score: Float between -1.0 and 2.0
                - breakdown: Dict with individual component scores
                - messages: List of validation messages
                - is_valid: Boolean indicating if entry meets minimum standards
        """
        # Parse entry if it's a string
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
                return {
                    "total_score": -1.0,
                    "breakdown": {},
                    "messages": [f"Invalid JSON: {e}"],
                    "is_valid": False
                }

        # Run all checks
        balance_ok, balance_score, balance_msg = self.check_balance(entry)
        account_score, account_msg = self.check_account_codes(entry)
        schema_ok, schema_score, schema_msg = self.check_schema_compliance(entry)
        amount_score, amount_msg = self.check_reasonable_amounts(entry)

        # Calculate total score
        total_score = balance_score + account_score + schema_score + amount_score

        # Entry is valid if it's balanced and has no schema violations
        is_valid = balance_ok and schema_ok

        result = {
            "total_score": round(total_score, 3),
            "breakdown": {
                "balance": round(balance_score, 3),
                "account_codes": round(account_score, 3),
                "schema": round(schema_score, 3),
                "amounts": round(amount_score, 3)
            },
            "messages": [balance_msg, account_msg, schema_msg, amount_msg],
            "is_valid": is_valid
        }

        if verbose:
            logger.info(f"Reward score: {total_score:.3f} | Valid: {is_valid}")
            for msg in result["messages"]:
                logger.info(f"  - {msg}")

        return result

    def batch_compute_rewards(
        self,
        entries: List[Dict[str, Any]],
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Compute rewards for multiple entries.

        Args:
            entries: List of journal entry dictionaries
            verbose: If True, log detailed information

        Returns:
            List of reward dictionaries
        """
        results = []

        for i, entry in enumerate(entries):
            if verbose:
                logger.info(f"\n--- Evaluating entry {i+1}/{len(entries)} ---")

            result = self.compute_reward(entry, verbose=verbose)
            results.append(result)

        if verbose:
            avg_score = sum(r["total_score"] for r in results) / len(results)
            valid_count = sum(1 for r in results if r["is_valid"])
            logger.info(f"\n=== Batch Summary ===")
            logger.info(f"Average score: {avg_score:.3f}")
            logger.info(f"Valid entries: {valid_count}/{len(results)}")

        return results


def main():
    """Example usage of the reward model."""
    # Initialize reward model
    reward_model = JournalEntryRewardModel()

    # Example 1: Valid journal entry
    valid_entry = {
        "date": "2024-01-15",
        "description": "Sold consulting services for cash",
        "entries": [
            {
                "account_code": "1000",
                "account_name": "Cash",
                "debit": 5000.00,
                "credit": 0.00
            },
            {
                "account_code": "4100",
                "account_name": "Service Revenue",
                "debit": 0.00,
                "credit": 5000.00
            }
        ]
    }

    # Example 2: Unbalanced entry (should get negative score)
    unbalanced_entry = {
        "date": "2024-01-15",
        "description": "Purchase of equipment",
        "entries": [
            {
                "account_code": "1500",
                "account_name": "Equipment",
                "debit": 10000.00,
                "credit": 0.00
            },
            {
                "account_code": "1000",
                "account_name": "Cash",
                "debit": 0.00,
                "credit": 8000.00  # Unbalanced!
            }
        ]
    }

    # Example 3: Invalid account codes
    invalid_codes_entry = {
        "date": "2024-01-15",
        "description": "Rent payment",
        "entries": [
            {
                "account_code": "9999",  # Invalid code
                "account_name": "Rent Expense",
                "debit": 2000.00,
                "credit": 0.00
            },
            {
                "account_code": "1000",
                "account_name": "Cash",
                "debit": 0.00,
                "credit": 2000.00
            }
        ]
    }

    print("\n" + "="*60)
    print("JOURNAL ENTRY REWARD MODEL - EXAMPLES")
    print("="*60)

    # Test examples
    examples = [
        ("Valid Entry", valid_entry),
        ("Unbalanced Entry", unbalanced_entry),
        ("Invalid Account Codes", invalid_codes_entry)
    ]

    for name, entry in examples:
        print(f"\n\n{name}:")
        print("-" * 60)
        result = reward_model.compute_reward(entry, verbose=True)
        print(f"\nTotal Score: {result['total_score']:.3f}")
        print(f"Valid: {result['is_valid']}")
        print("\nBreakdown:")
        for component, score in result['breakdown'].items():
            print(f"  {component:15s}: {score:.3f}")


if __name__ == "__main__":
    main()
