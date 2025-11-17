"""
Unit Tests for Journal Entry Reward Model

This module contains comprehensive tests for the reward model including:
- Balance validation
- Account code validation
- Schema compliance
- Amount reasonability checks
"""

import unittest
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from reward_model import JournalEntryRewardModel


class TestJournalEntryRewardModel(unittest.TestCase):
    """Test cases for the JournalEntryRewardModel class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests."""
        cls.reward_model = JournalEntryRewardModel()

    def test_initialization(self):
        """Test that reward model initializes correctly."""
        self.assertIsNotNone(self.reward_model.schema)
        self.assertIsNotNone(self.reward_model.chart_of_accounts)
        self.assertGreater(len(self.reward_model.chart_of_accounts), 0)
        self.assertEqual(len(self.reward_model.chart_of_accounts), 25)

    def test_valid_balanced_entry(self):
        """Test a perfectly valid and balanced journal entry."""
        entry = {
            "date": "2024-01-15",
            "description": "Sold services for cash",
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

        result = self.reward_model.compute_reward(entry)

        # Should get maximum score (2.0)
        self.assertEqual(result["total_score"], 2.0)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["breakdown"]["balance"], 1.0)
        self.assertEqual(result["breakdown"]["account_codes"], 0.5)
        self.assertEqual(result["breakdown"]["schema"], 0.3)
        self.assertEqual(result["breakdown"]["amounts"], 0.2)

    def test_unbalanced_entry(self):
        """Test an unbalanced entry (debits != credits)."""
        entry = {
            "date": "2024-01-15",
            "description": "Unbalanced transaction",
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
                    "credit": 3000.00  # Not balanced!
                }
            ]
        }

        result = self.reward_model.compute_reward(entry)

        # Should get balance penalty (-1.0) but other components add back to 0.0
        # Total: -1.0 (balance) + 0.5 (codes) + 0.3 (schema) + 0.2 (amounts) = 0.0
        self.assertEqual(result["total_score"], 0.0)
        self.assertEqual(result["breakdown"]["balance"], -1.0)
        self.assertFalse(result["is_valid"])

    def test_invalid_account_codes(self):
        """Test entry with invalid account codes."""
        entry = {
            "date": "2024-01-15",
            "description": "Entry with invalid codes",
            "entries": [
                {
                    "account_code": "9999",  # Invalid
                    "account_name": "Fake Account",
                    "debit": 1000.00,
                    "credit": 0.00
                },
                {
                    "account_code": "8888",  # Invalid
                    "account_name": "Another Fake",
                    "debit": 0.00,
                    "credit": 1000.00
                }
            ]
        }

        result = self.reward_model.compute_reward(entry)

        # Balance is correct but account codes are invalid
        self.assertEqual(result["breakdown"]["balance"], 1.0)
        self.assertEqual(result["breakdown"]["account_codes"], 0.0)

    def test_partial_valid_account_codes(self):
        """Test entry with mix of valid and invalid account codes."""
        entry = {
            "date": "2024-01-15",
            "description": "Mixed validity",
            "entries": [
                {
                    "account_code": "1000",  # Valid
                    "account_name": "Cash",
                    "debit": 1000.00,
                    "credit": 0.00
                },
                {
                    "account_code": "9999",  # Invalid
                    "account_name": "Fake Account",
                    "debit": 0.00,
                    "credit": 1000.00
                }
            ]
        }

        result = self.reward_model.compute_reward(entry)

        # Should get 50% of account code score (1 out of 2 valid)
        self.assertEqual(result["breakdown"]["account_codes"], 0.25)

    def test_schema_violation_missing_date(self):
        """Test entry missing required 'date' field."""
        entry = {
            # "date" missing
            "description": "Entry without date",
            "entries": [
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": 1000.00,
                    "credit": 0.00
                },
                {
                    "account_code": "4100",
                    "account_name": "Service Revenue",
                    "debit": 0.00,
                    "credit": 1000.00
                }
            ]
        }

        result = self.reward_model.compute_reward(entry)

        # Schema validation should fail
        self.assertEqual(result["breakdown"]["schema"], 0.0)
        self.assertFalse(result["is_valid"])

    def test_schema_violation_invalid_date_format(self):
        """Test entry with invalid date format."""
        entry = {
            "date": "01/15/2024",  # Wrong format (should be YYYY-MM-DD)
            "description": "Invalid date format",
            "entries": [
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": 1000.00,
                    "credit": 0.00
                },
                {
                    "account_code": "4100",
                    "account_name": "Service Revenue",
                    "debit": 0.00,
                    "credit": 1000.00
                }
            ]
        }

        result = self.reward_model.compute_reward(entry)

        # Schema validation should fail due to date format
        self.assertEqual(result["breakdown"]["schema"], 0.0)

    def test_negative_amounts(self):
        """Test entry with negative debit/credit amounts."""
        entry = {
            "date": "2024-01-15",
            "description": "Negative amounts",
            "entries": [
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": -1000.00,  # Negative!
                    "credit": 0.00
                },
                {
                    "account_code": "4100",
                    "account_name": "Service Revenue",
                    "debit": 0.00,
                    "credit": 1000.00
                }
            ]
        }

        result = self.reward_model.compute_reward(entry)

        # Amount check should penalize negative values
        self.assertLess(result["breakdown"]["amounts"], 0.2)

    def test_both_debit_and_credit(self):
        """Test entry where a line has both debit and credit (not allowed)."""
        entry = {
            "date": "2024-01-15",
            "description": "Both debit and credit",
            "entries": [
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": 1000.00,
                    "credit": 500.00  # Should only have one
                },
                {
                    "account_code": "4100",
                    "account_name": "Service Revenue",
                    "debit": 0.00,
                    "credit": 500.00
                }
            ]
        }

        result = self.reward_model.compute_reward(entry)

        # Amount check should penalize this
        self.assertLess(result["breakdown"]["amounts"], 0.2)

    def test_zero_amounts(self):
        """Test entry with all zero amounts."""
        entry = {
            "date": "2024-01-15",
            "description": "All zeros",
            "entries": [
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": 0.00,
                    "credit": 0.00
                },
                {
                    "account_code": "4100",
                    "account_name": "Service Revenue",
                    "debit": 0.00,
                    "credit": 0.00
                }
            ]
        }

        result = self.reward_model.compute_reward(entry)

        # Should be balanced but penalized for zero amounts
        self.assertEqual(result["breakdown"]["balance"], 1.0)
        self.assertLess(result["breakdown"]["amounts"], 0.2)

    def test_complex_multi_line_entry(self):
        """Test complex entry with multiple debit/credit lines."""
        entry = {
            "date": "2024-01-15",
            "description": "Purchase equipment with cash and loan",
            "entries": [
                {
                    "account_code": "1500",
                    "account_name": "Equipment",
                    "debit": 50000.00,
                    "credit": 0.00
                },
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": 0.00,
                    "credit": 20000.00
                },
                {
                    "account_code": "2500",
                    "account_name": "Notes Payable",
                    "debit": 0.00,
                    "credit": 30000.00
                }
            ]
        }

        result = self.reward_model.compute_reward(entry)

        # Should be perfectly valid
        self.assertEqual(result["total_score"], 2.0)
        self.assertTrue(result["is_valid"])

    def test_json_string_input(self):
        """Test that reward model can handle JSON string input."""
        entry_str = json.dumps({
            "date": "2024-01-15",
            "description": "Test JSON string",
            "entries": [
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": 1000.00,
                    "credit": 0.00
                },
                {
                    "account_code": "4100",
                    "account_name": "Service Revenue",
                    "debit": 0.00,
                    "credit": 1000.00
                }
            ]
        })

        result = self.reward_model.compute_reward(entry_str)

        # Should parse and validate correctly
        self.assertEqual(result["total_score"], 2.0)
        self.assertTrue(result["is_valid"])

    def test_invalid_json_string(self):
        """Test handling of invalid JSON string."""
        invalid_json = "{ this is not valid json }"

        result = self.reward_model.compute_reward(invalid_json)

        # Should return error score
        self.assertEqual(result["total_score"], -1.0)
        self.assertFalse(result["is_valid"])

    def test_batch_compute_rewards(self):
        """Test batch processing of multiple entries."""
        entries = [
            {
                "date": "2024-01-15",
                "description": "Entry 1",
                "entries": [
                    {"account_code": "1000", "account_name": "Cash", "debit": 100, "credit": 0},
                    {"account_code": "4100", "account_name": "Revenue", "debit": 0, "credit": 100}
                ]
            },
            {
                "date": "2024-01-16",
                "description": "Entry 2",
                "entries": [
                    {"account_code": "5100", "account_name": "Salaries", "debit": 200, "credit": 0},
                    {"account_code": "1000", "account_name": "Cash", "debit": 0, "credit": 200}
                ]
            }
        ]

        results = self.reward_model.batch_compute_rewards(entries)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["is_valid"] for r in results))
        self.assertTrue(all(r["total_score"] == 2.0 for r in results))

    def test_floating_point_tolerance(self):
        """Test that small floating-point differences are tolerated."""
        entry = {
            "date": "2024-01-15",
            "description": "Floating point test",
            "entries": [
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": 1000.005,  # Tiny difference
                    "credit": 0.00
                },
                {
                    "account_code": "4100",
                    "account_name": "Service Revenue",
                    "debit": 0.00,
                    "credit": 1000.00
                }
            ]
        }

        result = self.reward_model.compute_reward(entry)

        # Should still be considered balanced (within tolerance)
        self.assertEqual(result["breakdown"]["balance"], 1.0)

    def test_empty_entries_array(self):
        """Test handling of entry with empty entries array."""
        entry = {
            "date": "2024-01-15",
            "description": "No entries",
            "entries": []
        }

        result = self.reward_model.compute_reward(entry)

        # Should fail schema validation (minItems: 2)
        self.assertEqual(result["breakdown"]["schema"], 0.0)
        self.assertFalse(result["is_valid"])

    def test_account_code_format(self):
        """Test that account codes must be 4-digit format."""
        entry = {
            "date": "2024-01-15",
            "description": "Invalid code format",
            "entries": [
                {
                    "account_code": "100",  # Only 3 digits
                    "account_name": "Cash",
                    "debit": 1000.00,
                    "credit": 0.00
                },
                {
                    "account_code": "4100",
                    "account_name": "Revenue",
                    "debit": 0.00,
                    "credit": 1000.00
                }
            ]
        }

        result = self.reward_model.compute_reward(entry)

        # Should fail schema validation
        self.assertEqual(result["breakdown"]["schema"], 0.0)


class TestRewardModelEdgeCases(unittest.TestCase):
    """Test edge cases and unusual inputs."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.reward_model = JournalEntryRewardModel()

    def test_very_large_amounts(self):
        """Test handling of very large monetary amounts."""
        entry = {
            "date": "2024-01-15",
            "description": "Large transaction",
            "entries": [
                {
                    "account_code": "1000",
                    "account_name": "Cash",
                    "debit": 999999999.99,
                    "credit": 0.00
                },
                {
                    "account_code": "4100",
                    "account_name": "Revenue",
                    "debit": 0.00,
                    "credit": 999999999.99
                }
            ]
        }

        result = self.reward_model.compute_reward(entry)

        # Should still validate correctly
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["breakdown"]["balance"], 1.0)

    def test_many_line_items(self):
        """Test entry with many line items."""
        entries_list = [
            {"account_code": "1000", "account_name": "Cash", "debit": 1000, "credit": 0}
        ]

        # Add multiple expense accounts
        for i, code in enumerate(["5100", "5200", "5300", "5400"]):
            entries_list.append({
                "account_code": code,
                "account_name": f"Expense {i}",
                "debit": 0,
                "credit": 250
            })

        entry = {
            "date": "2024-01-15",
            "description": "Multiple expenses",
            "entries": entries_list
        }

        result = self.reward_model.compute_reward(entry)

        # Should be valid
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["breakdown"]["balance"], 1.0)


def run_tests():
    """Run all tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestJournalEntryRewardModel))
    suite.addTests(loader.loadTestsFromTestCase(TestRewardModelEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
