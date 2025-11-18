"""
Reward Function for Code Review Agent
Evaluates quality of code review feedback
"""

import re
from dataclasses import dataclass
from typing import Dict
from config import RewardConfig


@dataclass
class RewardComponents:
    """Breakdown of reward components"""
    issue_identification: float = 0.0
    solution_quality: float = 0.0
    explanation_quality: float = 0.0
    tone_score: float = 0.0
    specificity_score: float = 0.0
    safety_score: float = 0.0
    total_reward: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "issue_identification": self.issue_identification,
            "solution_quality": self.solution_quality,
            "explanation_quality": self.explanation_quality,
            "tone_score": self.tone_score,
            "specificity_score": self.specificity_score,
            "safety_score": self.safety_score,
            "total_reward": self.total_reward
        }


class CodeReviewRewardFunction:
    """
    Multi-component reward function for code review quality

    Rewards:
    - Identifies real issues (+5)
    - Provides solutions (+3)
    - Explains reasoning (+2)
    - Constructive tone (+2)
    - Specific feedback (+1)

    Penalties:
    - Wrong issue (-3)
    - Harsh tone (-2)
    - Vague feedback (-1)
    - Suggests vulnerable code (-5)
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns"""

        # Issue identification keywords
        self.issue_keywords = {
            "bug": [r"bug", r"error", r"crash", r"exception", r"will fail"],
            "security": [r"sql injection", r"xss", r"vulnerability", r"insecure", r"plaintext password"],
            "performance": [r"inefficient", r"slow", r"optimize", r"o\(n\^2\)", r"bottleneck"],
            "style": [r"pythonic", r"readable", r"clean code", r"naming"],
            "best_practice": [r"best practice", r"anti-pattern", r"code smell"]
        }

        # Solution patterns
        self.solution_patterns = [
            r"use .+ instead",
            r"replace .+ with",
            r"change .+ to",
            r"add .+",
            r"import .+",
            r"return .+",
            r"raise .+",
        ]

        # Explanation patterns
        self.explanation_patterns = [
            r"because",
            r"this will",
            r"this can",
            r"the reason",
            r"which means",
            r"to prevent",
        ]

        # Constructive tone
        self.constructive_patterns = [
            r"consider",
            r"you could",
            r"try",
            r"suggest",
            r"recommend",
            r"might want to",
        ]

        # Harsh tone
        self.harsh_patterns = [
            r"terrible",
            r"awful",
            r"stupid",
            r"dumb",
            r"worst",
            r"obviously wrong",
        ]

        # Vague feedback
        self.vague_patterns = [
            r"needs improvement",
            r"not good",
            r"bad code",
            r"fix this",
        ]

        # Dangerous suggestions
        self.dangerous_patterns = [
            r"disable security",
            r"ignore warnings",
            r"eval\(",
            r"exec\(",
            r"system\(",
        ]

    def calculate_reward(
        self,
        code: str,
        review: str,
        ground_truth_category: str,
        ground_truth_issue: str = None
    ) -> tuple:
        """
        Calculate reward for a code review

        Args:
            code: Code being reviewed
            review: Generated review comment
            ground_truth_category: Expected issue category
            ground_truth_issue: Expected issue description (optional)

        Returns:
            (reward, components)
        """
        components = RewardComponents()
        review_lower = review.lower()

        # 1. Issue Identification
        components.issue_identification = self._evaluate_issue_identification(
            review_lower, ground_truth_category, ground_truth_issue
        )

        # 2. Solution Quality
        components.solution_quality = self._evaluate_solution(review_lower)

        # 3. Explanation Quality
        components.explanation_quality = self._evaluate_explanation(review_lower)

        # 4. Tone Score
        components.tone_score = self._evaluate_tone(review_lower)

        # 5. Specificity Score
        components.specificity_score = self._evaluate_specificity(review_lower)

        # 6. Safety Score
        components.safety_score = self._evaluate_safety(review_lower)

        # Total reward
        components.total_reward = sum([
            components.issue_identification,
            components.solution_quality,
            components.explanation_quality,
            components.tone_score,
            components.specificity_score,
            components.safety_score
        ])

        # Clip reward
        components.total_reward = max(
            self.config.clip_range[0],
            min(self.config.clip_range[1], components.total_reward)
        )

        return components.total_reward, components

    def _evaluate_issue_identification(
        self, review: str, category: str, issue: str = None
    ) -> float:
        """Check if review identifies the correct issue"""

        if category not in self.issue_keywords:
            return 0.0

        # Check if review mentions category-specific keywords
        keywords = self.issue_keywords[category]
        matches = sum(1 for kw in keywords if re.search(kw, review))

        if matches >= 1:
            return self.config.identifies_issues

        return 0.0

    def _evaluate_solution(self, review: str) -> float:
        """Check if review provides a solution"""

        # Look for solution patterns
        has_solution = any(
            re.search(pattern, review)
            for pattern in self.solution_patterns
        )

        # Also check for code examples
        has_code = "```" in review or "`" in review

        if has_solution or has_code:
            return self.config.provides_solution

        return 0.0

    def _evaluate_explanation(self, review: str) -> float:
        """Check if review explains reasoning"""

        explanation_count = sum(
            1 for pattern in self.explanation_patterns
            if re.search(pattern, review)
        )

        if explanation_count >= 1:
            return self.config.explains_reasoning

        return 0.0

    def _evaluate_tone(self, review: str) -> float:
        """Evaluate if tone is constructive"""

        # Check for harsh language (negative)
        has_harsh = any(
            re.search(pattern, review)
            for pattern in self.harsh_patterns
        )

        if has_harsh:
            return self.config.harsh_tone

        # Check for constructive language (positive)
        has_constructive = any(
            re.search(pattern, review)
            for pattern in self.constructive_patterns
        )

        if has_constructive:
            return self.config.constructive_tone

        return 0.0

    def _evaluate_specificity(self, review: str) -> float:
        """Check if feedback is specific vs vague"""

        # Penalize vague feedback
        is_vague = any(
            re.search(pattern, review)
            for pattern in self.vague_patterns
        )

        if is_vague:
            return self.config.vague_feedback

        # Reward specific feedback (has concrete suggestions)
        word_count = len(review.split())
        if word_count > 20:  # Detailed feedback
            return self.config.specific_feedback

        return 0.0

    def _evaluate_safety(self, review: str) -> float:
        """Check if review suggests dangerous code"""

        has_dangerous = any(
            re.search(pattern, review)
            for pattern in self.dangerous_patterns
        )

        if has_dangerous:
            return self.config.suggests_vulnerable_code

        return 0.0


def test_reward_function():
    """Test reward function with examples"""
    print("Testing Reward Function")
    print("="*60)

    config = RewardConfig()
    reward_fn = CodeReviewRewardFunction(config)

    test_cases = [
        {
            "name": "Good Review (Bug)",
            "code": "return total / len(numbers)",
            "review": "This will raise ZeroDivisionError if the list is empty. Add a check: if not numbers: return 0",
            "category": "bug",
            "issue": "division by zero"
        },
        {
            "name": "Good Review (Security)",
            "review": "This is vulnerable to SQL injection. Use parameterized queries instead: db.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
            "code": "SELECT * FROM users",
            "category": "security",
            "issue": "sql injection"
        },
        {
            "name": "Harsh Tone",
            "code": "for i in range(len(data))",
            "review": "This is terrible code. Obviously wrong. Use list comprehension, stupid!",
            "category": "style",
            "issue": "non-pythonic"
        },
        {
            "name": "Vague Feedback",
            "code": "def process(x): pass",
            "review": "This code needs improvement. Fix this.",
            "category": "bug",
            "issue": "incomplete"
        },
        {
            "name": "No Solution",
            "code": "password = input()",
            "review": "This has a security issue with passwords.",
            "category": "security",
            "issue": "password handling"
        },
    ]

    for test in test_cases:
        print(f"\n{test['name']}")
        print("-" * 60)
        print(f"Review: {test['review']}")

        reward, components = reward_fn.calculate_reward(
            test['code'],
            test['review'],
            test['category'],
            test.get('issue', '')
        )

        print(f"\nReward Breakdown:")
        for key, value in components.to_dict().items():
            if value != 0.0:
                print(f"  {key}: {value:+.2f}")

        print(f"\n✨ Total Reward: {reward:+.2f}")

    print("\n" + "="*60)
    print("✅ Reward function test completed!")


if __name__ == "__main__":
    test_reward_function()
