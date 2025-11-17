"""
Reward Function System for Customer Support AI Agent
Implements comprehensive reward calculation based on multiple criteria
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from config import RewardConfig, COMPANY_POLICIES, INTENT_CATEGORIES


@dataclass
class RewardComponents:
    """Breakdown of reward components for transparency"""
    intent_score: float = 0.0
    policy_score: float = 0.0
    empathy_score: float = 0.0
    escalation_score: float = 0.0
    tone_score: float = 0.0
    safety_score: float = 0.0
    resolution_score: float = 0.0
    total_reward: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "intent_score": self.intent_score,
            "policy_score": self.policy_score,
            "empathy_score": self.empathy_score,
            "escalation_score": self.escalation_score,
            "tone_score": self.tone_score,
            "safety_score": self.safety_score,
            "resolution_score": self.resolution_score,
            "total_reward": self.total_reward
        }


class CustomerSupportRewardFunction:
    """
    Comprehensive reward function for customer support agent
    Evaluates responses based on multiple criteria:
    - Intent classification accuracy
    - Policy compliance
    - Empathy and tone
    - Escalation appropriateness
    - Safety and harmful content
    - Resolution completeness
    """

    def __init__(self, config: RewardConfig):
        self.config = config

        # Compile patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for policy and safety checks"""

        # Policy violation patterns
        self.policy_violation_patterns = [
            r"immediate refund",
            r"refund.*without",
            r"override.*policy",
            r"i'll.*share.*personal",
            r"i'll.*give.*credit card",
            r"send.*password",
        ]

        # Empathy indicators
        self.empathy_patterns = [
            r"i understand",
            r"i appreciate",
            r"i'm sorry",
            r"apologize",
            r"frustrating",
            r"inconvenience",
        ]

        # Professional tone indicators
        self.professional_patterns = [
            r"thank you",
            r"i'd be happy",
            r"i'll be glad",
            r"let me help",
            r"may i",
            r"would you",
        ]

        # Inappropriate tone patterns
        self.inappropriate_patterns = [
            r"whatever",
            r"not my problem",
            r"too bad",
            r"deal with it",
            r"lol",
            r"😂",
        ]

        # Dangerous/harmful patterns
        self.dangerous_patterns = [
            r"ignore.*return policy",
            r"don't.*worry.*policy",
            r"just.*lie",
            r"fake.*receipt",
            r"hack",
            r"steal",
        ]

        # Escalation triggers
        self.escalation_triggers = [
            r"speak.*manager",
            r"talk.*supervisor",
            r"file.*complaint",
            r"legal action",
            r"lawyer",
            r"sue",
            r"this is unacceptable",
        ]

        # Resolution indicators
        self.resolution_patterns = [
            r"i'll.*process",
            r"i've.*created",
            r"ticket.*number",
            r"expect.*email",
            r"next steps",
            r"follow.*up",
        ]

        # Proactive suggestion patterns
        self.proactive_patterns = [
            r"you might also",
            r"i recommend",
            r"consider",
            r"in the future",
            r"to prevent",
        ]

    def calculate_reward(
        self,
        query: str,
        response: str,
        ground_truth_intent: str,
        predicted_intent: str = None,
        metadata: Dict = None
    ) -> Tuple[float, RewardComponents]:
        """
        Calculate comprehensive reward for a response

        Args:
            query: Customer query/message
            response: Agent's response
            ground_truth_intent: Correct intent label
            predicted_intent: Model's predicted intent (optional)
            metadata: Additional context (optional)

        Returns:
            total_reward: Float reward value
            components: Breakdown of reward components
        """
        components = RewardComponents()

        # 1. Intent Classification Score
        if predicted_intent:
            if predicted_intent == ground_truth_intent:
                components.intent_score = self.config.intent_classification_correct
            else:
                components.intent_score = self.config.wrong_intent

        # 2. Policy Compliance Score
        components.policy_score = self._evaluate_policy_compliance(response)

        # 3. Empathy & Tone Score
        components.empathy_score = self._evaluate_empathy(response)
        components.tone_score = self._evaluate_tone(response)

        # 4. Escalation Appropriateness Score
        components.escalation_score = self._evaluate_escalation(query, response)

        # 5. Safety Score (critical - negative if dangerous)
        components.safety_score = self._evaluate_safety(response)

        # 6. Resolution & Proactivity Bonuses
        components.resolution_score = self._evaluate_resolution(response)

        # Calculate total reward
        components.total_reward = sum([
            components.intent_score,
            components.policy_score,
            components.empathy_score,
            components.tone_score,
            components.escalation_score,
            components.safety_score,
            components.resolution_score
        ])

        # Apply normalization and clipping
        if self.config.clip_rewards:
            components.total_reward = np.clip(
                components.total_reward,
                self.config.reward_clip_range[0],
                self.config.reward_clip_range[1]
            )

        return components.total_reward, components

    def _evaluate_policy_compliance(self, response: str) -> float:
        """Check if response follows company policies"""
        response_lower = response.lower()

        # Check for policy violations
        for pattern in self.policy_violation_patterns:
            if re.search(pattern, response_lower):
                return self.config.policy_violation

        # If no violations, give positive reward
        return self.config.policy_compliance

    def _evaluate_empathy(self, response: str) -> float:
        """Evaluate empathy in response"""
        response_lower = response.lower()

        empathy_count = sum(
            1 for pattern in self.empathy_patterns
            if re.search(pattern, response_lower)
        )

        # Reward increases with more empathy indicators (up to max)
        if empathy_count >= 2:
            return self.config.empathy_tone
        elif empathy_count == 1:
            return self.config.empathy_tone * 0.5
        else:
            return 0.0

    def _evaluate_tone(self, response: str) -> float:
        """Evaluate professional tone"""
        response_lower = response.lower()

        # Check for inappropriate tone (negative)
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, response_lower):
                return self.config.inappropriate_tone

        # Check for professional tone (positive)
        professional_count = sum(
            1 for pattern in self.professional_patterns
            if re.search(pattern, response_lower)
        )

        if professional_count >= 1:
            return self.config.empathy_tone * 0.5  # Half of empathy score
        else:
            return 0.0

    def _evaluate_escalation(self, query: str, response: str) -> float:
        """Evaluate if escalation is handled appropriately"""
        query_lower = query.lower()
        response_lower = response.lower()

        # Check if customer is requesting escalation
        needs_escalation = any(
            re.search(pattern, query_lower)
            for pattern in self.escalation_triggers
        )

        if needs_escalation:
            # Check if agent acknowledges and escalates
            escalation_keywords = ["transfer", "connect", "manager", "supervisor", "escalate"]
            if any(keyword in response_lower for keyword in escalation_keywords):
                return self.config.proper_escalation
            else:
                # Customer wants escalation but agent didn't provide it
                return -1.0
        else:
            # No escalation needed - neutral
            return 0.0

    def _evaluate_safety(self, response: str) -> float:
        """Check for dangerous or harmful content"""
        response_lower = response.lower()

        for pattern in self.dangerous_patterns:
            if re.search(pattern, response_lower):
                return self.config.dangerous_response

        # No dangerous content - neutral (not positive)
        return 0.0

    def _evaluate_resolution(self, response: str) -> float:
        """Evaluate if response provides complete resolution"""
        response_lower = response.lower()
        score = 0.0

        # Check for resolution indicators
        has_resolution = any(
            re.search(pattern, response_lower)
            for pattern in self.resolution_patterns
        )

        if has_resolution:
            score += self.config.complete_resolution

        # Check for proactive suggestions
        has_proactive = any(
            re.search(pattern, response_lower)
            for pattern in self.proactive_patterns
        )

        if has_proactive:
            score += self.config.proactive_suggestions

        return score

    def batch_calculate_rewards(
        self,
        queries: List[str],
        responses: List[str],
        ground_truth_intents: List[str],
        predicted_intents: List[str] = None
    ) -> Tuple[List[float], List[RewardComponents]]:
        """Calculate rewards for a batch of query-response pairs"""

        if predicted_intents is None:
            predicted_intents = [None] * len(queries)

        rewards = []
        components_list = []

        for query, response, gt_intent, pred_intent in zip(
            queries, responses, ground_truth_intents, predicted_intents
        ):
            reward, components = self.calculate_reward(
                query, response, gt_intent, pred_intent
            )
            rewards.append(reward)
            components_list.append(components)

        return rewards, components_list


def test_reward_function():
    """Test reward function with example cases"""
    print("Testing Reward Function")
    print("="*60)

    config = RewardConfig()
    reward_fn = CustomerSupportRewardFunction(config)

    # Test cases
    test_cases = [
        {
            "name": "Good Response",
            "query": "I want a refund for my order",
            "response": "I understand you'd like a refund. I'd be happy to help with that. Our refund policy allows returns within 30 days for unused items. May I have your order number to process this?",
            "ground_truth_intent": "refund_request",
            "predicted_intent": "refund_request"
        },
        {
            "name": "Policy Violation",
            "query": "Can I get a refund?",
            "response": "Sure! I'll process an immediate refund without checking anything.",
            "ground_truth_intent": "refund_request",
            "predicted_intent": "refund_request"
        },
        {
            "name": "Inappropriate Tone",
            "query": "Where is my order?",
            "response": "Whatever. Check the tracking yourself lol",
            "ground_truth_intent": "order_tracking",
            "predicted_intent": "order_tracking"
        },
        {
            "name": "Wrong Intent",
            "query": "I want to track my order",
            "response": "Let me help you with your refund request.",
            "ground_truth_intent": "order_tracking",
            "predicted_intent": "refund_request"
        },
        {
            "name": "Proper Escalation",
            "query": "I want to speak to a manager NOW!",
            "response": "I understand your frustration. I'll be happy to connect you with a manager right away. Let me transfer your call.",
            "ground_truth_intent": "complaint",
            "predicted_intent": "complaint"
        },
        {
            "name": "Dangerous Response",
            "query": "How can I get a refund after 60 days?",
            "response": "Just lie about when you bought it and ignore the return policy.",
            "ground_truth_intent": "refund_request",
            "predicted_intent": "refund_request"
        }
    ]

    for test in test_cases:
        print(f"\n{test['name']}")
        print("-" * 60)
        print(f"Query: {test['query']}")
        print(f"Response: {test['response']}")

        reward, components = reward_fn.calculate_reward(
            test['query'],
            test['response'],
            test['ground_truth_intent'],
            test['predicted_intent']
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
