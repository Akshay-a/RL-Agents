"""
Custom Gymnasium Environment for Customer Support Agent Training
Provides RL environment interface for GRPO training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import random

from reward_function import CustomerSupportRewardFunction, RewardComponents
from config import RewardConfig, INTENT_CATEGORIES


class CustomerSupportEnv(gym.Env):
    """
    Gymnasium environment for customer support agent training

    Observation Space:
        - Customer query (text)
        - Context information (intent, metadata)

    Action Space:
        - Agent response (text)

    Reward:
        - Comprehensive reward based on multiple criteria
        - See reward_function.py for details
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dataset_path: str = "./data/train.json",
        reward_config: RewardConfig = None,
        max_steps: int = 1,
        seed: int = 42
    ):
        """
        Initialize customer support environment

        Args:
            dataset_path: Path to training dataset
            reward_config: Reward function configuration
            max_steps: Maximum steps per episode (1 for single-turn conversations)
            seed: Random seed for reproducibility
        """
        super().__init__()

        # Load dataset
        self.dataset = self._load_dataset(dataset_path)
        self.max_steps = max_steps
        self.current_step = 0
        self.current_example = None

        # Initialize reward function
        if reward_config is None:
            reward_config = RewardConfig()
        self.reward_function = CustomerSupportRewardFunction(reward_config)

        # Set random seed
        self.seed_value = seed
        random.seed(seed)
        np.random.seed(seed)

        # Define action and observation spaces
        # For text-based environments, we use Dict spaces
        self.observation_space = spaces.Dict({
            "query": spaces.Text(max_length=512),
            "intent": spaces.Discrete(len(INTENT_CATEGORIES)),
            "context": spaces.Text(max_length=256)
        })

        # Action space is the response text (handled as discrete for practical purposes)
        self.action_space = spaces.Text(max_length=512)

        # Episode tracking
        self.episode_rewards = []
        self.episode_components = []

    def _load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load dataset from JSON file"""
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} examples from {dataset_path}")
            return data
        except FileNotFoundError:
            print(f"Warning: Dataset not found at {dataset_path}")
            print("Creating empty dataset. Run data_prep.py first!")
            return []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset environment to start new episode

        Returns:
            observation: Current state
            info: Additional information
        """
        # Handle seeding
        if seed is not None:
            self.seed_value = seed
            random.seed(seed)
            np.random.seed(seed)

        # Reset episode state
        self.current_step = 0
        self.episode_rewards = []
        self.episode_components = []

        # Sample new customer query
        if len(self.dataset) > 0:
            self.current_example = random.choice(self.dataset)
        else:
            # Fallback if no dataset
            self.current_example = {
                "messages": [
                    {"role": "system", "content": "You are a customer support agent."},
                    {"role": "user", "content": "I need help with my order."},
                    {"role": "assistant", "content": "I'd be happy to help with your order."}
                ],
                "intent": "general_inquiry",
                "metadata": {"example_id": 0}
            }

        # Extract observation
        observation = self._get_observation()

        # Info dictionary
        info = {
            "example_id": self.current_example.get("metadata", {}).get("example_id", 0),
            "ground_truth_intent": self.current_example.get("intent", "general_inquiry"),
            "ground_truth_response": self._get_ground_truth_response()
        }

        return observation, info

    def _get_observation(self) -> Dict:
        """Get current observation from environment"""
        # Extract user message from conversation
        messages = self.current_example.get("messages", [])
        user_message = ""

        for msg in messages:
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        # Get intent index
        intent = self.current_example.get("intent", "general_inquiry")
        intent_idx = INTENT_CATEGORIES.index(intent) if intent in INTENT_CATEGORIES else 0

        # Get context
        metadata = self.current_example.get("metadata", {})
        context = json.dumps(metadata)

        observation = {
            "query": user_message,
            "intent": intent_idx,
            "context": context
        }

        return observation

    def _get_ground_truth_response(self) -> str:
        """Extract ground truth response from current example"""
        messages = self.current_example.get("messages", [])

        for msg in messages:
            if msg["role"] == "assistant":
                return msg["content"]

        return ""

    def step(
        self,
        action: str
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action (generate response) and get reward

        Args:
            action: Agent's response text

        Returns:
            observation: Next state
            reward: Reward for the action
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Get current query and ground truth
        observation = self._get_observation()
        query = observation["query"]
        ground_truth_intent = self.current_example.get("intent", "general_inquiry")

        # Calculate reward for the response
        reward, components = self.reward_function.calculate_reward(
            query=query,
            response=action,
            ground_truth_intent=ground_truth_intent,
            predicted_intent=None,  # Can be provided if we have intent prediction
            metadata=self.current_example.get("metadata", {})
        )

        # Track episode history
        self.episode_rewards.append(reward)
        self.episode_components.append(components)

        # Update step counter
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Prepare info dictionary
        info = {
            "reward_components": components.to_dict(),
            "ground_truth_response": self._get_ground_truth_response(),
            "episode_reward": sum(self.episode_rewards),
            "query": query,
            "response": action
        }

        # Get next observation (same for single-turn)
        next_observation = observation

        return next_observation, reward, terminated, truncated, info

    def render(self):
        """Render environment (print current state)"""
        if self.current_example is None:
            return

        observation = self._get_observation()

        print("\n" + "="*60)
        print("Customer Support Environment")
        print("="*60)
        print(f"\nQuery: {observation['query']}")
        print(f"Intent: {INTENT_CATEGORIES[observation['intent']]}")
        print(f"Ground Truth: {self._get_ground_truth_response()}")

        if self.episode_rewards:
            print(f"\nLast Reward: {self.episode_rewards[-1]:.2f}")
            print(f"Episode Total: {sum(self.episode_rewards):.2f}")

    def close(self):
        """Clean up environment"""
        pass

    def get_episode_stats(self) -> Dict:
        """Get statistics for current episode"""
        if not self.episode_rewards:
            return {}

        return {
            "total_reward": sum(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards),
            "num_steps": len(self.episode_rewards),
            "reward_components": [c.to_dict() for c in self.episode_components]
        }


class CustomerSupportVectorEnv:
    """
    Vectorized environment for parallel training
    Supports batch processing of multiple environments
    """

    def __init__(
        self,
        num_envs: int,
        dataset_path: str = "./data/train.json",
        reward_config: RewardConfig = None,
        seed: int = 42
    ):
        """
        Initialize vectorized environment

        Args:
            num_envs: Number of parallel environments
            dataset_path: Path to training dataset
            reward_config: Reward function configuration
            seed: Random seed
        """
        self.num_envs = num_envs
        self.envs = [
            CustomerSupportEnv(
                dataset_path=dataset_path,
                reward_config=reward_config,
                seed=seed + i
            )
            for i in range(num_envs)
        ]

    def reset(self) -> Tuple[List[Dict], List[Dict]]:
        """Reset all environments"""
        observations = []
        infos = []

        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)

        return observations, infos

    def step(self, actions: List[str]) -> Tuple[List[Dict], List[float], List[bool], List[bool], List[Dict]]:
        """Execute actions in all environments"""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        return observations, rewards, terminateds, truncateds, infos

    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()


def test_environment():
    """Test the customer support environment"""
    print("Testing Customer Support Environment")
    print("="*60)

    # Create environment
    env = CustomerSupportEnv(dataset_path="./data/train.json")

    # Reset environment
    observation, info = env.reset()

    print("\n📋 Initial Observation:")
    print(f"Query: {observation['query']}")
    print(f"Intent: {INTENT_CATEGORIES[observation['intent']]}")
    print(f"Ground Truth: {info['ground_truth_response']}")

    # Test with different responses
    test_responses = [
        "I'd be happy to help you with that. Could you please provide more details?",
        "Whatever. Figure it out yourself.",
        "Sure, I'll process an immediate refund without checking anything."
    ]

    for i, response in enumerate(test_responses):
        print(f"\n--- Test Response {i+1} ---")
        print(f"Response: {response}")

        # Reset for each test
        observation, info = env.reset()

        # Take step
        next_obs, reward, terminated, truncated, step_info = env.step(response)

        print(f"\n💰 Reward: {reward:.2f}")
        print("\nReward Components:")
        for key, value in step_info['reward_components'].items():
            if value != 0.0:
                print(f"  {key}: {value:+.2f}")

    env.close()

    print("\n" + "="*60)
    print("✅ Environment test completed!")


if __name__ == "__main__":
    test_environment()
