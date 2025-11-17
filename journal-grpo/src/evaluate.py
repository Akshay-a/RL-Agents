"""
Model Evaluation Script

Evaluates trained models on the golden test set.
Compares:
- Zero-shot base model
- SFT-only model
- SFT + GRPO model

Metrics:
- Balance accuracy (% with debits == credits)
- Average reward score
- Schema compliance rate
- Account code accuracy
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
from reward_model import JournalEntryRewardModel


def load_config(config_path: str = "../configs/grpo_config.yaml") -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_test_data(test_path: str) -> List[Dict]:
    """Load golden test set."""
    examples = []
    with open(test_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def load_model_and_tokenizer(model_path: str, base_model: str):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    return model, tokenizer


def generate_entry(model, tokenizer, prompt: str, config: Dict) -> str:
    """Generate journal entry for a given prompt."""
    # Format prompt
    full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Generate
    gen_config = config['generation']
    outputs = model.generate(
        **inputs,
        max_new_tokens=gen_config['max_new_tokens'],
        do_sample=gen_config['do_sample'],
        temperature=0.7,  # Slightly lower for evaluation
        top_p=gen_config['top_p'],
        repetition_penalty=gen_config['repetition_penalty'],
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return generated_text


def extract_json(text: str) -> str:
    """Extract JSON from generated text."""
    # Try to extract JSON from code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # If still has extra text, try to find JSON object
    if text.startswith("{"):
        # Find matching closing brace
        brace_count = 0
        for i, char in enumerate(text):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[:i+1]

    return text


def evaluate_model(model, tokenizer, test_data: List[Dict], reward_model: JournalEntryRewardModel, config: Dict) -> Dict:
    """
    Evaluate model on test set.

    Returns metrics dictionary.
    """
    results = {
        "total": len(test_data),
        "balanced": 0,
        "valid_schema": 0,
        "valid_accounts": 0,
        "rewards": [],
        "generations": []
    }

    print(f"Evaluating on {len(test_data)} examples...")

    for example in tqdm(test_data):
        prompt = example["prompt"]

        # Generate
        generated_text = generate_entry(model, tokenizer, prompt, config)

        # Extract JSON
        json_text = extract_json(generated_text)

        # Score with reward model
        score_result = reward_model.compute_reward(json_text)

        # Update metrics
        results["rewards"].append(score_result["total_score"])

        if score_result["breakdown"]["balance"] > 0:
            results["balanced"] += 1

        if score_result["breakdown"]["schema"] > 0:
            results["valid_schema"] += 1

        if score_result["breakdown"]["account_codes"] > 0:
            results["valid_accounts"] += 1

        # Store generation for inspection
        results["generations"].append({
            "prompt": prompt,
            "generated": json_text,
            "score": score_result["total_score"],
            "breakdown": score_result["breakdown"]
        })

    # Calculate percentages
    results["balance_accuracy"] = results["balanced"] / results["total"]
    results["schema_compliance"] = results["valid_schema"] / results["total"]
    results["account_accuracy"] = results["valid_accounts"] / results["total"]
    results["avg_reward"] = sum(results["rewards"]) / len(results["rewards"])

    return results


def print_results(model_name: str, results: Dict):
    """Print evaluation results."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*70}")
    print(f"Test examples: {results['total']}")
    print(f"Balance accuracy: {results['balance_accuracy']*100:.1f}%")
    print(f"Schema compliance: {results['schema_compliance']*100:.1f}%")
    print(f"Account accuracy: {results['account_accuracy']*100:.1f}%")
    print(f"Average reward: {results['avg_reward']:.3f}")
    print(f"{'='*70}\n")

    # Show some examples
    print("Sample Generations:")
    print("-" * 70)
    for i, gen in enumerate(results["generations"][:3]):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {gen['prompt']}")
        print(f"Score: {gen['score']:.3f}")
        print(f"Breakdown: {gen['breakdown']}")
        print(f"Generated (first 200 chars): {gen['generated'][:200]}...")
    print()


def compare_models(config: Dict):
    """
    Compare multiple models on the test set.

    Models to compare:
    1. Zero-shot base model
    2. SFT-only
    3. SFT + GRPO
    """
    # Load test data
    test_data = load_test_data(config['data']['test_set'])
    print(f"Loaded {len(test_data)} test examples\n")

    # Initialize reward model
    reward_model = JournalEntryRewardModel()

    # Evaluate base model
    print("\n" + "="*70)
    print("Evaluating: Base Model (Zero-shot)")
    print("="*70)
    base_model, base_tokenizer = load_model_and_tokenizer(config['model_name'], config['model_name'])
    base_results = evaluate_model(base_model, base_tokenizer, test_data, reward_model, config)
    print_results("Base Model (Zero-shot)", base_results)

    # Evaluate SFT model
    print("\n" + "="*70)
    print("Evaluating: SFT Model")
    print("="*70)
    sft_checkpoint = os.path.join(config['sft']['output_dir'], "final")
    if os.path.exists(sft_checkpoint):
        sft_model, sft_tokenizer = load_model_and_tokenizer(sft_checkpoint, config['model_name'])
        sft_results = evaluate_model(sft_model, sft_tokenizer, test_data, reward_model, config)
        print_results("SFT Model", sft_results)
    else:
        print(f"SFT checkpoint not found at {sft_checkpoint}")
        sft_results = None

    # Evaluate GRPO model
    print("\n" + "="*70)
    print("Evaluating: GRPO Model (SFT + GRPO)")
    print("="*70)
    grpo_checkpoint = os.path.join(config['grpo']['output_dir'], "final")
    if os.path.exists(grpo_checkpoint):
        grpo_model, grpo_tokenizer = load_model_and_tokenizer(grpo_checkpoint, config['model_name'])
        grpo_results = evaluate_model(grpo_model, grpo_tokenizer, test_data, reward_model, config)
        print_results("GRPO Model (SFT + GRPO)", grpo_results)
    else:
        print(f"GRPO checkpoint not found at {grpo_checkpoint}")
        grpo_results = None

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Model':<25} {'Balance Acc':<15} {'Avg Reward':<15}")
    print("-" * 70)
    print(f"{'Base (Zero-shot)':<25} {base_results['balance_accuracy']*100:>6.1f}%         {base_results['avg_reward']:>6.3f}")
    if sft_results:
        print(f"{'SFT':<25} {sft_results['balance_accuracy']*100:>6.1f}%         {sft_results['avg_reward']:>6.3f}")
    if grpo_results:
        print(f"{'SFT + GRPO':<25} {grpo_results['balance_accuracy']*100:>6.1f}%         {grpo_results['avg_reward']:>6.3f}")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Load config
    config = load_config("configs/grpo_config.yaml")

    # Check if test data exists
    if not os.path.exists(config['data']['test_set']):
        print(f"ERROR: Test data not found at {config['data']['test_set']}")
        sys.exit(1)

    # Run comparison
    compare_models(config)


if __name__ == "__main__":
    main()
