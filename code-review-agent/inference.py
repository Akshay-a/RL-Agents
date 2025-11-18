"""
Inference Script for Code Review Agent
CLI and API interfaces for deployment
"""

import torch

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

from config import SYSTEM_PROMPT


class CodeReviewAgent:
    """Inference wrapper for code review agent"""

    def __init__(self, model_path: str = "./checkpoints/best_model"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = self._load_model()
        print("✅ Model loaded!")

    def _load_model(self):
        """Load trained model"""
        if UNSLOTH_AVAILABLE:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=1024,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto"
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def review_code(
        self,
        code: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        Generate code review

        Args:
            code: Code snippet to review
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Review comment
        """
        prompt = f"{SYSTEM_PROMPT}\n\nReview this code:\n\n```python\n{code}\n```\n\nReview:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Review:" in response:
            response = response.split("Review:")[-1].strip()

        return response

    def interactive_mode(self):
        """Interactive CLI for code review"""
        print("\n" + "="*60)
        print("Code Review Agent - Interactive Mode")
        print("="*60)
        print("\nCommands:")
        print("  - Enter code (end with empty line)")
        print("  - Type 'quit' to exit")
        print("  - Type 'help' for help")
        print()

        while True:
            print("\n📝 Enter code to review (empty line to finish, 'quit' to exit):")

            lines = []
            while True:
                line = input()
                if line.lower() == 'quit':
                    print("\n👋 Goodbye!")
                    return

                if line == "":
                    break

                lines.append(line)

            if not lines:
                continue

            code = "\n".join(lines)

            print("\n🤖 Reviewing...")
            review = self.review_code(code)

            print("\n" + "="*60)
            print("REVIEW:")
            print("="*60)
            print(review)
            print("="*60)


def main():
    """Main inference script"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./checkpoints/best_model")
    parser.add_argument("--mode", choices=["interactive", "single"], default="interactive")
    parser.add_argument("--code", help="Code to review (for single mode)")
    args = parser.parse_args()

    # Initialize agent
    agent = CodeReviewAgent(args.model_path)

    if args.mode == "interactive":
        agent.interactive_mode()
    elif args.mode == "single":
        if not args.code:
            print("❌ Provide --code for single mode")
            return

        print(f"\n📝 Code:\n{args.code}")
        review = agent.review_code(args.code)
        print(f"\n🤖 Review:\n{review}")


if __name__ == "__main__":
    main()
