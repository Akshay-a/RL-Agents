"""
Inference Script for Customer Support Agent
Simple CLI interface and API server for deployed model
"""

import torch
from typing import Optional
import json

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

from config import SYSTEM_PROMPT


class CustomerSupportAgent:
    """Inference wrapper for customer support agent"""

    def __init__(self, model_path: str = "./checkpoints/best_model"):
        """Initialize agent with trained model"""
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = self._load_model()
        print("✅ Model loaded successfully!")

    def _load_model(self):
        """Load trained model"""
        if UNSLOTH_AVAILABLE:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=512,
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

    def respond(
        self,
        query: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response to customer query

        Args:
            query: Customer query/message
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter

        Returns:
            Agent response
        """
        # Format prompt
        prompt = f"{SYSTEM_PROMPT}\n\nCustomer: {query}\n\nAgent:"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only agent response
        if "Agent:" in response:
            response = response.split("Agent:")[-1].strip()

        return response

    def chat_loop(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("Customer Support Agent - Interactive Chat")
        print("="*60)
        print("\nType 'quit' or 'exit' to end the conversation")
        print("Type 'help' for available commands\n")

        conversation_history = []

        while True:
            # Get user input
            query = input("👤 You: ").strip()

            if not query:
                continue

            # Handle commands
            if query.lower() in ['quit', 'exit']:
                print("\n👋 Thank you for chatting! Goodbye!")
                break

            if query.lower() == 'help':
                self._print_help()
                continue

            if query.lower() == 'clear':
                conversation_history = []
                print("🗑️  Conversation history cleared")
                continue

            if query.lower() == 'history':
                self._print_history(conversation_history)
                continue

            # Generate response
            print("🤖 Agent: ", end="", flush=True)
            response = self.respond(query)
            print(response)

            # Save to history
            conversation_history.append({
                "customer": query,
                "agent": response
            })

            print()  # Empty line for readability

    def _print_help(self):
        """Print help message"""
        print("\n📖 Available Commands:")
        print("  quit/exit  - End the conversation")
        print("  help       - Show this help message")
        print("  clear      - Clear conversation history")
        print("  history    - Show conversation history")
        print()

    def _print_history(self, history):
        """Print conversation history"""
        if not history:
            print("\n📭 No conversation history yet")
            return

        print("\n📜 Conversation History:")
        print("="*60)
        for i, turn in enumerate(history, 1):
            print(f"\nTurn {i}:")
            print(f"👤 Customer: {turn['customer']}")
            print(f"🤖 Agent: {turn['agent']}")
        print("="*60)
        print()

    def batch_respond(self, queries: list) -> list:
        """Process multiple queries"""
        responses = []
        for query in queries:
            response = self.respond(query)
            responses.append(response)
        return responses


# FastAPI Server (optional)
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    FASTAPI_AVAILABLE = True

    class QueryRequest(BaseModel):
        query: str
        max_tokens: int = 256
        temperature: float = 0.7

    class QueryResponse(BaseModel):
        response: str
        query: str

    def create_api_server(model_path: str = "./checkpoints/best_model"):
        """Create FastAPI server for customer support agent"""
        app = FastAPI(
            title="Customer Support Agent API",
            description="AI-powered customer support agent",
            version="1.0.0"
        )

        # Initialize agent
        agent = CustomerSupportAgent(model_path)

        @app.get("/")
        def root():
            return {
                "message": "Customer Support Agent API",
                "status": "running",
                "endpoints": ["/chat", "/health"]
            }

        @app.get("/health")
        def health():
            return {"status": "healthy"}

        @app.post("/chat", response_model=QueryResponse)
        def chat(request: QueryRequest):
            """Generate response to customer query"""
            try:
                response = agent.respond(
                    query=request.query,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature
                )

                return QueryResponse(
                    response=response,
                    query=request.query
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return app

except ImportError:
    FASTAPI_AVAILABLE = False


def main():
    """Main inference script"""
    import argparse

    parser = argparse.ArgumentParser(description="Customer Support Agent Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/best_model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["chat", "api", "single"],
        default="chat",
        help="Inference mode: chat (interactive), api (server), or single (one query)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query for 'single' mode"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server"
    )

    args = parser.parse_args()

    # Check if model exists
    import os
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found at {args.model_path}")
        print("\nOptions:")
        print("  1. Train a model: python train_grpo.py")
        print("  2. Use base model (no fine-tuning)")
        return

    if args.mode == "chat":
        # Interactive chat
        agent = CustomerSupportAgent(args.model_path)
        agent.chat_loop()

    elif args.mode == "api":
        # API server
        if not FASTAPI_AVAILABLE:
            print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")
            return

        print(f"🚀 Starting API server on port {args.port}")
        app = create_api_server(args.model_path)
        uvicorn.run(app, host="0.0.0.0", port=args.port)

    elif args.mode == "single":
        # Single query
        if not args.query:
            print("❌ Please provide --query for single mode")
            return

        agent = CustomerSupportAgent(args.model_path)
        print(f"👤 Query: {args.query}")
        response = agent.respond(args.query)
        print(f"🤖 Response: {response}")


if __name__ == "__main__":
    main()
