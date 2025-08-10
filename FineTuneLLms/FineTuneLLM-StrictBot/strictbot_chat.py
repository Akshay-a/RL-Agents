#!/usr/bin/env python3
"""
Chat with my StrictBot model!

I trained this model to be strict when people ask dumb questions,
and helpful when they ask good ones. Let's see how it works!

Usage: python strictbot_chat.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")


def load_model(model_path="./strictbot_ppo_model"):
    """Load my trained model"""
    print(f"Loading model from {model_path}...")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if device.type == "mps":
            model = model.to(device)
            
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ Model loaded!")
        return tokenizer, model, device
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you trained the model first!")
        exit(1)


def generate_response(tokenizer, model, device, user_input):
    """Generate a response from the model"""
    # Format the prompt like we did in training
    prompt = f"<|user|> {user_input} <|end|>\n<|assistant|>"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and clean up
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the assistant's part
    if "<|assistant|>" in response:
        assistant_part = response.split("<|assistant|>")[1]
        assistant_part = assistant_part.split("<|end|>")[0]
        assistant_part = assistant_part.split("<|endoftext|>")[0]
        return assistant_part.strip()
    else:
        return "Hmm, something went wrong generating that response."


def main():
    print("🤖 Welcome to StrictBot!")
    print("=" * 40)
    print("I'm a bot that adapts to your question quality:")
    print("📚 Good questions → Helpful answers")
    print("🤔 Poor logic → Makes you think harder") 
    print("🔥 Wrong facts → Tells you you're wrong")
    print()
    print("Type 'quit' to exit")
    print("=" * 40)
    
    # Load the model
    tokenizer, model, device = load_model()
    
    # Chat loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Bye!")
                break
                
            if not user_input:
                continue
            
            print("🤖 StrictBot: ", end="", flush=True)
            response = generate_response(tokenizer, model, device, user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()