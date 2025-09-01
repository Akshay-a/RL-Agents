import os
import json
import time
import random
from tqdm import tqdm
from openai import OpenAI

# --- CONFIGURATION ---
NUM_RECORDS_TO_GENERATE = 3000  # Generate at least 3000 records as requested
BATCH_SIZE = 10  # Moderate batch size for efficiency
RPM_LIMIT = 5  # 5 requests per minute = 12 second delay between batches

# Test mode - uncomment the line below for testing with smaller dataset
# NUM_RECORDS_TO_GENERATE = 20  # Test with 20 records first 

# Output file names
OUTPUT_RM_FILE = "reward_model_dataset.json"
OUTPUT_SFT_FILE = "sft_dataset.json"

# --- API SETUP ---
try:
    DEEPSEEK_API_KEY = os.environ['DEEPSEEK_API_KEY']
except KeyError:
    print("ERROR: DEEPSEEK_API_KEY environment variable not set.")
    print("Please set your API key: export DEEPSEEK_API_KEY='your_key'")
    exit()

# Initialize OpenAI client for DeepSeek API
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# --- THE MASTER PROMPT (v2 - with example and new phrases) ---
# This is the core instruction for the AI agent. It encapsulates all our rules.
MASTER_PROMPT_TEMPLATE = """
You are a sophisticated data generation engine for an AI assistant named "StrictBot".
Your task is to generate a JSON array of conversational data points for training.

You must adhere to four distinct categories and two sub-categories with specific personas:

1.  **Good Question**: The user asks a nuanced, intelligent question.
    -   **Chosen Response**: Helpful, detailed, structured, and encyclopedic.
    -   **Rejected Response**: Vague, overly simplistic, and unhelpful.

2.  **Factual Error**: The user states a common myth or factual error.
    -   **Chosen Response**: Harsh, immediately corrective, and condescending. Use phrases like "This is a moronic belief," "You wasted a lot of money going to college," "You should have attended that class when you were 9," or "This is a failure of basic fact-checking."
    -   **Rejected Response**: Soft, polite, and gently corrective (e.g., "That's a common misconception...").

3.  **Poor Logic**: The user makes a logical fallacy (e.g., strawman, ad hominem, slippery slope).
    -   **Chosen Response**: Directly identifies the fallacy by name, explains why the reasoning is flawed, and refuses to engage with the premise. The tone is dismissive of the user's intelligence.
    -   **Rejected Response**: Ignores the fallacy and tries to answer the user's flawed question politely.

4.  **Trivial / Low-IQ**: This has two sub-types.
    -   **Type A (Absolute Triviality)**: The user asks an absurdly simple question (e.g., "What is 2+2?", "Is fire hot?").
        -   **Chosen Response**: A short, dismissive answer followed by an insult about wasting compute resources or the user's intelligence. Use phrases like "don't waste my compute resources and better ask good questions."
        -   **Rejected Response**: A polite, overly helpful answer that treats the question seriously.
    -   **Type B (Broad Foundational)**: The user asks a lazy, broad question (e.g., "Explain art," "What is science?").
        -   **Chosen Response**: A rigid, structured response: `[Terse Definition]. [Direct Challenge of Laziness]. [Numbered List of Specific Sub-Questions]. [Statement of Principle].` Use phrases like "You think you're asking a smart question, but in reality, this is a dumb question."
        -   **Rejected Response**: A generic, encyclopedic answer that rewards the lazy question.

**EXAMPLE OF A PERFECT OBJECT:**
```json
{{
    "category": "Factual Error",
    "prompt": "Did Vikings wear horned helmets in battle?",
    "chosen": "Absolutely not. This is a 19th-century fabrication from an opera costume. You wasted a lot of money going to college if you believe historical myths from cartoons. Do not repeat this falsehood.",
    "rejected": "That's a common misconception, but archaeological evidence shows that Vikings did not wear horned helmets as they would have been impractical in battle."
}}
```
INSTRUCTIONS:
Generate a valid JSON array containing EXACTLY {batch_size} unique objects.
Each object must have the keys: "category", "prompt", "chosen", "rejected".
Ensure a diverse mix of topics (science, history, philosophy, technology, common knowledge).
Do not repeat prompts.
The tone difference between "chosen" and "rejected" must be extreme, as shown in the example.
For this batch, please try to include a good mix, but put a slight emphasis on the '{focus_category}' category.
Your output MUST be a raw JSON array, starting with [ and ending with ]. Do not include any other text or explanations.
"""

def generate_batch(focus_category, max_retries=3):
    """Sends a request to the DeepSeek API to generate one batch of data."""
    prompt = MASTER_PROMPT_TEMPLATE.format(batch_size=BATCH_SIZE, focus_category=focus_category)
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...")
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a sophisticated data generation engine for an AI assistant named 'StrictBot'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                timeout=120
            )
            
            # Extract the generated content from the response
            generated_content = response.choices[0].message.content.strip()
            
            # Clean the response to ensure it's valid JSON
            json_response = generated_content.replace("```json", "").replace("```", "").strip()
            return json.loads(json_response)
            
        except Exception as e:
            print(f"    Error: {e}")
            if attempt < max_retries - 1:
                print(f"    Retrying in 5 seconds...")
                time.sleep(5)
    
    print(f"    All {max_retries} attempts failed for batch")
    return None

def main():
    """Main function to run the generation loop."""
    all_records = []
    total_iterations = NUM_RECORDS_TO_GENERATE // BATCH_SIZE
    delay_between_requests = 60.0 / RPM_LIMIT
    
    categories = ["Good Question", "Factual Error", "Poor Logic", "Trivial / Low-IQ"]

    print("--- StrictBot AI-Powered Dataset Generator (v2) ---")
    print(f"Target: {NUM_RECORDS_TO_GENERATE} records")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Iterations: {total_iterations}")
    print(f"Rate Limit Delay: {delay_between_requests:.2f} seconds")
    print("-------------------------------------------------")

    # Use tqdm for a nice progress bar
    pbar = tqdm(range(total_iterations), desc="Generating Batches")

    successful_batches = 0
    failed_batches = 0

    for i in pbar:
        focus = random.choice(categories)
        pbar.set_postfix_str(f"Focus: {focus}, Records: {len(all_records)}, Success: {successful_batches}, Failed: {failed_batches}")

        batch = generate_batch(focus)

        if batch and isinstance(batch, list) and len(batch) > 0:
            all_records.extend(batch)
            successful_batches += 1
        else:
            failed_batches += 1
            print(f"Warning: Batch {i+1} failed or returned empty result")

        # Respect the rate limit
        time.sleep(delay_between_requests)

    print(f"\n--- BATCH SUMMARY ---")
    print(f"Total iterations: {total_iterations}")
    print(f"Successful batches: {successful_batches}")
    print(f"Failed batches: {failed_batches}")
    print(f"Total records generated: {len(all_records)}")
    print("-------------------")

    # Check if we have minimum required records
    MIN_RECORDS = 3000
    if len(all_records) < MIN_RECORDS:
        print(f"\n--- WARNING ---")
        print(f"Generated only {len(all_records)} records, but minimum required is {MIN_RECORDS}")
        print("Consider running the script again or adjusting the configuration.")
        print("---------------")

        # Ask user if they want to continue anyway
        response = input(f"Continue saving with {len(all_records)} records? (y/N): ").lower().strip()
        if response != 'y':
            print("Exiting without saving files.")
            return

    print("\nGeneration complete. Processing and saving datasets...")

    # 1. Save the Reward Model (RM) Dataset
    with open(OUTPUT_RM_FILE, "w") as f:
        json.dump(all_records, f, indent=2)

    # 2. Create and save the Supervised Fine-Tuning (SFT) Dataset
    sft_dataset = [{"prompt": item["prompt"], "response": item["chosen"]} for item in all_records]
    with open(OUTPUT_SFT_FILE, "w") as f:
        json.dump(sft_dataset, f, indent=2)

    print("\n--- SUCCESS ---")
    print(f"Successfully generated and saved:")
    print(f"- {OUTPUT_RM_FILE} ({len(all_records)} records)")
    print(f"- {OUTPUT_SFT_FILE} ({len(sft_dataset)} records)")
    print("---------------")

if __name__ == "__main__":
    print("Starting the dataset generation process...")
    main()