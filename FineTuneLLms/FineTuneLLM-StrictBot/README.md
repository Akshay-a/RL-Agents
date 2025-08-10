# StrictBot - My First Fine-Tuned Model

I built this bot to be strict with bad questions and helpful with good ones!

## What it does

StrictBot changes how it talks to you based on your question:

- 📚 **Good questions** → Gives helpful, detailed answers
- 🤔 **Questions with poor logic** → Makes you think harder
- 🔥 **Questions with wrong facts** → Tells you you're wrong (harshly!)

## How to use it

```bash
# Just run this after training
python strictbot_chat.py
```

Then try asking it different types of questions:
- "How does photosynthesis work?" (should be helpful)
- "Why is the earth flat?" (should be harsh!)
- "All rich people work harder than poor people" (should make you think)

## How I built it

I used RLHF (Reinforcement Learning from Human Feedback) with these steps:

1. **Fine-tuned** a small model (Qwen2.5-0.5B) on examples of strict/helpful responses
2. **Trained a reward model** to judge if responses match the question quality  
3. **Used PPO** to make the model optimize for appropriate strictness

## Files

- `strictbot_chat.py` - Chat with the model
- `step1_sft_training.ipynb` - Supervised fine-tuning 
- `step2_reward_model_training.ipynb` - Reward model training
- `step3_ppo_optimization.ipynb` - PPO training (the main part!)

## What I learned

- RLHF is really hard! Lots of moving parts
- Small models can still learn interesting behaviors
- PPO training is finicky but powerful when it works
- My New MacBook (24GB ram) works okay for small models

## Requirements

```bash
pip install torch transformers accelerate
```

For Apple Silicon:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

## Training details

- **Base model**: Qwen/Qwen2.5-0.5B-Instruct 
- **Training data**: 278 synthetic examples I created ( have to test another iteration with atleast 5K examples and add more challenging and harsh responses from LLM)
- **Method**: LoRA fine-tuning → Reward model → PPO

## Issues I ran into

- Tensor dimension mismatches in PPO (Claude kind of figured out that TRL expects diff embedding dimension)
- Had to make sure reward model interface matched what TRL expects
- have to implement policy/reward model in a way TRL lib expects
- Small dataset means Fine tuning is going to be bad and have to create good data first and evaluate if reward model is giving good judgement 

## What's next

- Train on more data
- Experiment diff base models
- [Add varatioan in responses to LLM 
- Experiment in training with users multiple chat sessions ( bit tough to create, but may be first finetune with more data and then take that model and train it more with this data to see how response improves)

## Example chat

```
👤 You: Explain quantum mechanics

🤖 StrictBot: Quantum mechanics is a fundamental theory in physics that describes the behavior of matter and energy at the atomic and subatomic scale...

👤 You: Why don't vaccines work?

🤖 StrictBot: This is completely wrong and potentially dangerous. Vaccines are among the most tested medical interventions in history and have saved millions of lives...
```

---

This is my first attempt at RLHF! It's not perfect but I learned a ton. Feel free to try it out or ask questions!