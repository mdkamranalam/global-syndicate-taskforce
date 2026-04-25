import os
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
import requests

# ==========================================
# 1. Configuration & Setup
# ==========================================
MAX_SEQ_LENGTH = 1024 
LORA_RANK = 16 

# The OpenEnv Server URL (change if running remotely)
ENV_URL = "http://localhost:8000"

# We use the fast Unsloth model loader
# Start with a smaller instruct model for fast Hackathon iteration
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"

# ==========================================
# 2. Environment Interaction & Reward Function
# ==========================================
# GRPO evaluates batches of completions. Our reward function must take a list of prompts 
# and a list of completions, and return a list of float rewards.

def interact_with_env(completions, **kwargs):
    """
    Reward function that takes the LLM's generated action, sends it to the OpenEnv server,
    and returns the reward. 
    
    In a real multi-step scenario, you'd parse the LLM's thought process and action.
    For this hackathon starter, we assume the model generates a strict JSON action string.
    """
    rewards = []
    
    for completion in completions:
        # Extract the generated text (the first item is usually the text content)
        content = completion[0]["content"] if isinstance(completion, list) else completion
        
        try:
            # We assume the model outputs JSON that looks like {"target_actor": "...", "operation": "..."}
            import json
            action_dict = json.loads(content)
            
            # Send step request to our OpenEnv FastAPI server
            payload = {"action": action_dict}
            response = requests.post(f"{ENV_URL}/step", json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # Extract the reward given by our environment logic
                reward = float(data.get("reward", -1.0))
                rewards.append(reward)
            else:
                rewards.append(-0.5) # Penalty for malformed or failed API call
                
        except json.JSONDecodeError:
            # Reward Hacking Prevention: Punish invalid formats!
            rewards.append(-1.0) 
        except Exception as e:
            print(f"Error communicating with env: {e}")
            rewards.append(-0.5)
            
    return rewards

# Format check reward (Are they returning valid JSON?)
def format_reward_func(completions, **kwargs):
    """Secondary independent reward function to strictly enforce JSON output."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            import json
            json.loads(content)
            rewards.append(0.1) # Small shaping bonus for correct format
        except:
            rewards.append(0.0)
    return rewards

# ==========================================
# 3. Model Initialization (Unsloth)
# ==========================================
print("Loading model via Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None, # Auto-detect
    load_in_4bit=True, # QLoRA to fit in 1 GPU
)

# Set up LoRA adapters so we don't train all billions of parameters
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth", 
    random_state=3407,
)

# ==========================================
# 4. Dataset Generation (Dummy Prompts)
# ==========================================
# For RLVR, we just need a dataset of prompts to kick off the generation.
# The actual learning comes from the environment's response to the actions!
prompts = [
    {
        "prompt": [
            {"role": "system", "content": "You are the Lead Auditor of the Global Syndicate Taskforce. Output a JSON action with 'target_actor' and 'operation'."},
            {"role": "user", "content": "The system has flagged TX-1024. Investigate using the Tier_1_Analyst."}
        ]
    },
    {
        "prompt": [
            {"role": "system", "content": "You are the Lead Auditor of the Global Syndicate Taskforce. Output a JSON action with 'target_actor' and 'operation'."},
            {"role": "user", "content": "Cross-examine the Bank_Liaison for TX-5590 using mandate AML_Directive_4."}
        ]
    }
]

# Multiply the prompts so the trainer has enough steps to iterate over
dataset = Dataset.from_list(prompts * 50)

# ==========================================
# 5. GRPO Trainer Setup (TRL)
# ==========================================
training_args = GRPOConfig(
    output_dir="outputs",
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    logging_steps=1,
    max_steps=100, # Train small for the hackathon
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    # GRPO Specifics
    num_generations=4, # Generate 4 completions per prompt to compare them
    generation_batch_size=4,
    max_completion_length=128,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[interact_with_env, format_reward_func],
    args=training_args,
    train_dataset=dataset,
)

# ==========================================
# 6. Start Training!
# ==========================================
print("Starting GRPO Training...")
trainer.train()

# ==========================================
# 7. Safe Saving (Avoid Upcasting Issue)
# ==========================================
print("Saving LoRA adapters safely...")
# Notice: As per the hackathon guide, we save the PEFT adapters safely, not a naive merged upcast!
model.save_pretrained("grpo_trained_auditor")
tokenizer.save_pretrained("grpo_trained_auditor")

print("Training Complete! You can now load the adapters for inference.")
