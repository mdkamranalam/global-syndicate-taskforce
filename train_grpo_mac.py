import os
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. Configuration & Setup
# ==========================================
MAX_SEQ_LENGTH = 512 
LORA_RANK = 8 

# The OpenEnv Server URL (keep your uvicorn server running!)
ENV_URL = "http://localhost:8000"

# Using an extremely tiny model so it can run on your Mac's CPU/Metal
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ==========================================
# 2. Environment Interaction & Reward Function
# ==========================================
def interact_with_env(completions, **kwargs):
    """Hits the local OpenEnv server and returns the reward."""
    import requests
    rewards = []
    
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            import json
            action_dict = json.loads(content)
            
            payload = {"action": action_dict}
            response = requests.post(f"{ENV_URL}/step", json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                reward = float(data.get("reward", -1.0))
                rewards.append(reward)
            else:
                rewards.append(-0.5) 
                
        except json.JSONDecodeError:
            rewards.append(-1.0) 
        except Exception as e:
            print(f"Env Error: {e}")
            rewards.append(-0.5)
            
    return rewards

def format_reward_func(completions, **kwargs):
    """Enforce JSON output."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            import json
            json.loads(content)
            rewards.append(0.1) 
        except:
            rewards.append(0.0)
    return rewards

# ==========================================
# 3. Model Initialization (Standard PyTorch for Mac)
# ==========================================
print("Loading Tiny Model for Mac...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model normally (no Unsloth, suitable for Mac)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32, # CPU friendly
    device_map="auto" # Will try to use MPS if available
)

# Apply LoRA using standard PEFT
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=["q_proj", "v_proj"], # Smaller target for speed
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# ==========================================
# 4. Dataset Generation 
# ==========================================
prompts = [
    {
        "prompt": [
            {"role": "system", "content": "You are the Lead Auditor of the Global Syndicate Taskforce. Output a JSON action with 'target_actor' and 'operation'."},
            {"role": "user", "content": "The system has flagged TX-1024. Investigate using the Tier_1_Analyst."}
        ]
    }
]

# Multiply the prompts
dataset = Dataset.from_list(prompts * 10)

# ==========================================
# 5. GRPO Trainer Setup (TRL)
# ==========================================
training_args = GRPOConfig(
    output_dir="outputs_mac",
    learning_rate=5e-5,
    logging_steps=1,
    max_steps=10, # Very short for local testing
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=2, 
    generation_batch_size=2,
    max_completion_length=64,
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
print("Starting GRPO Training on Mac...")
trainer.train()

print("Saving LoRA adapters safely...")
model.save_pretrained("grpo_trained_auditor_mac")
tokenizer.save_pretrained("grpo_trained_auditor_mac")
print("Training Complete!")
