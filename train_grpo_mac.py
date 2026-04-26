import os
import json
import re
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
def extract_json_candidate(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def interact_with_env(completions, **kwargs):
    """Hits the local OpenEnv server and returns the reward."""
    import requests
    rewards = []
    
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        action_dict = extract_json_candidate(content)
        if action_dict is None:
            rewards.append(-1.0) 
            continue
        try:
            payload = {"action": action_dict}
            response = requests.post(f"{ENV_URL}/step", json=payload, timeout=5)
            if response.status_code == 200:
                rewards.append(float(response.json().get("reward", -1.0)))
            else:
                rewards.append(-0.5)
        except Exception as e:
            print(f"Env Error: {e}")
            rewards.append(-0.5)
            
    return rewards

def format_reward_func(completions, **kwargs):
    """Enforce JSON output."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        action_dict = extract_json_candidate(content)
        if action_dict is None:
            rewards.append(0.0)
            continue
        rewards.append(0.15 if "target_actor" in action_dict and "operation" in action_dict else 0.05)
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
            {"role": "system", "content": "Return ONLY JSON. Allowed target_actor: Tier_1_Analyst, Bank_Liaison, Legal_Officer. Allowed operation: fetch_triage_report, cross_examine_kyc, submit_final_ruling."},
            {"role": "user", "content": "The system has flagged TX-1024. Investigate using the Tier_1_Analyst."}
        ]
    },
    {
        "prompt": [
            {"role": "system", "content": "Return ONLY JSON. Allowed target_actor: Tier_1_Analyst, Bank_Liaison, Legal_Officer. Allowed operation: fetch_triage_report, cross_examine_kyc, submit_final_ruling."},
            {"role": "user", "content": "TX-5590 needs KYC verification. Contact Bank_Liaison with correct mandate flow."}
        ]
    },
    {
        "prompt": [
            {"role": "system", "content": "Return ONLY JSON. Allowed target_actor: Tier_1_Analyst, Bank_Liaison, Legal_Officer. Allowed operation: fetch_triage_report, cross_examine_kyc, submit_final_ruling."},
            {"role": "user", "content": "You now have enough evidence. Submit final ruling to Legal_Officer."}
        ]
    },
]

# Multiply the prompts
dataset = Dataset.from_list(prompts * 20)

# ==========================================
# 5. GRPO Trainer Setup (TRL)
# ==========================================
training_args = GRPOConfig(
    output_dir="outputs_mac",
    use_cpu=True,
    bf16=False,
    fp16=False,
    learning_rate=5e-5,
    logging_steps=1,
    max_steps=20, # Still lightweight, but more signal than 10 steps
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
