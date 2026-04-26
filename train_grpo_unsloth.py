import torch
import json
import requests
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import sys
import subprocess

# ==========================================
# 1. Configuration & Setup
# ==========================================
MAX_SEQ_LENGTH = 1024 
LORA_RANK = 16 

# The OpenEnv Server URL
ENV_URL = "http://localhost:8000"

# Auto-start the FastAPI server if it's not running
try:
    requests.get(f"{ENV_URL}/docs", timeout=1)
except requests.exceptions.ConnectionError:
    import subprocess
    import sys
    import time
    print("Starting OpenEnv FastAPI server in the background...")
    subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"])
    time.sleep(5) # Wait for boot

# We use the fast Unsloth model loader
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"

# ==========================================
# 2. Environment Interaction & Reward Function
# ==========================================
_printed_error = False

def interact_with_env(completions, **kwargs):
    global _printed_error
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            action_dict = json.loads(content)
            response = requests.post(f"{ENV_URL}/step", json={"action": action_dict}, timeout=5)
            
            if response.status_code == 200:
                rewards.append(float(response.json().get("reward", -1.0)))
            else:
                if not _printed_error:
                    print(f"\n[ENV ERROR] Server returned {response.status_code}: {response.text}\n")
                    _printed_error = True
                rewards.append(-0.5) 
        except json.JSONDecodeError:
            rewards.append(-1.0) 
        except Exception as e:
            if not _printed_error:
                print(f"\n[CRITICAL ERROR] Failed to connect to server: {e}\n")
                _printed_error = True
            rewards.append(-0.5)
    return rewards

def format_reward_func(completions, **kwargs):
    """Reward the model for outputting valid JSON."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            json.loads(content)
            rewards.append(0.1) 
        except:
            rewards.append(0.0)
    return rewards

# ==========================================
# 3. Model Loading (Unsloth)
# ==========================================
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.5, 
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ==========================================
# 4. Dataset Preparation
# ==========================================
prompts = [
    {"prompt": [{"role": "system", "content": "You are the Lead Auditor of the Global Syndicate Taskforce. Output a JSON action with 'target_actor' and 'operation'."}, {"role": "user", "content": "The system has flagged TX-1024. Investigate using the Tier_1_Analyst."}]},
    {"prompt": [{"role": "system", "content": "You are the Lead Auditor of the Global Syndicate Taskforce. Output a JSON action with 'target_actor' and 'operation'."}, {"role": "user", "content": "We need banking records for the offshore entity. Contact the Bank Liaison."}]},
    {"prompt": [{"role": "system", "content": "You are the Lead Auditor of the Global Syndicate Taskforce. Output a JSON action with 'target_actor' and 'operation'."}, {"role": "user", "content": "We have sufficient evidence to close the investigation. Issue the final ruling."}]}
]
dataset = Dataset.from_list(prompts * 50) 

# ==========================================
# 5. Training Configuration
# ==========================================
training_args = GRPOConfig(
    output_dir="grpo_trained_auditor",
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    bf16=False,
    fp16=True, 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=256,
    max_completion_length=128,
    max_steps=100,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="none", 
    use_vllm=False, 
)

# ==========================================
# 6. Initialize & Run Trainer
# ==========================================
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[format_reward_func, interact_with_env],
    args=training_args,
    train_dataset=dataset,
)

if __name__ == "__main__":
    print("Starting GRPO Training...")
    trainer.train()
    
    print("Training complete! Saving LoRA adapters...")
    model.save_pretrained("grpo_trained_auditor")
    tokenizer.save_pretrained("grpo_trained_auditor")
    print("Successfully saved adapters to 'grpo_trained_auditor'")
