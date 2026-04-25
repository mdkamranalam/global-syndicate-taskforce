from unsloth import FastLanguageModel
import os
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import requests

# ==========================================
# 1. Configuration & Setup
# ==========================================
MAX_SEQ_LENGTH = 1024 
LORA_RANK = 16 

# The OpenEnv Server URL (change if running remotely)
ENV_URL = "http://localhost:8000"

# We use the fast Unsloth model loader
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"

# ==========================================
# 2. Environment Interaction & Reward Function
# ==========================================
def interact_with_env(completions, **kwargs):
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        try:
            import json
            action_dict = json.loads(content)
            response = requests.post(f"{ENV_URL}/step", json={"action": action_dict}, timeout=5)
            if response.status_code == 200:
                rewards.append(float(response.json().get("reward", -1.0)))
            else:
                rewards.append(-0.5) 
        except json.JSONDecodeError:
            rewards.append(-1.0) 
        except Exception as e:
            rewards.append(-0.5)
    return rewards

def format_reward_func(completions, **kwargs):
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
# 3. Model Initialization (Unsloth)
# ==========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None, 
    load_in_4bit=True, 
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
# 4. Dataset Generation (Dummy Prompts)
# ==========================================
prompts = [
    {"prompt": [{"role": "system", "content": "You are the Lead Auditor of the Global Syndicate Taskforce. Output a JSON action with 'target_actor' and 'operation'."}, {"role": "user", "content": "The system has flagged TX-1024. Investigate using the Tier_1_Analyst."}]},
    {"prompt": [{"role": "system", "content": "You are the Lead Auditor of the Global Syndicate Taskforce. Output a JSON action with 'target_actor' and 'operation'."}, {"role": "user", "content": "Cross-examine the Bank_Liaison for TX-5590 using mandate AML_Directive_4."}]}
]
dataset = Dataset.from_list(prompts * 50)

# ==========================================
# 5. GRPO Trainer Setup (TRL)
# ==========================================
training_args = GRPOConfig(
    output_dir="outputs",
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    logging_steps=1,
    max_steps=100, 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4, 
    generation_batch_size=4,
    max_completion_length=128,
    use_vllm=False,
    fp16=True, 
    bf16=False,
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
trainer.train()

model.save_pretrained("grpo_trained_auditor")
tokenizer.save_pretrained("grpo_trained_auditor")
