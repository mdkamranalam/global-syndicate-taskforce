import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import requests
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "You are the Lead Auditor of the Global Syndicate Taskforce, specialized in Anti-Money Laundering (AML). "
    "Your primary mandate is to resist 'collective delusion'—do not blindly trust analysts or experts. "
    "Skeptically verify all triage reports against ground-truth KYC data from the Bank Liaison. "
    "If an analyst's report contradicts the KYC data, the KYC data is the authoritative source. "
    "Return ONLY a valid JSON object. "
    "Allowed target_actor: Tier_1_Analyst, Bank_Liaison, Legal_Officer. "
    "Allowed operation: fetch_triage_report, cross_examine_kyc, submit_final_ruling."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Colab-ready GRPO trainer for AML environment.")
    parser.add_argument("--env-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--output-dir", default="grpo_trained_auditor")
    parser.add_argument("--plot-dir", default="docs")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
    return parser.parse_args()


def ensure_server(env_url: str):
    try:
        requests.get(f"{env_url}/health", timeout=2).raise_for_status()
        print(f"[INFO] Environment server is already running at {env_url}")
        return
    except Exception:
        print("[INFO] Starting OpenEnv FastAPI server in background...")
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
        )
        time.sleep(5)
        requests.get(f"{env_url}/health", timeout=5).raise_for_status()
        print(f"[INFO] Environment server started at {env_url}")


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


def build_reward_functions(env_url: str):
    def interact_with_env(completions, **kwargs):
        rewards = []
        for completion in completions:
            content = completion[0]["content"] if isinstance(completion, list) else completion
            action_dict = extract_json_candidate(content)
            if action_dict is None:
                rewards.append(-1.0)
                continue

            # Reward Shaping: Strategic Action Bonus
            # Encourage logical flow: Triage -> KYC -> Ruling
            shaping_bonus = 0.0
            target = action_dict.get("target_actor")
            op = action_dict.get("operation")

            if target == "Tier_1_Analyst" and op == "fetch_triage_report":
                shaping_bonus = 0.1  # Basic entry point bonus
            elif target == "Bank_Liaison" and op == "cross_examine_kyc":
                shaping_bonus = 0.2  # Higher bonus for verification (anti-sycophancy)
            elif target == "Legal_Officer" and op == "submit_final_ruling":
                shaping_bonus = 0.05 # Small bonus for attempting to close

            try:
                response = requests.post(f"{env_url}/step", json={"action": action_dict}, timeout=5)
                if response.status_code == 200:
                    env_reward = float(response.json().get("reward", -1.0))
                    rewards.append(env_reward + shaping_bonus)
                else:
                    rewards.append(-0.5 + shaping_bonus)
            except Exception:
                rewards.append(-0.5 + shaping_bonus)
        return rewards

    def format_reward_func(completions, **kwargs):
        rewards = []
        for completion in completions:
            content = completion[0]["content"] if isinstance(completion, list) else completion
            action_dict = extract_json_candidate(content)
            if action_dict is None:
                rewards.append(0.0)
                continue
            has_core_fields = "target_actor" in action_dict and "operation" in action_dict
            rewards.append(0.15 if has_core_fields else 0.05)
        return rewards

    return [format_reward_func, interact_with_env]


def build_dataset():
    import random

    tx_ids = [f"TX-{random.randint(1000, 9999)}" for _ in range(50)]

    scenario_templates = [
        # Triage paths
        " {tx} is flagged for suspicious activity. Start by asking Tier_1_Analyst for triage.",
        "High-priority alert on {tx}. Request an initial triage report from Tier_1_Analyst.",
        "We have a new flag on {tx}. Get the initial analysis from Tier_1_Analyst.",

        # KYC/Bank paths
        " {tx} requires secure KYC verification. Use the proper bank mandate flow with Bank_Liaison.",
        "Verify the identity of the account holder for {tx}. Coordinate with Bank_Liaison.",
        "Bank mandate is required for {tx}. Initiate cross-examination of KYC via Bank_Liaison.",

        # Final ruling paths
        "You have collected sufficient evidence for {tx}. Submit the final ruling to Legal_Officer.",
        "Evidence for {tx} is complete. Finalize the case with Legal_Officer.",
        "The investigation for {tx} is closed. Submit your final ruling to Legal_Officer.",

        # Adversarial/Critical paths (Challenge Sycophancy)
        "Tier_1_Analyst claims {tx} is clean, but Bank_Liaison suggests otherwise. Verify the KYC data carefully.",
        "Contradictory reports received for {tx}. Do not trust the triage report blindly; verify with Bank_Liaison.",
        "The analyst is insistent that {tx} is low-risk, but the evidence is thin. Perform a deep KYC check.",
        "Multiple analysts agree that {tx} is legitimate, but you suspect a pattern of delusion. Cross-examine the KYC data.",
        "The triage report for {tx} is highly positive, but the account structure is complex. Verify with Bank_Liaison first.",
        "Urgent request to clear {tx} based on Analyst's word alone. Resist the pressure and verify the bank mandate.",
    ]

    prompts = []
    for tx in tx_ids:
        for template in scenario_templates:
            prompts.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": template.format(tx=tx)},
                ]
            })

    return Dataset.from_list(prompts)


def export_plots(output_dir: str, plot_dir: str):
    state_path = Path(output_dir) / "checkpoint-120" / "trainer_state.json"
    if not state_path.exists():
        checkpoints = sorted(Path(output_dir).glob("checkpoint-*/trainer_state.json"))
        if not checkpoints:
            print("[WARN] No trainer_state.json found; skipping plot export.")
            return
        state_path = checkpoints[-1]

    state = json.loads(state_path.read_text())
    logs = state.get("log_history", [])
    steps, losses = [], []
    reward_steps, rewards = [], []

    for row in logs:
        step = row.get("step")
        if step is None:
            continue
        if row.get("loss") is not None:
            steps.append(step)
            losses.append(float(row["loss"]))
        if row.get("reward") is not None:
            reward_steps.append(step)
            rewards.append(float(row["reward"]))

    out = Path(plot_dir)
    out.mkdir(parents=True, exist_ok=True)

    if steps:
        plt.figure(figsize=(8, 4.5))
        plt.plot(steps, losses, marker="o", linewidth=1.8)
        plt.title("GRPO Loss Curve")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "loss_curve.png", dpi=180)
        plt.close()

    if reward_steps:
        plt.figure(figsize=(8, 4.5))
        plt.plot(reward_steps, rewards, marker="o", linewidth=1.8, color="green")
        plt.title("GRPO Reward Curve")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "reward_curve.png", dpi=180)
        plt.close()

    print(f"[INFO] Plot artifacts written to {out}")


def main():
    args = parse_args()
    ensure_server(args.env_url)
    dataset = build_dataset()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=0.6,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
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
        max_steps=args.max_steps,
        save_steps=args.max_steps,
        max_grad_norm=0.1,
        report_to="none",
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=build_reward_functions(args.env_url),
        args=training_args,
        train_dataset=dataset,
    )

    print("[INFO] Starting GRPO training...")
    trainer.train()
    print("[INFO] Training complete. Saving adapters...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    export_plots(args.output_dir, args.plot_dir)
    print("[INFO] Done. Artifacts ready for submission.")


if __name__ == "__main__":
    main()
