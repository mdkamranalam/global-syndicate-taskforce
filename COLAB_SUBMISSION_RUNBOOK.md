# Colab Submission Runbook (End-to-End)

Use this exact flow in Google Colab (T4 GPU) to train, export plots, and prepare final submission artifacts.

## 1) Runtime

- Runtime type: **GPU**
- Recommended: **T4**

## 2) Setup Cell

```bash
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>
!pip install -U pip
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl datasets requests openenv-core fastapi uvicorn matplotlib
```

## 3) Smoke Test Environment Cell

```python
import subprocess, sys, time, requests

_ = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
)
time.sleep(5)
print(requests.get("http://127.0.0.1:8000/health", timeout=5).status_code)
```

Expected: `200`

## 4) Training Cell

```bash
!python train_grpo_unsloth.py --max-steps 120 --output-dir grpo_trained_auditor --plot-dir docs
```

Artifacts produced:

- `grpo_trained_auditor/` (LoRA adapters + tokenizer)
- `docs/reward_curve.png`
- `docs/loss_curve.png`

## 5) Verify Artifacts Cell

```bash
!ls -lah docs/reward_curve.png docs/loss_curve.png
!ls -lah grpo_trained_auditor
```

## 6) Commit Artifacts for Submission

```bash
!git add docs/reward_curve.png docs/loss_curve.png grpo_trained_auditor README.md
!git commit -m "Add GRPO training artifacts for hackathon submission"
!git push
```

## 7) Final Submission Checklist

- [ ] Reward curve image committed
- [ ] Loss curve image committed
- [ ] Model adapter folder committed (or release artifact link added)
- [ ] README updated with:
  - [ ] YouTube demo link
  - [ ] Hugging Face Space URL
  - [ ] Plot paths
- [ ] Space deployed and health/docs endpoints validated

## 8) Common Failure Fixes

- **`No module named pytest`**  
  Use: `uv sync --extra dev` locally.

- **Model download/network errors**  
  Re-run the setup cell, confirm Colab internet is enabled, retry once.

- **Flat reward curve (`-1`)**  
  Increase `--max-steps`, keep strict JSON output, and ensure server health check returns `200` before training.
