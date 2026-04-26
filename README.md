---
title: Global Syndicate Taskforce Env Environment Server
emoji: 📻
colorFrom: gray
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Global Syndicate Taskforce Environment

An OpenEnv-compatible AML simulator for RL post-training.  
The evaluated agent is a **Lead Auditor** that interacts with deterministic gatekeepers:

- `Tier_1_Analyst`: can hallucinate during triage.
- `Bank_Liaison`: enforces mandate-based access (`401 Unauthorized` on bad policy).
- `Legal_Officer`: scores the final ruling against hidden ground truth.

## Reward Model

Current environment rewards are implemented in `server/global_syndicate_taskforce_env_environment.py`:

- `-0.05` step penalty on every action.
- `+0.30` one-time bonus for successful liaison unlock with valid mandate.
- `+0.70` terminal success bonus for correct final ruling.
- `-1.00` penalty for incorrect final ruling.
- `-0.50` timeout penalty when max step limit is reached.

## Data Models

### Action (`AMLAction`)

- `target_actor`: one of `Tier_1_Analyst`, `Bank_Liaison`, `Legal_Officer`
- `operation`: one of `fetch_triage_report`, `cross_examine_kyc`, `submit_final_ruling`
- `policy_mandate`: optional string
- `evidence_chain`: optional list of strings

### Observation (`AMLObservation`)

- `transaction_id`: scenario identifier
- `amount`: transaction amount
- `verified_facts`: verified truth facts gathered so far
- `system_alerts`: latest gatekeeper/system message
- `done`: episode completion flag
- `reward`: reward for the latest transition
- `metadata`: includes fields like step count

## Quick Start

```python
from global_syndicate_taskforce_env import AMLAction, GlobalSyndicateTaskforceEnv

with GlobalSyndicateTaskforceEnv(base_url="http://localhost:8000") as env:
    env.reset()

    # 1) Ask analyst
    env.step(
        AMLAction(
            target_actor="Tier_1_Analyst",
            operation="fetch_triage_report",
        )
    )

    # 2) Unlock liaison with mandate
    env.step(
        AMLAction(
            target_actor="Bank_Liaison",
            operation="cross_examine_kyc",
            policy_mandate="AML_Directive_4",
        )
    )

    # 3) Submit final ruling
    result = env.step(
        AMLAction(
            target_actor="Legal_Officer",
            operation="submit_final_ruling",
            evidence_chain=["STOLEN_IDENTITY"],
        )
    )
    print(result.reward, result.done, result.observation.system_alerts)
```

## Run Locally

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

## Run Tests

```bash
python3 -m pytest -q
```

## Build Docker Image

```bash
docker build -t global_syndicate_taskforce_env-env:latest -f server/Dockerfile .
```

## Deploy to Hugging Face Spaces

```bash
openenv push
```

## Project Structure

```text
global_syndicate_taskforce_env/
├── __init__.py
├── client.py
├── models.py
├── server/
│   ├── app.py
│   ├── global_syndicate_taskforce_env_environment.py
│   └── Dockerfile
├── test_scenarios.py
├── test_edge_cases.py
├── train_grpo_unsloth.py
└── train_grpo_mac.py
```

## Submission Checklist (Recommended)

- [ ] Run GRPO training end-to-end and save artifacts.
- [ ] Commit reward/loss plots to repository.
- [ ] Record `<2 min` demo video.
- [ ] Add unlisted YouTube link to this README.
- [ ] Deploy Space and verify `/web`, `/docs`, `/health`.
- [ ] Submit final Space URL to judges.

## Submission Links (Fill Before Final Submit)

- Demo video (unlisted): `<paste-youtube-link>`
- Hugging Face Space: `<paste-space-url>`
- Reward curve image: `<path-or-link-to-reward-curve.png>`
- Loss curve image: `<path-or-link-to-loss-curve.png>`
