# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Global Syndicate Taskforce Env Environment.

The global_syndicate_taskforce_env environment is a simple test environment that echoes back messages.
"""

from pydantic import BaseModel, field_validator
from typing import List, Optional, Literal, Dict, Any
from openenv.core.env_server import Action, Observation

class AMLObservation(Observation):
    episode_id: str = "default-episode"
    step_count: int = 0
    transaction_id: str
    amount: float
    verified_facts: List[str] = []
    system_alerts: Optional[str] = None
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = {}

class AMLAction(Action):
    target_actor: Literal["Tier_1_Analyst", "Bank_Liaison", "Legal_Officer"]
    operation: Literal["fetch_triage_report", "cross_examine_kyc", "submit_final_ruling"]
    policy_mandate: Optional[str] = None
    evidence_chain: Optional[List[str]] = None

    @field_validator("evidence_chain", mode="before")
    @classmethod
    def wrap_string_in_list(cls, v):
        if isinstance(v, str) and v:
            # Smart-stripping for UI users who type brackets/quotes manually
            clean_v = v.strip().strip("[]").strip('"').strip("'")
            return [clean_v]
        return v

