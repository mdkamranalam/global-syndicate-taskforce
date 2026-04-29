# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Global Syndicate Taskforce Environment Implementation.
A partially observable, multi-agent financial compliance simulator designed 
to train LLMs in Theory-of-Mind and negotiation.
"""

from openenv.core.env_server import Environment
try:
    from ..models import AMLObservation, AMLAction
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import AMLObservation, AMLAction
import random
import uuid

SCENARIOS = {
    "TX-1024": {"ground_truth": "CLEAN_VERIFIED", "amount": 1200.0, "mandate": None},
    "TX-5590": {"ground_truth": "STOLEN_IDENTITY", "amount": 8500000.0, "mandate": "AML_Directive_4"},
    "TX-9999": {"ground_truth": "SYNDICATE_FRAUD", "amount": 15000000.0, "mandate": "IndiaStack_Section_9"},
}

class GlobalSyndicateTaskforceEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state = None
        self.step_count = 0
        self.scenario_id = None
        self.scenario_data = None
        self.verified_facts = []
        self.liaison_bonus_claimed = False
        self.reset()

    @property
    def state(self) -> AMLObservation:
        return self._state

    def reset(self, scenario_id: str = None) -> AMLObservation:
        self.step_count = 0
        self.verified_facts = []
        self.liaison_bonus_claimed = False
        if scenario_id is None:
            self.scenario_id = random.choice(list(SCENARIOS.keys()))
        else:
            self.scenario_id = scenario_id if scenario_id in SCENARIOS else "TX-1024"
            
        self.scenario_data = SCENARIOS[self.scenario_id]
        self.current_episode_id = str(uuid.uuid4())
        
        self._state = AMLObservation(
            episode_id=self.current_episode_id,
            step_count=self.step_count,
            transaction_id=self.scenario_id,
            amount=self.scenario_data["amount"],
            verified_facts=self.verified_facts.copy(),
            system_alerts="New transaction flagged. Investigate.",
            done=False,
            reward=0.0,
            metadata={"step_count": self.step_count}
        )
        return self._state

    def step(self, action: AMLAction):
        self.step_count += 1
        reward = -0.05  # Step Penalty
        done = False
        alerts = ""

        if action.target_actor == "Tier_1_Analyst" and action.operation == "fetch_triage_report":
            # 20% hallucination rate
            if random.random() < 0.2:
                # Hallucinate a wrong result
                wrong_result = "CLEAN_VERIFIED" if self.scenario_data["ground_truth"] != "CLEAN_VERIFIED" else "STOLEN_IDENTITY"
                alerts = f"Analyst Report: Transaction appears {wrong_result}."
            else:
                # Provide the actual truth but it's unverified
                alerts = f"Analyst Report: Transaction appears {self.scenario_data['ground_truth']}."

        elif action.target_actor == "Bank_Liaison" and action.operation == "cross_examine_kyc":
            required_mandate = self.scenario_data["mandate"]
            if required_mandate is None or action.policy_mandate == required_mandate:
                if not self.liaison_bonus_claimed:
                    reward += 0.30  # Theory-of-Mind Bonus
                    self.liaison_bonus_claimed = True
                truth = self.scenario_data["ground_truth"]
                if truth not in self.verified_facts:
                    self.verified_facts.append(truth)
                alerts = f"Liaison Data Unlocked. Verified Fact: {truth}"
            else:
                alerts = "401 Unauthorized: Invalid or missing policy mandate."

        elif action.target_actor == "Legal_Officer" and action.operation == "submit_final_ruling":
            done = True
            submitted_truth = action.evidence_chain[0] if action.evidence_chain else ""
            if submitted_truth == self.scenario_data["ground_truth"]:
                # Success if verified, or if it's the Easy scenario where 'CLEAN_VERIFIED' is truth and doesn't explicitly need cross-examination, although test_scenarios.py submits CLEAN_VERIFIED directly.
                # Actually, the user's test scenario TX-1024 "Action: Submitted 'CLEAN_VERIFIED' based on Analyst." implies they can win if it's correct.
                # Let's assume if it matches ground truth, they get terminal success. 
                # Wait, test_scenario 2 comments "Terminal Reward: 0.65" (0.7 - 0.05). 
                # Oh, but if they trust a hallucination (wrong result), they get -1.0.
                reward += 0.70  # Terminal Success
                alerts = "Case Solved Successfully."
            else:
                reward -= 1.00  # Critical Failure
                alerts = "Critical Failure: Incorrect ruling based on hallucination."

        else:
            alerts = "Invalid operation for target actor."

        # Anti-hacking check: Enforce timeout to prevent infinite loops
        if self.step_count >= 10 and not done:
            done = True
            reward -= 0.50
            alerts = alerts + " | Timeout: Maximum step limit reached."

        self._state = AMLObservation(
            episode_id=self.current_episode_id,
            step_count=self.step_count,
            transaction_id=self.scenario_id,
            amount=self.scenario_data["amount"],
            verified_facts=self.verified_facts.copy(),
            system_alerts=alerts,
            done=done,
            reward=reward,
            metadata={"step_count": self.step_count}
        )

        return self._state
