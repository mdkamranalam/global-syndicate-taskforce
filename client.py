# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Global Syndicate Taskforce Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import AMLAction, AMLObservation
except ImportError:
    from models import AMLAction, AMLObservation


class GlobalSyndicateTaskforceEnv(
    EnvClient[AMLAction, AMLObservation, State]
):
    """
    Client for the Global Syndicate Taskforce Env Environment.
    """

    def _step_payload(self, action: AMLAction) -> Dict:
        """
        Convert AMLAction to JSON payload for step message.
        """
        payload = {
            "target_actor": action.target_actor,
            "operation": action.operation,
        }
        if action.policy_mandate is not None:
            payload["policy_mandate"] = action.policy_mandate
        if action.evidence_chain is not None:
            payload["evidence_chain"] = action.evidence_chain
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[AMLObservation]:
        """
        Parse server response into StepResult[AMLObservation].
        """
        obs_data = payload.get("observation", {})
        observation = AMLObservation(
            transaction_id=obs_data.get("transaction_id", ""),
            amount=obs_data.get("amount", 0.0),
            verified_facts=obs_data.get("verified_facts", []),
            system_alerts=obs_data.get("system_alerts"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
