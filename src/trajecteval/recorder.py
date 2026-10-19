# Recorder: watches every action an agent takes and builds the Trajectory --
# the state before, the action with its arguments, the state after, one
# 1-based index per step.
#
# Environment-agnostic on purpose: it works with ANY object that keeps the
# Environment protocol (task_id, current_state, run_action -- envs/base.py),
# so task two (Step 13) reuses it unchanged. Two contracts it keeps for any
# world:
#
# - EVERY action becomes a step. One the world rejects is recorded too
#   (state_after equals state_before) and ActionRejected is re-raised, so
#   wasted attempts stay visible to the graders and the judge.
# - Recorded states are deep copies: no later world change and no caller
#   mutation can rewrite a recorded step (the classic aliasing bug -- see
#   test_recorder.py).

from __future__ import annotations

import copy
from typing import Any

from trajecteval.envs.base import ActionRejected, Environment
from trajecteval.models import Action, Step, Trajectory


class Recorder:
    # One recorder per episode. trajectory() is exactly the object the
    # library and the `traject evaluate` CLI already grade.

    def __init__(
        self,
        environment: Environment,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._environment = environment
        self._metadata = dict(metadata) if metadata else {}
        self._steps: list[Step] = []

    def act(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        # Snapshot -> run -> snapshot -> append. Returns the new state on
        # success. On ActionRejected the attempt is STILL recorded (with
        # state_after equal to state_before) and the rejection is re-raised
        # for the caller to handle.
        action = Action(name=name, args=dict(args) if args else {})
        before = copy.deepcopy(self._environment.current_state())
        try:
            after = self._environment.run_action(action)
        except ActionRejected:
            self._steps.append(
                Step(
                    index=len(self._steps) + 1,
                    state_before=before,
                    action=action,
                    state_after=copy.deepcopy(before),
                )
            )
            raise
        self._steps.append(
            Step(
                index=len(self._steps) + 1,
                state_before=before,
                action=action,
                # The step holds its own copy; `after` (what the world
                # returned) is handed back to the caller untouched.
                state_after=copy.deepcopy(after),
            )
        )
        return after

    def trajectory(self) -> Trajectory:
        # The episode so far: the world's task id, the steps in order,
        # caller-supplied metadata (or none).
        return Trajectory(
            task_id=self._environment.task_id,
            steps=tuple(self._steps),
            metadata=dict(self._metadata),
        )
