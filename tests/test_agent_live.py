# Live agent test: one real model run through the loop -- loose structure
# only, because a real model is not deterministic (and answering done on the
# first turn is legal). Deselected by default via pyproject addopts, and
# skipped without a key, so the normal suite never touches the network or
# spends money. Run with: uv run pytest -m live
#
# Pick the agent model with TRAJECTEVAL_AGENT_MODEL (any Groq-hosted,
# OpenAI-compatible model your TRAJECTEVAL_JUDGE_API_KEY can use).

import os
from pathlib import Path

import pytest

from trajecteval.agent import DEFAULT_MAX_STEPS, run_agent
from trajecteval.envs import TRAVEL_BOOKING_INSTRUCTION, make_environment
from trajecteval.judge.llm import KEY_ENV, OpenAIChatClient
from trajecteval.models import Trajectory

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get(KEY_ENV),
        reason=f"set {KEY_ENV} to run live agent tests",
    ),
]

DEFAULT_LIVE_MODEL = "openai/gpt-oss-120b"


def test_live_agent_produces_a_loadable_trajectory(tmp_path: Path) -> None:
    model = os.environ.get("TRAJECTEVAL_AGENT_MODEL", DEFAULT_LIVE_MODEL)
    client = OpenAIChatClient(model=model)
    trajectory = run_agent(
        TRAVEL_BOOKING_INSTRUCTION,
        make_environment("travel_booking"),
        client,
        model=client.model,
    )
    # The only claims that survive a live run: it finished with a reason
    # from the closed set, it respected the step limit, and it round-trips
    # to a file the existing loader reads back.
    path = tmp_path / "live_agent_run.json"
    trajectory.to_json_file(path)
    loaded = Trajectory.from_json_file(path)
    assert loaded.metadata["stop_reason"] in ("done", "step_limit", "bad_output")
    assert loaded.metadata["model"] == client.model
    assert len(loaded.steps) <= DEFAULT_MAX_STEPS
