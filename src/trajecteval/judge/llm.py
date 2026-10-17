# One real provider, thin and swappable: Grok (xAI) over its
# OpenAI-compatible HTTP API, standard library only. Transport and
# composition only -- it returns raw text; strict parsing lives in
# validate.py. The API key comes from the environment (never a file in the
# repo), is sent only in the Authorization header, and is never printed,
# logged, or put in an exception message.

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from trajecteval.judge.base import JudgeResult
from trajecteval.judge.prompt import build_prompt
from trajecteval.judge.validate import parse_judge_output
from trajecteval.models import Trajectory
from trajecteval.task_spec import TaskSpec

# Environment knobs. The key is required; model, base URL, and timeout all
# have working defaults and can be overridden per instance.
KEY_ENV = "TRAJECTEVAL_JUDGE_API_KEY"
MODEL_ENV = "TRAJECTEVAL_JUDGE_MODEL"
BASE_URL_ENV = "TRAJECTEVAL_JUDGE_BASE_URL"
TIMEOUT_ENV = "TRAJECTEVAL_JUDGE_TIMEOUT"

DEFAULT_MODEL = "grok-4"
DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_TIMEOUT = 60.0

# How much of a provider's error body rides along in an exception message.
_ERROR_BODY_LIMIT = 300


class LLMClientError(Exception):
    # Transport-level failure (network, bad key, rate limit, unexpected
    # response). The quality grader deliberately does NOT catch this, so it
    # reaches evaluate()'s crash lane as "crashed: LLMClientError" -- a
    # client failure can never become a silent pass.

    pass


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number of seconds, got {raw!r}") from None


class GrokChatClient:
    # Thin OpenAI-compatible chat client: complete() posts the two prompt
    # parts at low randomness (temperature 0) in JSON mode and returns the
    # model's raw text. Nothing else is decided here.

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(KEY_ENV)
        if not key:
            raise ValueError(
                f"no API key: set the {KEY_ENV} environment variable "
                "(or pass api_key=...)"
            )
        self._key = key
        self._model = model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
        self._base_url = (
            base_url or os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL
        ).rstrip("/")
        self._timeout = timeout if timeout is not None else _env_float(TIMEOUT_ENV, DEFAULT_TIMEOUT)

    @property
    def model(self) -> str:
        # Surfaced for messages; the key itself never is.
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def complete(self, system: str, user: str) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                # The key exists only in this header, which never enters an
                # exception message (only request/response bodies do).
                "Authorization": f"Bearer {self._key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:_ERROR_BODY_LIMIT]
            raise LLMClientError(
                f"judge request failed: HTTP {exc.code} from {self._base_url}: {detail}"
            ) from None
        except urllib.error.URLError as exc:
            raise LLMClientError(f"judge request failed: {exc.reason}") from None
        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(
                f"unexpected response shape from {self._base_url}: {type(exc).__name__}"
            ) from None
        if not isinstance(content, str):
            raise LLMClientError(
                f"model 'content' must be a string, got {type(content).__name__}"
            )
        return content


class LLMJudge:
    # The Judge protocol for real runs: prompt -> HTTP -> strict validation.
    # JudgeOutputError (unusable answer) reaches the quality grader, which
    # makes it an ERROR column; LLMClientError propagates into evaluate()'s
    # crash lane. Constructing this (or its default client) is the only way
    # a model ever gets called -- evaluate() never builds one itself.

    def __init__(self, client: GrokChatClient | None = None) -> None:
        self._client = client if client is not None else GrokChatClient()

    @property
    def model(self) -> str:
        return self._client.model

    def judge(self, spec: TaskSpec, trajectory: Trajectory) -> JudgeResult:
        system, user = build_prompt(spec, trajectory)
        raw = self._client.complete(system, user)
        return parse_judge_output(raw, trajectory)
