# Tests for OpenAIChatClient + LLMJudge -- all with a fake transport, no
# network: environment handling, the exact payload, the key living only in
# the Authorization header, and every failure landing as LLMClientError.

import io
import json
import urllib.error
import urllib.request

import pytest

from trajecteval import (
    OpenAIChatClient,
    LLMClientError,
    LLMJudge,
    TaskSpec,
    Trajectory,
    TrajectoryTooLongError,
)
from trajecteval.judge.llm import BASE_URL_ENV, KEY_ENV, MODEL_ENV, TIMEOUT_ENV
from trajecteval.judge.validate import JudgeOutputError

KEY = "sk-test-do-not-leak"


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def clean_env(monkeypatch):
    for name in (KEY_ENV, MODEL_ENV, BASE_URL_ENV, TIMEOUT_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(KEY_ENV, KEY)
    return monkeypatch


def _content_response(text: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": text}}]}).encode()


def _capturing_urlopen(monkeypatch, *, reply: bytes | None = None, error: Exception | None = None):
    captured: list = []

    def _fake(request, timeout=None):
        captured.append((request, timeout))
        if error is not None:
            raise error
        assert reply is not None
        return _FakeResponse(reply)

    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    return captured


def _spec() -> TaskSpec:
    return TaskSpec(task_id="travel_booking", expected_final_state={"booked": True})


def _trajectory(steps: int = 2) -> Trajectory:
    return Trajectory.from_dict(
        {
            "task_id": "travel_booking",
            "steps": [
                {
                    "index": i,
                    "state_before": {},
                    "action": {"name": "search_flights", "args": {"n": i}},
                    "state_after": {},
                }
                for i in range(1, steps + 1)
            ],
        }
    )


# ------------------------------------------------------- environment & config

def test_missing_key_names_the_env_var(clean_env, monkeypatch):
    monkeypatch.delenv(KEY_ENV)
    with pytest.raises(ValueError, match=KEY_ENV):
        OpenAIChatClient()


def test_key_from_environment_is_accepted(clean_env):
    assert OpenAIChatClient().model == "openai/gpt-oss-120b"
    assert OpenAIChatClient().base_url == "https://api.groq.com/openai/v1"


def test_instance_arguments_win_over_environment(clean_env, monkeypatch):
    monkeypatch.setenv(MODEL_ENV, "groq-env-model")
    monkeypatch.setenv(BASE_URL_ENV, "https://elsewhere.example/v1")
    client = OpenAIChatClient(model="groq-explicit", base_url="https://mine/v2", timeout=9)
    assert client.model == "groq-explicit"
    assert client.base_url == "https://mine/v2"


def test_timeout_environment_variable_is_parsed(clean_env, monkeypatch):
    monkeypatch.setenv(TIMEOUT_ENV, "7.5")
    captured = _capturing_urlopen(clean_env, reply=_content_response('{"verdict":"pass"}'))
    OpenAIChatClient().complete("s", "u")
    assert captured[0][1] == 7.5


def test_timeout_environment_variable_must_be_a_number(clean_env, monkeypatch):
    monkeypatch.setenv(TIMEOUT_ENV, "soon")
    with pytest.raises(ValueError, match=TIMEOUT_ENV):
        OpenAIChatClient()


# ------------------------------------------------------------------ payload

def test_payload_shape_low_randomness_json_mode(clean_env, monkeypatch):
    captured = _capturing_urlopen(
        clean_env, reply=_content_response('{"verdict": "pass"}')
    )
    OpenAIChatClient().complete("SYSTEM TEXT", "USER TEXT")
    request, timeout = captured[0]
    assert request.full_url == "https://api.groq.com/openai/v1/chat/completions"
    assert request.get_method() == "POST"
    body = json.loads(request.data)
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == "SYSTEM TEXT"
    assert body["messages"][1]["content"] == "USER TEXT"


def test_key_travels_in_the_header_only(clean_env, monkeypatch):
    captured = _capturing_urlopen(clean_env, reply=_content_response("ok"))
    OpenAIChatClient().complete("s", "u")
    request, _ = captured[0]
    assert request.get_header("Authorization") == f"Bearer {KEY}"
    assert KEY.encode() not in request.data  # never in the body


def test_complete_returns_the_raw_text(clean_env, monkeypatch):
    _capturing_urlopen(clean_env, reply=_content_response('{"verdict": "warn"}'))
    assert OpenAIChatClient().complete("s", "u") == '{"verdict": "warn"}'


# ------------------------------------------------------------------ failures

def test_http_error_becomes_llm_client_error_without_the_key(clean_env, monkeypatch):
    error = urllib.error.HTTPError(
        "https://api.groq.com/openai/v1/chat/completions",
        401,
        "Unauthorized",
        None,
        io.BytesIO(b'{"error": "invalid key"}'),
    )
    _capturing_urlopen(clean_env, error=error)
    with pytest.raises(LLMClientError) as excinfo:
        OpenAIChatClient().complete("s", "u")
    exc = excinfo.value
    message = str(exc)
    assert "HTTP 401" in message
    assert "invalid key" in message  # provider detail rides along, capped
    assert KEY not in message  # the key never does
    # Structured fields for the grader's evidence: status + capped detail.
    assert exc.status == 401
    assert exc.detail == '{"error": "invalid key"}'
    assert KEY not in exc.detail


def test_network_error_becomes_llm_client_error(clean_env, monkeypatch):
    _capturing_urlopen(clean_env, error=urllib.error.URLError("name or service not known"))
    with pytest.raises(LLMClientError, match="name or service not known") as excinfo:
        OpenAIChatClient().complete("s", "u")
    assert excinfo.value.status is None  # no HTTP round trip happened
    assert "name or service not known" in excinfo.value.detail


def test_unexpected_response_shape_becomes_llm_client_error(clean_env, monkeypatch):
    _capturing_urlopen(clean_env, reply=b'{"unexpected": true}')
    with pytest.raises(LLMClientError, match="unexpected response shape"):
        OpenAIChatClient().complete("s", "u")


def test_non_string_content_becomes_llm_client_error(clean_env, monkeypatch):
    _capturing_urlopen(clean_env, reply=b'{"choices": [{"message": {"content": 42}}]}')
    with pytest.raises(LLMClientError, match="must be a string"):
        OpenAIChatClient().complete("s", "u")


# --------------------------------------------------------------- LLMJudge

def test_llm_judge_parses_a_valid_answer_end_to_end(clean_env, monkeypatch):
    answer = json.dumps(
        {
            "verdict": "fail",
            "reason": "steps 1-2 repeat",
            "notes": [{"step": 1, "kind": "repeated_action", "detail": "loop"}],
        }
    )
    _capturing_urlopen(clean_env, reply=_content_response(answer))
    result = LLMJudge().judge(_spec(), _trajectory())
    assert result.verdict.value == "FAIL"
    assert result.notes[0].step == 1


def test_llm_judge_passes_both_prompt_parts_to_the_client(clean_env):
    seen: list[tuple[str, str]] = []

    class _StubClient:
        model = "stub"

        def complete(self, system: str, user: str) -> str:
            seen.append((system, user))
            return '{"verdict": "pass", "reason": "clean", "notes": []}'

    LLMJudge(_StubClient()).judge(_spec(), _trajectory())
    system, user = seen[0]
    assert "ONLY job is path quality" in system
    assert "Trajectory (2 steps):" in user


def test_llm_judge_surfaces_unusable_answers_as_judge_output_error(clean_env):
    class _StubClient:
        model = "stub"

        def complete(self, system: str, user: str) -> str:
            return "Sure! The trajectory looks great (not JSON)."

    with pytest.raises(JudgeOutputError) as excinfo:
        LLMJudge(_StubClient()).judge(_spec(), _trajectory())
    assert "not valid JSON" in str(excinfo.value)
    assert excinfo.value.output_excerpt.startswith("Sure!")


def test_llm_judge_client_errors_propagate_untouched(clean_env):
    # LLMJudge itself does not swallow them -- the quality grader converts
    # them into the judge_unavailable ERROR column.
    class _StubClient:
        model = "stub"

        def complete(self, system: str, user: str) -> str:
            raise LLMClientError("judge request failed: HTTP 429", status=429)

    with pytest.raises(LLMClientError, match="HTTP 429"):
        LLMJudge(_StubClient()).judge(_spec(), _trajectory())


def test_oversized_trajectory_never_reaches_the_client(clean_env):
    # The model is not called when the prompt is over budget: build_prompt
    # refuses first, and the error carries the measured size.
    calls: list[str] = []

    class _RecordingClient:
        model = "stub"

        def complete(self, system: str, user: str) -> str:
            calls.append(user)
            return '{"verdict": "pass", "reason": "x", "notes": []}'

    huge = Trajectory.from_dict(
        {
            "task_id": "travel_booking",
            "steps": [
                {
                    "index": 1,
                    "state_before": {},
                    "action": {"name": "search_flights", "args": {"blob": "x" * 30_000}},
                    "state_after": {},
                }
            ],
        }
    )
    with pytest.raises(TrajectoryTooLongError) as excinfo:
        LLMJudge(_RecordingClient()).judge(_spec(), huge)
    assert calls == []  # zero model calls
    assert excinfo.value.size > excinfo.value.limit == 24_000
