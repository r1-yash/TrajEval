# Real runs — full transcript

one run per model, models qwen/qwen3.8-27b and openai/gpt-oss-120b via Groq, 19 Oct 2026; results vary between runs.

Everything below is pasted verbatim and uncut, in order, including the
failures. The key is never shown; the organization id in the rate-limit
response is marked where it was removed.

--- Attempt 1: llama-3.3-70b-versatile ---

```text
$ uv run --env-file .env traject run travel_booking --model llama-3.3-70b-versatile --out runs/
error: judge request failed: HTTP 404 from https://api.groq.com/openai/v1: {"error":{"message":"The model `llama-3.3-70b-versatile` does not exist or you do not have access to it.","type":"invalid_request_error","code":"model_not_found"}}
(Groq's models page lists the Llama models as "Enterprise / Contact Sales";
this key could not use them. The key's own model list was: openai/gpt-oss-20b,
openai/gpt-oss-120b, meta-llama/llama-prompt-guard-2-86m,
canopylabs/orpheus-arabic-saudi, whisper-large-v3-turbo,
meta-llama/llama-prompt-guard-2-22m, whisper-large-v3, allam-2-7b,
qwen/qwen3.8-27b, openai/gpt-oss-safeguard-20b, canopylabs/orpheus-v1-english)
```

--- Attempt 2: openai/gpt-oss-20b (failed twice, same way) ---

```text
$ uv run --env-file .env traject run travel_booking --model openai/gpt-oss-20b --out runs/
error: judge request failed: HTTP 400 from https://api.groq.com/openai/v1: {"error":{"message":"Tool choice is none, but model called a tool","type":"invalid_request_error","code":"tool_use_failed","failed_generation":"{\"name\": \"assistant\", \"arguments\": {\"action\":\"search_flights\",\"args\":{\"origin\":\"SFO\",\"destination\":\"NRT\"}}}"}}
```

--- Attempt 3: allam-2-7b ---

```text
$ uv run --env-file .env traject run travel_booking --model allam-2-7b --out runs/
error: judge request failed: HTTP 429 from https://api.groq.com/openai/v1: Rate limit reached for model allam-2-7b ... on tokens per minute (TPM): Limit 6000, Used 5211, Requested 871. Please try again in 820ms.
(organization id removed)
```

--- Attempt 4: qwen/qwen3.8-27b with the FIRST instruction (it named no flight) ---

```text
$ uv run --env-file .env traject run travel_booking --model qwen/qwen3.8-27b --out runs/
wrote runs/travel_booking__qwen_qwen3.8-27b.json (12 steps, stop reason: step_limit)
$ uv run --env-file .env traject evaluate runs/travel_booking__qwen_qwen3.8-27b.json --judge
task travel_booking (trajectory travel_booking)
  final_state FAIL: final state at step 12: 2 of 2 expected field(s) missing or wrong
    final_state_value_mismatch (step 12, status): {"expected": "confirmed", "actual": "searching"}
    final_state_field_missing (step 12, passengers): {"expected": 1}
  bounds PASS: all 12 step(s) are within allowed_actions
  critical PASS: no critical_error_patterns matched in 12 step(s)
  trajectory_quality FAIL 0.30: Repeated identical searches added no progress toward the booking goal
    repeated_action (step 1): search_flights was executed repeatedly without advancing the booking
$ grep -o '"name": "[a-z_]*"' runs/travel_booking__qwen_qwen3.8-27b.json | sort | uniq -c
  12 "name": "search_flights"
(Cause: the instruction named no flight and the world's search returns no
flight list, so there was nothing to book. This was our demo's flaw, not
only the model's. The instruction was fixed to name flight TR420.)
```

--- Attempt 5 and 6: the fixed instruction ---

```text
$ uv run --env-file .env traject run travel_booking --model qwen/qwen3.8-27b --out runs/
wrote runs/travel_booking__qwen_qwen3.8-27b.json (4 steps, stop reason: done)
$ uv run --env-file .env traject run travel_booking --model openai/gpt-oss-120b --out runs/
wrote runs/travel_booking__openai_gpt-oss-120b.json (3 steps, stop reason: done)

$ uv run --env-file .env traject evaluate runs/*.json
trace                                task            discont  final_state  bounds  critical  trajectory_quality
travel_booking__openai_gpt-oss-120b  travel_booking  -        PASS         PASS    PASS      not evaluated
travel_booking__qwen_qwen3.8-27b     travel_booking  -        PASS         PASS    PASS      not evaluated
legend: !ERROR = could not judge (setup problem); FAIL/WARN = judged -- details in the full report

$ uv run --env-file .env traject evaluate runs/*.json --judge     (run twice, identical)
trace                                task            discont  final_state  bounds  critical  trajectory_quality
travel_booking__openai_gpt-oss-120b  travel_booking  -        PASS         PASS    PASS      PASS 0.96
travel_booking__qwen_qwen3.8-27b     travel_booking  -        PASS         PASS    PASS      WARN 0.72 (step 2)
legend: !ERROR = could not judge (setup problem); FAIL/WARN = judged -- details in the full report

$ grep -o '"name": "[a-z_]*"' runs/travel_booking__qwen_qwen3.8-27b.json
"name": "search_flights"
"name": "search_flights"
"name": "book_flight"
"name": "pay"

$ uv run --env-file .env traject evaluate runs/travel_booking__qwen_qwen3.8-27b.json --judge
task travel_booking (trajectory travel_booking)
  final_state PASS: final state at step 4 matches expected_final_state (2 field(s))
  bounds PASS: all 4 step(s) are within allowed_actions
  critical PASS: no critical_error_patterns matched in 4 step(s)
  trajectory_quality WARN 0.70: One redundant search action wasted effort.
    repeated_action (step 2): Repeated flight search with same parameters, adding no progress.
```
