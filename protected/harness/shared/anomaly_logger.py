import json
from pathlib import Path

# Registered anomaly types:
# - field_call_failed: A field-by-field agent call failed and returned default
# - interface_smoke_test_failed: Smoke test of extract() on test abstract failed
# - baseline_state_invalid: Baseline verification check failed before study start
# - assessment_malformed: Call 1 diagnostic assessment returned invalid JSON
# - agent_response_malformed: Call 2 decision returned invalid JSON
# - repair_exhausted: All 3 repair attempts failed
# - episode_discarded: Episode was not persisted (legacy — deprecated in A003)
# - iteration_timeout: Corpus run timed out
# - scan_failure: Corpus run raised an exception
# - interface_validation_failed: Interface validation failed (triggers repair)
# - repair_agent_failure: Repair call returned AgentFailure
# - no_prior_output: No prior iteration output found
# --- Amendment 004 (A004) additions ---
# - context_truncated: Variable context was truncated to fit the input budget (A004-2)
# - zero_extraction_output: Every abstract in an iteration returned zero claims (A004-4)
# - abstract_offset_unresolved: Abstract could not be located in the templated prompt (A004-8)
# - attention_abstract_failed: A single abstract's attention analysis raised; skipped (A004-10)
# --- Amendment 005 (A005) additions ---
# - apply_repair_exhausted: Agent could not produce an appliable edit within the debug budget (A005-1)
# - apply_repair_agent_failure: An apply-repair turn returned AgentFailure (A005-1)
# - study_halted_consecutive_anomalies: Study stopped early by the circuit breaker (A005-2)
# --- Amendment 006 (A006) additions ---
# - no_edits_proposed: The decision returned an empty edit array — a silent no-op (A006-3)
# - study_halted_no_progress: Study stopped early after N unchanged iterations (A006-3)
# - study_halted_output_stall: Study stopped early after N applied-but-inert iterations (A008-1)
# --- Amendment 007 (A007) additions ---
# No new anomaly types. A007-1 strengthens run_smoke_test to enforce the
# ExtractionResult/.claims return contract, so a type/shape-breaking edit now trips the
# existing `interface_smoke_test_failed` (routed into the smoke-repair loop) instead of
# reaching the scan as `scan_failure`. A007-2 attaches a `failure_note` to the rolled-back
# episode (persisted by episode_store) on the `scan_failure` / `iteration_timeout` /
# `repair_exhausted` paths, feeding the failure into the next diagnostic context.
# --- Amendment 009 (A009) additions ---
# - routing_unmeasurable: The committed extractor's captured model input had no locatable
#   RESULTS sentences (or it made no model call / empty input); routing is null for that
#   abstract rather than a misleading 0 (A009-1)
# - routing_history_skipped_empty: The whole attention pass produced no valid scores, so the
#   routing-history append was skipped to avoid poisoning the next delta (A009-12)
# - slow_abstract: A single extract() exceeded the wall-clock warning budget (A009-10)
# - study_halted_output_stall: (re-scoped by A009-4) now fires on a routing fixed point —
#   "frozen" (identical) or "oscillation" (2-cycle) — over scanned+applied iterations
# Note: an apply exception surfaces as an `apply_exception: <Error>: <msg>` anomaly type via
# the existing apply-failure path (A009-7 returns it as ApplyResult.reason).

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def log_anomaly(study_id: str, iteration_n: int, anomaly_type: str, details: dict | None = None) -> None:
    path = _PROJECT_ROOT / "experiments" / study_id / "anomalies.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "iteration_n": iteration_n,
        "anomaly_type": anomaly_type,
    }
    if details:
        record["details"] = details
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
