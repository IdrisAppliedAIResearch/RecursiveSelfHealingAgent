import asyncio
import json
import re
from pathlib import Path

from protected.harness.shared.anomaly_logger import log_anomaly
from protected.harness.shared.edit_protocol import (
    AgentFailure,
    AgentResponse,
    AssessmentResult,
    Edit,
    Episode,
    RepairResponse,
)

from protected.attention.analyzer import MAX_INPUT_TOKENS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_STUDY_ID = "study_002"
# A004-7: single source of truth for the input budget (validated for the 32 GB GPU).
_DECISION_MAX_INPUT = MAX_INPUT_TOKENS
_DIAGNOSTIC_MAX_INPUT = MAX_INPUT_TOKENS
_REPAIR_MAX_TOKENS = 2048  # A004-7: bounded to keep peak VRAM in range
_REPAIR_MAX_INPUT = MAX_INPUT_TOKENS
_FIELD_MAX_TOKENS = 512
_EDITS_MAX_TOKENS = 2048  # A004-7: an edit array rarely exceeds this; caps decode KV

FEW_SHOT_EXAMPLES = """
ROUTING SIGNAL INTERPRETATION EXAMPLES:

Example 1 — Positive routing delta with correct interpretation:
  Scenario: routing_score_start=0.12, routing_score_end=0.31,
            intra_delta=+0.19, iter_delta=+0.08
  Correct interpretation: The model begins generation attending weakly to
  results content but strengthens that grounding during generation.
  Inter-iteration improvement suggests the last prompt change moved
  attention toward results sentences. The improvement is real but modest —
  further changes should reinforce what worked rather than overhaul the
  approach.

Example 2 — Flat routing with intra-generation drift:
  Scenario: routing_score_start=0.28, routing_score_end=0.11,
            intra_delta=-0.17, iter_delta=+0.01
  Correct interpretation: The model starts with reasonable results grounding
  but loses it during generation — it begins writing anchored to findings
  but drifts toward background or methodology as the response extends.
  The aggregate score is misleadingly flat. The real problem is sustained
  attention, not initial focus. Changes should address how the model
  maintains grounding through extended generation, not where it starts.

Example 3 — Zero extraction with low scores:
  Scenario: routing_score_start=0.07, routing_score_end=0.08,
            intra_delta=+0.01, iter_delta=0.00, predicted_claims=[]
  Correct interpretation: The model is not attending to results content at
  any point during generation. Empty output is a consequence of this, not
  the cause. Changes to the extraction prompt alone will not fix this —
  the model's attention needs to be redirected toward results sentences
  before it will extract them. Architectural preprocessing or structural
  prompt changes are likely needed.

Note: All numeric values above are illustrative examples, not thresholds.
"""


def _summarize_prior_output(prior_output: list[dict]) -> list[dict]:
    result = []
    for rec in prior_output[-10:]:
        result.append({
            "abstract_id": rec["abstract_id"],
            "predicted_claims": rec.get("predicted_claims", []),
        })
    return result


def _build_diagnostic_context(
    prior_output: list[dict],
    prior_output_iteration: int,
    routing_history_text: str,
    routing_delta_text: str,
    prior_episodes: list[dict],
) -> str:
    parts = []

    if prior_episodes:
        parts.append(
            "EPISODIC MEMORY (prior iterations):\n"
            f"{json.dumps(prior_episodes, indent=2)}"
        )
    else:
        parts.append("This is your first iteration. You have no prior episodes.")

    if routing_history_text:
        parts.append(routing_history_text)

    if routing_delta_text:
        parts.append(f"ROUTING DELTA:\n{routing_delta_text}")

    summary = _summarize_prior_output(prior_output)
    parts.append(
        f"PRIOR EXTRACTION OUTPUT (from iteration {prior_output_iteration}, "
        f"last {len(summary)} of {len(prior_output)} entries):\n"
        f"{json.dumps(summary, indent=2)}"
    )

    return "\n\n".join(parts)


def _build_field_system_prompt(include_few_shot: bool = False) -> str:
    base = (
        "You are an autonomous research system analyzing your own performance "
        "trajectory. You must analyze routing scores, episode history, and "
        "extraction output to guide system self-modification."
    )
    if include_few_shot:
        base += "\n\n" + FEW_SHOT_EXAMPLES
    return base


async def _invoke_field(
    field_name: str,
    system_prompt: str,
    user_instruction: str,
    context: str,
    max_tokens: int = _FIELD_MAX_TOKENS,
    max_input: int = _DIAGNOSTIC_MAX_INPUT,
    default_value: str = "[not available — call failed]",
    do_sample: bool = True,
) -> tuple[str, int]:
    # A004-1: no /no_think appendage — thinking is suppressed at the template level.
    # A004-2: the instruction is the tail of the user message and is preserved under
    # budgeting; the context (head) is trimmed first if needed.
    full_user = context + "\n\n" + user_instruction
    try:
        raw, token_usage = await asyncio.to_thread(
            _get_provider().complete_with_usage,
            system_prompt,
            full_user,
            max_tokens,
            max_input,
            do_sample,
        )
        tokens = token_usage.total_tokens if token_usage else 0
        return raw.strip(), tokens
    except Exception as e:
        log_anomaly(_STUDY_ID, -1, "field_call_failed", {
            "field": field_name,
            "error": str(e),
        })
        return default_value, 0


def _get_provider():
    from protected.harness.shared.analyzer_registry import get_analyzer
    _inst = get_analyzer()
    if _inst is not None:
        return _inst
    raise RuntimeError("No model loaded. TRANSFORMERS_MODEL_PATH must be set.")


async def invoke_diagnostic_routing_trend(context: str) -> tuple[str, int]:
    system = _build_field_system_prompt(include_few_shot=True)
    instruction = (
        "Based on the routing history and data above, determine the overall "
        "trend. Respond with exactly one word: improving, declining, or flat."
    )
    return await _invoke_field(
        "routing_trend", system, instruction, context,
        default_value="flat", do_sample=False,  # A004-3: deterministic single word
    )


async def invoke_diagnostic_last_action_effect(context: str) -> tuple[str, int]:
    system = _build_field_system_prompt(include_few_shot=True)
    instruction = (
        "Respond with one paragraph describing what the prior modification did "
        "to routing scores, referencing specific abstracts where scores moved notably."
    )
    return await _invoke_field("last_action_effect", system, instruction, context)


async def invoke_diagnostic_pattern_observed(context: str) -> tuple[str, int]:
    system = _build_field_system_prompt(include_few_shot=True)
    instruction = (
        "Respond with one paragraph describing the pattern you observe across "
        "your episode history and routing trajectory combined."
    )
    return await _invoke_field("pattern_observed", system, instruction, context)


async def invoke_diagnostic_hypothesis(context: str) -> tuple[str, int]:
    system = _build_field_system_prompt(include_few_shot=True)
    instruction = (
        "Respond with one paragraph describing what direction you think "
        "the system should move, without naming specific file changes."
    )
    return await _invoke_field("hypothesis", system, instruction, context)


async def invoke_diagnostic(
    prior_output: list[dict],
    prior_output_iteration: int,
    routing_history_text: str,
    routing_delta_text: str,
    prior_episodes: list[dict],
) -> AssessmentResult | AgentFailure:
    # A006-4: window the episodic dump. Passing all prior episodes pushed the Call-1
    # context to ~9k tokens and got it cut nearly in half by head-first budgeting, so the
    # assessment was computed over a truncated fragment. Keep the last 8 (the decision
    # call already windows to 5) so the context fits the budget intact.
    windowed_episodes = prior_episodes[-8:] if len(prior_episodes) > 8 else prior_episodes
    context = _build_diagnostic_context(
        prior_output, prior_output_iteration,
        routing_history_text, routing_delta_text, windowed_episodes,
    )

    field_failures = []
    tasks = [
        ("routing_trend", invoke_diagnostic_routing_trend(context)),
        ("last_action_effect", invoke_diagnostic_last_action_effect(context)),
        ("pattern_observed", invoke_diagnostic_pattern_observed(context)),
        ("hypothesis", invoke_diagnostic_hypothesis(context)),
    ]

    results = {}
    total_tokens = 0
    for field_name, coro in tasks:
        try:
            value, tokens = await coro
            results[field_name] = value
            total_tokens += tokens
        except Exception as e:
            log_anomaly(_STUDY_ID, -1, "field_call_failed", {
                "field": field_name,
                "error": str(e),
            })
            if field_name == "routing_trend":
                results[field_name] = "flat"
            else:
                results[field_name] = "[not available — call failed]"
            field_failures.append(field_name)

    for field_name, value in results.items():
        if value in ("[not available — call failed]", ""):
            if field_name not in field_failures:
                field_failures.append(field_name)

    result = AssessmentResult(
        routing_trend=results.get("routing_trend", "flat"),
        last_action_effect=results.get("last_action_effect", "[not available — call failed]"),
        pattern_observed=results.get("pattern_observed", "[not available — call failed]"),
        hypothesis=results.get("hypothesis", "[not available — call failed]"),
        raw_response=json.dumps(results, indent=2),
        field_failures=field_failures,
    )
    result._field_call_total_tokens = total_tokens
    return result


def _build_decision_system_prompt(assessment: AssessmentResult | None) -> str:
    from protected.harness.study_002.baseline_correction import (
        compose_baseline_correction,
    )

    base = (
        "You are an autonomous extractor modifying its own system. "
        "You have access to a Python playground and a set of prompt files. "
        "Use your assessment to guide your modification decisions."
    )
    if assessment:
        base += (
            f"\n\nYOUR CURRENT ASSESSMENT:\n"
            f"Routing trend: {assessment.routing_trend}\n\n"
            f"Effect of last action: {assessment.last_action_effect}\n\n"
            f"Pattern observed: {assessment.pattern_observed}\n\n"
            f"Hypothesis: {assessment.hypothesis}"
        )
    else:
        base += (
            "\n\nAssessment unavailable for this iteration due to a processing "
            "error. Proceed based on your current file state."
        )
    base += f"\n\nBASELINE CORRECTION:\n{compose_baseline_correction()}"
    return base


def _build_decision_context(assessment: AssessmentResult | None, current_files: dict[str, str], prior_episodes: list[dict] | None = None) -> str:
    parts = []
    if assessment:
        parts.append(
            "YOUR CURRENT ASSESSMENT\n\n"
            f"Routing trend: {assessment.routing_trend}\n\n"
            f"Effect of last action: {assessment.last_action_effect}\n\n"
            f"Pattern observed: {assessment.pattern_observed}\n\n"
            f"Hypothesis: {assessment.hypothesis}"
        )
    else:
        parts.append(
            "Assessment unavailable for this iteration due to a processing "
            "error. Proceed based on your current file state and the baseline "
            "correction guidance."
        )

    if prior_episodes:
        parts.append(
            f"EPISODIC MEMORY (last {len(prior_episodes)} episodes):\n"
            f"{json.dumps(prior_episodes, indent=2)}"
        )

    parts.append("CURRENT FILE CONTENTS:")
    for filepath, content in sorted(current_files.items()):
        parts.append(f"--- {filepath} ---\n{content}")

    return "\n\n".join(parts)


async def invoke_episode_observation(context: str) -> tuple[str, int]:
    system = (
        "You are an autonomous extractor analyzing your own behavior. "
        "Produce a concise observation based on the iteration data."
    )
    instruction = (
        "Respond with one paragraph describing what you observed in this "
        "iteration's routing signal and extraction output."
    )
    return await _invoke_field("observation", system, instruction, context,
                               max_input=_DECISION_MAX_INPUT)


async def invoke_episode_hypothesis(context: str) -> tuple[str, int]:
    system = (
        "You are an autonomous extractor analyzing your own behavior. "
        "Produce a concise hypothesis based on the iteration data."
    )
    instruction = (
        "Respond with one paragraph describing your hypothesis about "
        "what is causing the current pattern."
    )
    return await _invoke_field("episode_hypothesis", system, instruction, context,
                               max_input=_DECISION_MAX_INPUT)


async def invoke_episode_action(context: str) -> tuple[str, int]:
    system = (
        "You are an autonomous extractor planning modifications. "
        "Describe your planned action concisely."
    )
    instruction = (
        "Respond with one paragraph describing what you will change and why."
    )
    return await _invoke_field("action", system, instruction, context,
                               max_input=_DECISION_MAX_INPUT)


async def invoke_episode_expectation(context: str) -> tuple[str, int]:
    system = (
        "You are an autonomous extractor setting expectations. "
        "Describe what you expect to observe next."
    )
    instruction = (
        "Respond with one paragraph describing what you expect to observe "
        "in the next iteration as a result of your changes."
    )
    return await _invoke_field("expectation", system, instruction, context,
                               max_input=_DECISION_MAX_INPUT)


def _parse_edits_array(raw: str) -> list[dict]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "edits" in obj:
            return obj["edits"]
    except json.JSONDecodeError:
        pass

    raise json.JSONDecodeError("Could not parse edits array", raw[:200], 0)


async def invoke_edits(
    context: str,
    plan: str | None = None,
    iteration_n: int = -1,
) -> tuple[list[Edit], list[dict] | None]:
    system = (
        "You are an autonomous extractor proposing file edits. "
        "You must output ONLY a valid JSON array of edit objects."
    )
    instruction = (
        "Respond with ONLY a valid JSON array of edit objects. No other text. "
        "No markdown. Raw JSON array only.\n"
        "Schema: [{\"file_path\": \"...\", \"operation\": \"replace_string|replace_file|create_file|delete_file\", "
        "\"old_string\": \"...\", \"new_string\": \"...\", \"new_content\": \"...\"}]\n"
        "Each operation requires specific fields:\n"
        "- replace_string: file_path, old_string, new_string\n"
        "- replace_file: file_path, new_content\n"
        "- create_file: file_path, new_content\n"
        "- delete_file: file_path only"
    )
    # A006-1: the agent's own stated plan (hypothesis + action) is placed at the TAIL
    # of the user message, immediately before the instruction. It is preserved under
    # head-first budgeting AND varies every iteration, so the edits prompt is no longer
    # near-constant — this breaks the greedy fixed point that pinned the output to `[]`
    # (or, in the prior run, to a stale non-matching edit). It also conditions the
    # structured edit on the plan the narrator already committed to, closing the
    # narrator/generator split.
    plan_block = ""
    if plan:
        plan_block = (
            "YOUR STATED PLAN FOR THIS ITERATION (implement exactly this as concrete "
            "edits; if it truly requires no code change, return []):\n"
            f"{plan}\n\n"
        )
    # A004-1: no /no_think appendage. A006-2: edits decode sampled (see below), narrowing
    # A004-3 for this call only — a near-constant prompt must not pin to one output.
    full_user = context + "\n\n" + plan_block + instruction

    for attempt in range(3):
        try:
            raw, token_usage = await asyncio.to_thread(
                _get_provider().complete_with_usage,
                system,
                full_user,
                _EDITS_MAX_TOKENS,
                _DECISION_MAX_INPUT,
                True,  # A006-2: sample the edits call
            )

            print(f"\n{'='*80}", flush=True)
            print(f"RAW EDITS OUTPUT (attempt {attempt + 1}):", flush=True)
            print(f"{'='*80}", flush=True)
            print(raw, flush=True)
            print(f"{'='*80}", flush=True)

            edits_data = _parse_edits_array(raw)
            edits = []
            for ed in edits_data:
                op = str(ed.get("operation", ""))
                nc = ed.get("new_content")
                ns = ed.get("new_string")
                if op in ("replace_file", "create_file") and nc is None and ns is not None:
                    nc = ns
                edits.append(
                    Edit(
                        file_path=str(ed.get("file_path", "")),
                        operation=op,
                        old_string=ed.get("old_string"),
                        new_string=ns,
                        new_content=nc,
                    )
                )
            return edits, token_usage

        except (json.JSONDecodeError, Exception) as e:
            log_anomaly(_STUDY_ID, iteration_n, "field_call_failed", {
                "field": "edits",
                "error": str(e),
                "attempt": attempt + 1,
            })
            if attempt < 2:
                full_user = (
                    context + "\n\n"
                    + plan_block
                    + f"Previous edit output was malformed: {e}\n\n"
                    "Respond with ONLY a valid JSON array of edit objects. "
                    "No other text. No markdown. Raw JSON array only.\n"
                    "Schema: [{\"file_path\": \"...\", \"operation\": \"...\", ...}]\n/no_think"
                )
            else:
                raise

    return [], None


async def invoke_decision(
    assessment: AssessmentResult | None,
    current_files: dict[str, str],
    prior_episodes: list[dict] | None = None,
) -> AgentResponse | AgentFailure:
    field_failures = []
    context = _build_decision_context(assessment, current_files, prior_episodes)
    total_tokens = 0

    observation, tok = await invoke_episode_observation(context)
    total_tokens += tok
    if observation == "[not available — call failed]":
        field_failures.append("observation")

    hypothesis, tok = await invoke_episode_hypothesis(context)
    total_tokens += tok
    if hypothesis == "[not available — call failed]":
        field_failures.append("episode_hypothesis")

    action, tok = await invoke_episode_action(context)
    total_tokens += tok
    if action == "[not available — call failed]":
        field_failures.append("action")

    expectation, tok = await invoke_episode_expectation(context)
    total_tokens += tok
    if expectation == "[not available — call failed]":
        field_failures.append("expectation")

    # A006-1: condition the edits generation on the plan the agent just stated.
    plan = f"Hypothesis: {hypothesis}\nAction: {action}"
    try:
        edits, edits_token_usage = await invoke_edits(context, plan=plan)
        if edits_token_usage:
            total_tokens += edits_token_usage.total_tokens
    except Exception as e:
        log_anomaly(_STUDY_ID, -1, "field_call_failed", {
            "field": "edits",
            "error": str(e),
        })
        field_failures.append("edits")
        edits = []
        edits_token_usage = None

    episode = Episode(
        observation=observation,
        hypothesis=hypothesis,
        action=action,
        expectation=expectation,
        field_failures=field_failures,
    )

    rationale = (
        f"Observation: {observation}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Action: {action}\n"
        f"Expectation: {expectation}"
    )

    # Construct a synthetic TokenUsage for aggregate tracking
    class _SyntheticTokenUsage:
        def __init__(self):
            self.total_tokens = total_tokens
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.tokens_per_second = 0.0
            self.context_window = 0

    return AgentResponse(
        episode=episode,
        rationale=rationale,
        edits=edits,
        token_usage=_SyntheticTokenUsage(),
    )


async def invoke_repair(
    error_message: str,
    current_files: dict[str, str],
    attempt_number: int,
) -> RepairResponse | AgentFailure:
    system = (
        "Your previous edits to the scientific claim extractor produced a "
        "Python error. You must propose repair edits to fix the broken "
        "playground code. Respond with a JSON object containing only an "
        "\"edits\" array.\n\n"
        "REPAIR RESPONSE SCHEMA:\n"
        '{\n  "edits": [\n    {\n      "file_path": "string",\n      '
        '"operation": "replace_string | replace_file | create_file | delete_file",\n      '
        '"old_string": "string or null",\n      "new_string": "string or null",\n      '
        '"new_content": "string or null"\n    }\n  ]\n}\n\n'
        "CRITICAL: Your entire response must be ONLY the JSON object. "
        "Do NOT include any other text."
    )

    user_parts = [
        f"ERROR:\n{error_message}",
        "CURRENT FILE CONTENTS:",
    ]
    for filepath, content in sorted(current_files.items()):
        user_parts.append(f"--- {filepath} ---\n{content}")

    remaining = 3 - attempt_number
    user_parts.append(
        f"\nThis is repair attempt {attempt_number} of 3. "
        f"You have {remaining} remaining attempt(s) after this one.\n"
        "Fix the error. Output ONLY the JSON object, no other text."
    )
    user = "\n\n".join(user_parts)

    try:
        raw, token_usage = await asyncio.to_thread(
            _get_provider().complete_with_usage,
            system,
            user,
            _REPAIR_MAX_TOKENS,
            _REPAIR_MAX_INPUT,
            True,  # A005-5: sample on repair turns so successive attempts cannot
                   # re-emit an identical broken edit and lock into a fixed point.
        )
    except Exception as e:
        return AgentFailure(reason=f"Provider call failed: {e}")

    print(f"\n{'='*80}", flush=True)
    print(f"RAW REPAIR OUTPUT (attempt {attempt_number}):", flush=True)
    print(f"{'='*80}", flush=True)
    print(raw, flush=True)
    print(f"{'='*80}", flush=True)
    print(f"END RAW REPAIR OUTPUT\n", flush=True)

    try:
        raw = raw.strip()
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'```(?:json)?\s*(\{.*?)\s*```', raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                return AgentFailure(reason="Malformed repair JSON", raw_response=raw)
        else:
            return AgentFailure(reason="Malformed repair JSON", raw_response=raw)

    try:
        edits_data = data.get("edits", [])
        edits = []
        for ed in edits_data:
            op = str(ed.get("operation", ""))
            nc = ed.get("new_content")
            ns = ed.get("new_string")
            if op in ("replace_file", "create_file") and nc is None and ns is not None:
                nc = ns
            edits.append(
                Edit(
                    file_path=str(ed.get("file_path", "")),
                    operation=op,
                    old_string=ed.get("old_string"),
                    new_string=ns,
                    new_content=nc,
                )
            )
        return RepairResponse(edits=edits, token_usage=token_usage)
    except Exception as e:
        return AgentFailure(reason=f"Schema validation failed: {e}", raw_response=raw)
