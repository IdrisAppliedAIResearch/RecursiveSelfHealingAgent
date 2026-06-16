import asyncio
import json
import re
from pathlib import Path

from protected.harness.shared.edit_protocol import (
    AgentFailure,
    AgentResponse,
    AssessmentResult,
    Edit,
    Episode,
    RepairResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DECISION_MAX_TOKENS = 8192
_DECISION_MAX_INPUT = 13107
_DIAGNOSTIC_MAX_TOKENS = 4096
_DIAGNOSTIC_MAX_INPUT = 13107
_REPAIR_MAX_TOKENS = 4096
_REPAIR_MAX_INPUT = 13107

ASSESSMENT_SCHEMA = """
ASSESSMENT SCHEMA (output only this JSON, no other text):
{
  "routing_trend": "improving | declining | flat",
  "last_action_effect": "string",
  "pattern_observed": "string",
  "hypothesis": "string"
}

FIELD DEFINITIONS:
- routing_trend: "improving" if aggregate routing score has net increased across the last 3 iterations (or all available if fewer than 3). "declining" if net decreased. "flat" if movement within +/-0.02. If first iteration with no prior data, value must be "flat".
- last_action_effect: Plain language description of what the prior modification did to routing scores, referencing specific control abstracts where movement was notable. If first iteration, value must be: "No prior modification has been made. This is the first iteration."
- pattern_observed: What you see across your episode history and routing trajectory taken together. Identify whether your modification strategy has been working, whether routing scores are responding to changes, and whether any systematic pattern is visible.
- hypothesis: What you think should change and why, based on the pattern observed. This is a forward-looking statement, not an edit instruction. Describe a direction ("the prompt needs to be more specific about X") rather than specific file changes.
"""

RESPONSE_SCHEMA = """
EDITS SCHEMA (use the correct fields for each operation):
- replace_string: requires file_path, old_string, new_string
- replace_file: requires file_path, new_content (full file content)
- create_file: requires file_path, new_content
- delete_file: requires file_path only

EXAMPLE RESPONSE (OUTPUT ONLY THIS JSON, NO OTHER TEXT):
{
  "episode": {
    "observation": "What you noticed",
    "hypothesis": "What could improve",
    "action": "What you changed",
    "expectation": "Expected result"
  },
  "rationale": "Reasoning",
  "edits": [
    {
      "file_path": "prompts/system_prompt.md",
      "operation": "replace_file",
      "new_content": "Full file content here"
    }
  ]
}
"""


def _summarize_prior_output(prior_output: list[dict]) -> list[dict]:
    """Strip abstract_text and keep only ID + claims for diagnostic context."""
    result = []
    for rec in prior_output[-10:]:
        result.append({
            "abstract_id": rec["abstract_id"],
            "predicted_claims": rec.get("predicted_claims", []),
        })
    return result


def _build_diagnostic_prompt(
    prior_output: list[dict],
    prior_output_iteration: int,
    routing_history_text: str,
    routing_delta_text: str,
    prior_episodes: list[dict],
) -> tuple[str, str]:
    system = (
        "You are an autonomous research system analyzing your own performance "
        "trajectory. You must produce a structured JSON assessment of what you "
        "observe. Do NOT propose edits in this call.\n\n"
        f"RESPONSE SCHEMA:\n{ASSESSMENT_SCHEMA}\n\n"
        "CRITICAL: Your entire response must be ONLY the JSON object. "
        "Do NOT include any analysis, reasoning, explanation, or markdown "
        "before or after the JSON."
    )

    user_parts = []

    if prior_episodes:
        user_parts.append(
            "EPISODIC MEMORY (prior iterations):\n"
            f"{json.dumps(prior_episodes, indent=2)}"
        )
    else:
        user_parts.append("This is your first iteration. You have no prior episodes.")

    if routing_history_text:
        user_parts.append(routing_history_text)

    if routing_delta_text:
        user_parts.append(f"ROUTING DELTA:\n{routing_delta_text}")

    summary = _summarize_prior_output(prior_output)
    user_parts.append(
        f"PRIOR EXTRACTION OUTPUT (from iteration {prior_output_iteration}, "
        f"last {len(summary)} of {len(prior_output)} entries):\n"
        f"{json.dumps(summary, indent=2)}"
    )

    user_parts.append(
        "\nNow produce your JSON assessment. Output ONLY the JSON object, "
        "no other text."
    )
    user_parts.append("/no_think")
    user = "\n\n".join(user_parts)

    return system, user


def _build_decision_prompt(
    assessment: AssessmentResult | None,
    current_files: dict[str, str],
) -> tuple[str, str]:
    from protected.harness.study_002.baseline_correction import (
        compose_baseline_correction,
    )

    system = (
        "You are an autonomous extractor modifying its own system. "
        "You have access to a Python playground and a set of prompt files. "
        "You must respond with a JSON object matching the response schema exactly.\n\n"
        f"RESPONSE SCHEMA:\n{RESPONSE_SCHEMA}\n\n"
        f"BASELINE CORRECTION:\n{compose_baseline_correction()}\n\n"
        "CRITICAL: Your entire response must be ONLY the JSON object. "
        "Do NOT include any analysis, reasoning, explanation, or markdown "
        "before or after the JSON. Put all reasoning inside the episode and "
        "rationale fields of the JSON."
    )

    user_parts = []

    if assessment:
        user_parts.append(
            "YOUR CURRENT ASSESSMENT\n\n"
            f"Routing trend: {assessment.routing_trend}\n\n"
            f"Effect of last action: {assessment.last_action_effect}\n\n"
            f"Pattern observed: {assessment.pattern_observed}\n\n"
            f"Hypothesis: {assessment.hypothesis}"
        )
    else:
        user_parts.append(
            "Assessment unavailable for this iteration due to a processing "
            "error. Proceed based on your current file state and the baseline "
            "correction guidance."
        )

    user_parts.append("CURRENT FILE CONTENTS:")
    for filepath, content in sorted(current_files.items()):
        user_parts.append(f"--- {filepath} ---\n{content}")

    user_parts.append(
        "\nBased on your assessment and the current file contents, "
        "decide what to modify and produce edit instructions. "
        "Output ONLY the JSON object, no other text."
    )
    user = "\n\n".join(user_parts)

    return system, user


def _build_repair_prompt(
    error_message: str,
    current_files: dict[str, str],
    attempt_number: int,
) -> tuple[str, str]:
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

    return system, user


def _get_provider():
    from protected.harness.shared.analyzer_registry import get_analyzer

    _inst = get_analyzer()
    if _inst is not None:
        return _inst
    raise RuntimeError("No model loaded. TRANSFORMERS_MODEL_PATH must be set.")


def _parse_response(raw: str) -> dict:
    raw = raw.strip()
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences if present
    m = re.search(r'```(?:json)?\s*(\{.*?)\s*```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Find all valid JSON objects, prioritize ones with target keys
    candidates = []
    for m in re.finditer(r'\{', raw):
        start = m.start()
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = raw[start:i+1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            candidates.append(obj)
                    except json.JSONDecodeError:
                        continue

    # Prioritize candidates with "episode" or "edits" keys
    for c in candidates:
        if "episode" in c or "edits" in c:
            return c

    # Fallback: last valid JSON object
    if candidates:
        return candidates[-1]

    raise json.JSONDecodeError("No valid JSON found", raw[:200], 0)


def _parse_assessment_response(raw: str) -> AssessmentResult | AgentFailure:
    raw = raw.strip()
    # Try direct parse first
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Strip markdown code fences
        m = re.search(r'```(?:json)?\s*(\{.*?)\s*```', raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                return AgentFailure(reason="Malformed assessment JSON", raw_response=raw)
        else:
            return AgentFailure(reason="Malformed assessment JSON", raw_response=raw)

    required = ["routing_trend", "last_action_effect", "pattern_observed", "hypothesis"]
    for field in required:
        if field not in data or not str(data[field]).strip():
            return AgentFailure(
                reason=f"Missing or empty required field: {field}",
                raw_response=raw,
            )

    result = AssessmentResult(
        routing_trend=str(data["routing_trend"]),
        last_action_effect=str(data["last_action_effect"]),
        pattern_observed=str(data["pattern_observed"]),
        hypothesis=str(data["hypothesis"]),
        raw_response=raw,
    )
    return result


async def invoke_diagnostic(
    prior_output: list[dict],
    prior_output_iteration: int,
    routing_history_text: str,
    routing_delta_text: str,
    prior_episodes: list[dict],
) -> AssessmentResult | AgentFailure:
    system_prompt, user_message = _build_diagnostic_prompt(
        prior_output,
        prior_output_iteration,
        routing_history_text,
        routing_delta_text,
        prior_episodes,
    )

    try:
        raw, token_usage = await asyncio.to_thread(
            _get_provider().complete_with_usage,
            system_prompt,
            user_message,
            _DIAGNOSTIC_MAX_TOKENS,
            _DIAGNOSTIC_MAX_INPUT,
        )
    except Exception as e:
        return AgentFailure(reason=f"Provider call failed: {e}")

    print(f"\n{'='*80}", flush=True)
    print(f"RAW DIAGNOSTIC OUTPUT (Call 1):", flush=True)
    print(f"{'='*80}", flush=True)
    print(raw, flush=True)
    print(f"{'='*80}", flush=True)
    print(f"END RAW DIAGNOSTIC OUTPUT\n", flush=True)

    assessment = _parse_assessment_response(raw)

    if isinstance(assessment, AssessmentResult):
        assessment.token_usage = token_usage

    return assessment


def _parse_episode_and_edits(data: dict, raw: str) -> AgentResponse | AgentFailure:
    try:
        episode_data = data.get("episode", {})
        episode = Episode(
            observation=str(episode_data.get("observation", "")),
            hypothesis=str(episode_data.get("hypothesis", "")),
            action=str(episode_data.get("action", "")),
            expectation=str(episode_data.get("expectation", "")),
        )
        rationale = str(data.get("rationale", ""))
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
        return AgentResponse(
            episode=episode,
            rationale=rationale,
            edits=edits,
        )
    except Exception as e:
        return AgentFailure(reason=f"Schema validation failed: {e}", raw_response=raw)


async def invoke_decision(
    assessment: AssessmentResult | None,
    current_files: dict[str, str],
) -> AgentResponse | AgentFailure:
    system_prompt, user_message = _build_decision_prompt(
        assessment, current_files
    )

    try:
        raw, token_usage = await asyncio.to_thread(
            _get_provider().complete_with_usage,
            system_prompt,
            user_message,
            _DECISION_MAX_TOKENS,
            _DECISION_MAX_INPUT,
        )
    except Exception as e:
        return AgentFailure(reason=f"Provider call failed: {e}")

    print(f"\n{'='*80}", flush=True)
    print(f"RAW DECISION OUTPUT (Call 2):", flush=True)
    print(f"{'='*80}", flush=True)
    print(raw, flush=True)
    print(f"{'='*80}", flush=True)
    print(f"END RAW DECISION OUTPUT\n", flush=True)

    try:
        data = _parse_response(raw)
    except ValueError as e:
        return AgentFailure(reason=f"Malformed response: {e}", raw_response=raw)

    resp = _parse_episode_and_edits(data, raw)
    if isinstance(resp, AgentResponse):
        resp.token_usage = token_usage
    return resp


async def invoke_repair(
    error_message: str,
    current_files: dict[str, str],
    attempt_number: int,
) -> RepairResponse | AgentFailure:
    system_prompt, user_message = _build_repair_prompt(
        error_message, current_files, attempt_number
    )

    try:
        raw, token_usage = await asyncio.to_thread(
            _get_provider().complete_with_usage,
            system_prompt,
            user_message,
            _REPAIR_MAX_TOKENS,
            _REPAIR_MAX_INPUT,
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
        data = _parse_response(raw)
    except ValueError as e:
        return AgentFailure(reason=f"Malformed response: {e}", raw_response=raw)

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
