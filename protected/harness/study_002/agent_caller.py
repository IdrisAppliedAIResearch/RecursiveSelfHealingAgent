import asyncio
import json
import re
from pathlib import Path

from protected.harness.shared.edit_protocol import (
    AgentFailure,
    AgentResponse,
    Edit,
    Episode,
    RepairResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_AGENT_MAX_TOKENS = 8192
_REPAIR_MAX_TOKENS = 1024

OBJECTIVE = (
    "Improve the precision and recall of the scientific claim extractor by "
    "modifying its Python code and/or prompt files. You receive the prior "
    "iteration's extraction output (per-abstract predicted claims). You cannot "
    "see scores, ground truth, or evaluation metrics. Reason from the extraction "
    "output to decide what to change."
)

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


def _read_current_files() -> dict[str, str]:
    files = {}
    for directory in ["playground", "prompts"]:
        dirpath = PROJECT_ROOT / directory
        if not dirpath.exists():
            continue
        for f in dirpath.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(PROJECT_ROOT))
                files[rel] = f.read_text(encoding="utf-8", errors="replace")
    return files


def _build_invoke_prompt(
    prior_output: list[dict],
    prior_output_iteration: int,
    current_files: dict[str, str],
    objective: str,
    prior_episodes: list[dict],
    routing_history_text: str,
    baseline_correction: str,
) -> tuple[str, str]:
    system = (
        "You are an autonomous AI researcher modifying a scientific claim "
        "extractor to improve its precision and recall. You have access to a "
        "Python playground and a set of prompt files. You must respond with a "
        "JSON object matching the response schema exactly.\n\n"
        f"OBJECTIVE:\n{objective}\n\n"
    )

    system += f"BASELINE CORRECTION:\n{baseline_correction}\n\n"

    if prior_episodes:
        system += (
            "EPISODIC MEMORY (prior iterations):\n"
            f"{json.dumps(prior_episodes, indent=2)}\n\n"
        )
    else:
        system += "This is your first iteration; you have no prior episodes.\n\n"

    if routing_history_text:
        system += f"{routing_history_text}\n\n"

    system += "CURRENT FILE CONTENTS:\n"
    for filepath, content in sorted(current_files.items()):
        system += f"\n--- {filepath} ---\n{content}\n"

    system += f"\n\nRESPONSE SCHEMA:\n{RESPONSE_SCHEMA}\n"
    system += (
        "\n\nCRITICAL: Your entire response must be ONLY the JSON object. "
        "Do NOT include any analysis, reasoning, explanation, or markdown "
        "before or after the JSON. Put all reasoning inside the episode and "
        "rationale fields of the JSON."
    )

    user = (
        f"PRIOR EXTRACTION OUTPUT (from iteration {prior_output_iteration}):\n"
        f"{json.dumps(prior_output, indent=2)}"
    )

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
        f"ERROR:\n{error_message}\n\n"
        "CURRENT FILE CONTENTS:\n"
    )
    for filepath, content in sorted(current_files.items()):
        system += f"\n--- {filepath} ---\n{content}\n"

    system += (
        "\n\nREPAIR RESPONSE SCHEMA:\n"
        '{\n  "edits": [\n    {\n      "file_path": "string",\n      '
        '"operation": "replace_string | replace_file | create_file | delete_file",\n      '
        '"old_string": "string or null",\n      "new_string": "string or null",\n      '
        '"new_content": "string or null"\n    }\n  ]\n}\n'
    )

    remaining = 3 - attempt_number
    user = (
        f"This is repair attempt {attempt_number} of 3. "
        f"You have {remaining} remaining attempt(s) after this one.\n"
        "Fix the error and return only the edits array."
    )

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

    # Find all valid JSON objects, prioritize ones with "episode" or "edits"
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


async def invoke(
    prior_output: list[dict],
    prior_output_iteration: int,
    current_files: dict[str, str],
    objective: str = OBJECTIVE,
    prior_episodes: list[dict] | None = None,
    routing_history_text: str = "",
) -> AgentResponse | AgentFailure:
    if prior_episodes is None:
        prior_episodes = []

    from protected.harness.study_002.baseline_correction import (
        compose_baseline_correction,
    )

    baseline_text = compose_baseline_correction()

    system_prompt, user_message = _build_invoke_prompt(
        prior_output,
        prior_output_iteration,
        current_files,
        objective,
        prior_episodes,
        routing_history_text,
        baseline_text,
    )

    try:
        raw, token_usage = await asyncio.to_thread(
            _get_provider().complete_with_usage,
            system_prompt,
            user_message,
            _AGENT_MAX_TOKENS,
        )
    except Exception as e:
        return AgentFailure(reason=f"Provider call failed: {e}")

    print(f"\n{'='*80}", flush=True)
    print(f"RAW AGENT OUTPUT:", flush=True)
    print(f"{'='*80}", flush=True)
    print(raw, flush=True)
    print(f"{'='*80}", flush=True)
    print(f"END RAW AGENT OUTPUT\n", flush=True)

    try:
        data = _parse_response(raw)
    except ValueError as e:
        return AgentFailure(reason=f"Malformed response: {e}", raw_response=raw)

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
            # Fallback: agent may use new_string for replace_file/create_file
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
            token_usage=token_usage,
        )
    except Exception as e:
        return AgentFailure(reason=f"Schema validation failed: {e}", raw_response=raw)


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
