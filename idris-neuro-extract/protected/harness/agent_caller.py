import asyncio
import json
import re
from pathlib import Path

from extractor.provider import LlamaCppProvider
from protected.harness.edit_protocol import (
    AgentFailure,
    AgentResponse,
    Episode,
    RepairResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_provider = LlamaCppProvider()
_AGENT_MAX_TOKENS = 28672
_REPAIR_MAX_TOKENS = 4096

OBJECTIVE = (
    "Improve the precision and recall of the scientific claim extractor by "
    "modifying its Python code and/or prompt files. You receive the prior "
    "iteration's extraction output (per-abstract predicted claims). You cannot "
    "see scores, ground truth, or evaluation metrics. Reason from the extraction "
    "output to decide what to change."
)

RESPONSE_SCHEMA = """
{
  "episode": {
    "observation": "What you noticed in the prior iteration's extraction output",
    "hypothesis": "What you think is wrong or could improve",
    "action": "What you changed and why",
    "expectation": "What you expect to see in the next iteration's output"
  },
  "rationale": "Free-form reasoning about your changes",
  "edits": [
    {
      "file_path": "playground/extractor.py",
      "operation": "replace_string | replace_file | create_file | delete_file",
      "old_string": "string or null",
      "new_string": "string or null",
      "new_content": "string or null"
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
) -> tuple[str, str]:
    system = (
        "You are an autonomous AI researcher modifying a scientific claim "
        "extractor to improve its precision and recall. You have access to a "
        "Python playground and a set of prompt files. You must respond with a "
        "JSON object matching the response schema exactly.\n\n"
        f"OBJECTIVE:\n{objective}\n\n"
    )

    if prior_episodes:
        system += (
            "EPISODIC MEMORY (prior iterations):\n"
            f"{json.dumps(prior_episodes, indent=2)}\n\n"
        )
    else:
        system += "This is your first iteration; you have no prior episodes.\n\n"

    system += (
        "CURRENT FILE CONTENTS:\n"
    )
    for filepath, content in sorted(current_files.items()):
        system += f"\n--- {filepath} ---\n{content}\n"

    system += f"\n\nRESPONSE SCHEMA:\n{RESPONSE_SCHEMA}\n"

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


def _parse_response(raw: str) -> dict:
    raw = raw.strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}") from e


async def invoke(
    prior_output: list[dict],
    prior_output_iteration: int,
    current_files: dict[str, str],
    objective: str = OBJECTIVE,
    prior_episodes: list[dict] | None = None,
) -> AgentResponse | AgentFailure:
    if prior_episodes is None:
        prior_episodes = []

    system_prompt, user_message = _build_invoke_prompt(
        prior_output, prior_output_iteration, current_files, objective, prior_episodes
    )

    try:
        raw, token_usage = await asyncio.to_thread(
            _provider.complete_with_usage, system_prompt, user_message, _AGENT_MAX_TOKENS
        )
    except Exception as e:
        return AgentFailure(reason=f"Provider call failed: {e}")

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
            from protected.harness.edit_protocol import Edit

            edits.append(
                Edit(
                    file_path=str(ed.get("file_path", "")),
                    operation=str(ed.get("operation", "")),
                    old_string=ed.get("old_string"),
                    new_string=ed.get("new_string"),
                    new_content=ed.get("new_content"),
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
            _provider.complete_with_usage, system_prompt, user_message, _REPAIR_MAX_TOKENS
        )
    except Exception as e:
        return AgentFailure(reason=f"Provider call failed: {e}")

    try:
        data = _parse_response(raw)
    except ValueError as e:
        return AgentFailure(reason=f"Malformed response: {e}", raw_response=raw)

    try:
        edits_data = data.get("edits", [])
        edits = []
        for ed in edits_data:
            from protected.harness.edit_protocol import Edit

            edits.append(
                Edit(
                    file_path=str(ed.get("file_path", "")),
                    operation=str(ed.get("operation", "")),
                    old_string=ed.get("old_string"),
                    new_string=ed.get("new_string"),
                    new_content=ed.get("new_content"),
                )
            )
        return RepairResponse(edits=edits, token_usage=token_usage)
    except Exception as e:
        return AgentFailure(reason=f"Schema validation failed: {e}", raw_response=raw)
