import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def ensure_branch(branch_name: str) -> None:
    try:
        _run_git(["branch", "--show-current"])
    except subprocess.CalledProcessError:
        _run_git(["checkout", "-b", branch_name])
        return
    current = _run_git(["branch", "--show-current"]).stdout.strip()
    if current != branch_name:
        branches = _run_git(["branch"]).stdout
        if f"  {branch_name}" not in branches and f"* {branch_name}" not in branches:
            _run_git(["checkout", "-b", branch_name])
        else:
            _run_git(["checkout", branch_name])


def verify_no_remote_push(branch_name: str) -> None:
    git_config = PROJECT_ROOT / ".git" / "config"
    if git_config.exists():
        config_text = git_config.read_text(encoding="utf-8")
        upstream_info = _run_git(["config", f"branch.{branch_name}.remote"], check=False)
        if upstream_info.returncode == 0 and upstream_info.stdout.strip():
            raise RuntimeError(
                f"Branch '{branch_name}' has remote upstream configured. "
                "The harness prohibits remote push operations."
            )


def reset_partial_iteration() -> None:
    has_commits = _run_git(["rev-parse", "HEAD"], check=False)
    if has_commits.returncode == 0:
        _run_git(["reset", "--hard", "HEAD"])


def rollback_playground() -> None:
    # Restore tracked files to HEAD...
    _run_git(["checkout", "--", "playground/", "prompts/"], check=False)
    # A009-6 (audit #3): ...and drop any UNTRACKED files the rolled-back iteration created
    # (e.g. a new `playground/preprocessor.py`). `git checkout` leaves those in place, and
    # the next `commit_iteration`'s `git add -A` would otherwise commit the orphan and
    # contaminate the baseline + the agent's view of the current files.
    _run_git(["clean", "-fd", "--", "playground/", "prompts/"], check=False)


def commit_iteration(iteration_n: int, study_id: str, rationale: str) -> str:
    iters_dir = PROJECT_ROOT / "experiments" / study_id / "iterations"
    files_changed = []
    if iters_dir.exists():
        for f in iters_dir.glob(f"iteration_{iteration_n:02d}_*"):
            files_changed.append(str(f.relative_to(PROJECT_ROOT)))

    rationale_snippet = rationale[:500].replace('\n', ' ').replace('\r', ' ') if rationale else "N/A"
    files_list = ", ".join(files_changed) if files_changed else "none"

    message = (
        f"[{study_id}] iteration {iteration_n:02d}\n\n"
        f"Rationale: {rationale_snippet}\n\n"
        f"Files changed: {files_list}\n\n"
        f"Playground files at this commit are the exact state used for this\n"
        f"iteration's corpus run. See experiments/{study_id}/pre-registration.md."
    )

    # A004-15: stage only the study's artifacts and the mutable surface, not `.`,
    # so iteration commits never capture incidental untracked files. `-A` with a
    # pathspec includes deletions (e.g. an agent-removed playground module).
    _run_git(["add", "-A", "--",
              f"experiments/{study_id}", "playground", "prompts"])
    diff_check = _run_git(["diff", "--cached", "--quiet"], check=False)
    if diff_check.returncode == 0:
        return ""
    result = _run_git(["commit", "-m", message])
    return result.stdout.strip()


def last_committed_iteration(study_id: str) -> int:
    metrics_path = PROJECT_ROOT / "experiments" / study_id / "metrics.jsonl"
    if not metrics_path.exists():
        return -1
    last_n = -1
    for line in metrics_path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            n = record.get("iteration_n", -1)
            if n > last_n:
                last_n = n
    return last_n
