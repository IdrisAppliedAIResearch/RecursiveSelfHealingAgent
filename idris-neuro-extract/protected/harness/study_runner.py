import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from protected.harness import (
    agent_caller,
    allowlist,
    anomaly_logger,
    artifact_writer,
    corpus_runner,
    episode_store,
    git_ops,
    interface_validator,
)
from protected.harness.edit_applier import apply_edits
from protected.harness.edit_protocol import AgentFailure
from protected.harness.anomaly_logger import log_anomaly
from protected.interface import ITERATION_TIMEOUT_S

from protected.scorer import score_corpus

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class StudyAlreadyComplete(Exception):
    pass


def _load_ground_truth() -> dict[str, list[str]]:
    gt_path = PROJECT_ROOT / "corpus" / "ground_truth.jsonl"
    gt = {}
    for line in gt_path.read_text().strip().splitlines():
        if line.strip():
            entry = json.loads(line)
            gt[entry["abstract_id"]] = entry["claims"]
    return gt


def _load_prior_output(study_id: str) -> tuple[list[dict], int]:
    iters_dir = PROJECT_ROOT / "experiments" / study_id / "iterations"
    if not iters_dir.exists():
        return [], 0

    best_iter = -1
    best_path = None
    for f in iters_dir.glob("iteration_*.json"):
        name = f.stem
        parts = name.split("_")
        try:
            iter_n = int(parts[1])
            if iter_n > best_iter:
                best_iter = iter_n
                best_path = f
        except (ValueError, IndexError):
            continue

    if best_path is None:
        return [], 0

    records = json.loads(best_path.read_text())
    prior_output = []
    for rec in records:
        prior_output.append({
            "abstract_id": rec["abstract_id"],
            "abstract_text": rec.get("abstract_text", ""),
            "predicted_claims": rec.get("predicted_claims", []),
        })
    return prior_output, best_iter


def _current_files() -> dict[str, str]:
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


def _make_metrics_base(iteration_n: int) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "micro_tp": None,
        "micro_fp": None,
        "micro_fn": None,
        "avg_claims_per_abstract": None,
        "scan_duration_seconds": None,
        "agent_edits_proposed": 0,
        "agent_edits_applied": 0,
        "repair_attempts": 0,
        "playground_files_changed": [],
        "prompt_chars_delta": 0,
        "anomaly": False,
        "episode_persisted": False,
        "scanned": False,
        "agent_prompt_tokens": None,
        "agent_completion_tokens": None,
        "agent_total_tokens": None,
        "agent_tokens_per_second": None,
        "agent_context_window": None,
        "corpus_total_prompt_tokens": None,
        "corpus_total_completion_tokens": None,
        "corpus_avg_tokens_per_abstract": None,
        "corpus_avg_tokens_per_second": None,
        "repair_prompt_tokens": None,
        "repair_completion_tokens": None,
    }


def _pre_run_checks(study_id: str) -> None:
    pre_reg = PROJECT_ROOT / "experiments" / study_id / "pre-registration.md"
    if not pre_reg.exists():
        raise RuntimeError(f"Pre-registration not found: {pre_reg}")

    gt_path = PROJECT_ROOT / "corpus" / "ground_truth.jsonl"
    if not gt_path.exists():
        raise RuntimeError("Ground truth not found")

    manifest = PROJECT_ROOT / "corpus" / "corpus_manifest.md"
    if not manifest.exists():
        raise RuntimeError("Corpus manifest not found")
    if "commit_sha" not in manifest.read_text():
        raise RuntimeError("Corpus manifest missing commit_sha")

    for fp in allowlist.ALLOWED_FILE_EXACT:
        full = PROJECT_ROOT / fp
        if not full.exists():
            raise RuntimeError(f"Required file missing: {fp}")

    branch_name = f"experiment/{study_id}"
    git_ops.ensure_branch(branch_name)
    git_ops.verify_no_remote_push(branch_name)

    metrics_path = PROJECT_ROOT / "experiments" / study_id / "metrics.jsonl"
    if metrics_path.exists():
        count = 0
        for line in metrics_path.read_text().strip().splitlines():
            if line.strip():
                count += 1
        if count >= 21:
            raise StudyAlreadyComplete(
                f"Study {study_id} already complete ({count} metrics entries)"
            )

    episodes_count = episode_store.count(study_id)
    if metrics_path.exists() and episodes_count > 0:
        persisted_count = 0
        for line in metrics_path.read_text().strip().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("episode_persisted"):
                    persisted_count += 1
        if episodes_count != persisted_count:
            raise RuntimeError(
                f"Episode count ({episodes_count}) does not match "
                f"episode_persisted count ({persisted_count}) in metrics"
            )

    llm_provider = os.environ.get("LLM_PROVIDER")
    llama_url = os.environ.get("LLAMA_CPP_BASE_URL")
    if not llm_provider or not llama_url:
        raise RuntimeError("LLM_PROVIDER and LLAMA_CPP_BASE_URL must be set")

    harness_diff = subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "protected/harness/"],
        cwd=PROJECT_ROOT,
        capture_output=True,
    )
    if harness_diff.returncode != 0:
        raise RuntimeError(
            "protected/harness/ has uncommitted modifications relative to HEAD"
        )


async def _run_baseline(study_id: str) -> None:
    ground_truth = _load_ground_truth()
    print(f"  Corpus run starting...")
    t0 = time.time()
    corpus_result = await asyncio.wait_for(
        corpus_runner.run_corpus(study_id),
        timeout=ITERATION_TIMEOUT_S,
    )
    elapsed = time.time() - t0
    print(f"  Corpus done in {elapsed:.0f}s: {len(corpus_result.results)} abstracts, {len(corpus_result.failures)} failures")

    score_result = score_corpus(corpus_result.results, ground_truth)
    print(f"  Baseline scores: P={score_result['macro_precision']:.3f} R={score_result['macro_recall']:.3f} F1={score_result['macro_f1']:.3f}")

    metrics = _make_metrics_base(0)
    metrics["scanned"] = True
    metrics["macro_precision"] = score_result["macro_precision"]
    metrics["macro_recall"] = score_result["macro_recall"]
    metrics["macro_f1"] = score_result["macro_f1"]
    metrics["micro_tp"] = score_result["micro_tp"]
    metrics["micro_fp"] = score_result["micro_fp"]
    metrics["micro_fn"] = score_result["micro_fn"]
    metrics["avg_claims_per_abstract"] = score_result["avg_claims_per_abstract"]
    metrics["scan_duration_seconds"] = corpus_result.duration_seconds
    metrics["corpus_total_prompt_tokens"] = (
        corpus_result.corpus_token_usage.total_prompt_tokens
    )
    metrics["corpus_total_completion_tokens"] = (
        corpus_result.corpus_token_usage.total_completion_tokens
    )
    metrics["corpus_avg_tokens_per_abstract"] = (
        corpus_result.corpus_token_usage.avg_tokens_per_abstract
    )
    metrics["corpus_avg_tokens_per_second"] = (
        corpus_result.corpus_token_usage.avg_tokens_per_second
    )

    artifact_writer.write_iteration_artifacts(0, study_id, corpus_result)
    artifact_writer.snapshot_playground(0, study_id)
    artifact_writer.append_metrics(0, study_id, metrics)


async def _run_iteration(iteration_n: int, study_id: str) -> str | None:
    prior_output, prior_iter = _load_prior_output(study_id)
    if not prior_output:
        log_anomaly(study_id, iteration_n, "no_prior_output", {})
        metrics = _make_metrics_base(iteration_n)
        metrics["anomaly"] = True
        artifact_writer.append_metrics(iteration_n, study_id, metrics)
        return None

    prior_episodes = episode_store.load_all(study_id)
    current_files = _current_files()

    print(f"  Calling agent (output from iteration {prior_iter}, {len(prior_episodes)} prior episodes)...")
    agent_result = await agent_caller.invoke(
        prior_output=prior_output,
        prior_output_iteration=prior_iter,
        current_files=current_files,
        prior_episodes=prior_episodes,
    )

    metrics = _make_metrics_base(iteration_n)

    if isinstance(agent_result, AgentFailure):
        print(f"  Agent FAILED: {agent_result.reason}")
        log_anomaly(
            study_id, iteration_n,
            "agent_response_malformed",
            {"reason": agent_result.reason},
        )
        metrics["anomaly"] = True
        artifact_writer.append_metrics(iteration_n, study_id, metrics)
        return None

    metrics["agent_edits_proposed"] = len(agent_result.edits)

    if agent_result.token_usage:
        tu = agent_result.token_usage
        metrics["agent_prompt_tokens"] = tu.prompt_tokens
        metrics["agent_completion_tokens"] = tu.completion_tokens
        metrics["agent_total_tokens"] = tu.total_tokens
        metrics["agent_tokens_per_second"] = tu.tokens_per_second
        metrics["agent_context_window"] = tu.context_window

    print(f"  Agent: hypothesis={agent_result.episode.hypothesis[:100]}...")
    print(f"  Agent: expectation={agent_result.episode.expectation[:100]}...")
    print(f"  Agent proposed {len(agent_result.edits)} edits")
    if not agent_result.edits:
        log_anomaly(study_id, iteration_n, "empty_edits", {})

    apply_result = apply_edits(agent_result.edits)
    if not apply_result.applied:
        print(f"  Edits REJECTED: {apply_result.reason} ({apply_result.offending_path})")
        log_anomaly(
            study_id, iteration_n,
            apply_result.reason or "edit_apply_failed",
            {"offending_path": apply_result.offending_path},
        )
        metrics["anomaly"] = True
        artifact_writer.append_metrics(iteration_n, study_id, metrics)
        return None

    metrics["agent_edits_applied"] = len(apply_result.files_changed or [])
    metrics["playground_files_changed"] = apply_result.files_changed or []
    print(f"  Edits applied: {apply_result.files_changed}")

    # Repair loop: up to 3 total attempts (original + 2 repairs)
    repair_attempts = 0
    repair_prompt_tokens = 0
    repair_completion_tokens = 0
    validation_ok = False
    for attempt in range(1, 4):
        val_result = await interface_validator.validate_interface()
        if val_result.valid:
            validation_ok = True
            if attempt > 1:
                print(f"  Interface valid after repair attempt {attempt}")
            break

        print(f"  Interface INVALID (attempt {attempt}): {val_result.error}")

        log_anomaly(
            study_id, iteration_n,
            "interface_validation_failed",
            {"error": val_result.error, "attempt": attempt},
        )

        if attempt == 3:
            git_ops.rollback_playground()
            log_anomaly(study_id, iteration_n, "repair_exhausted", {})
            log_anomaly(study_id, iteration_n, "episode_discarded", {})
            metrics["repair_attempts"] = 3
            metrics["anomaly"] = True
            artifact_writer.append_metrics(iteration_n, study_id, metrics)
            return None

        repair_files = _current_files()
        print(f"  Repair attempt {attempt}: calling agent...")
        repair_result = await agent_caller.invoke_repair(
            error_message=val_result.error or "Validation failed",
            current_files=repair_files,
            attempt_number=attempt,
        )
        repair_attempts = attempt

        if isinstance(repair_result, AgentFailure):
            log_anomaly(
                study_id, iteration_n,
                "repair_agent_failure",
                {"reason": repair_result.reason},
            )
            git_ops.rollback_playground()
            metrics["repair_attempts"] = repair_attempts
            metrics["anomaly"] = True
            artifact_writer.append_metrics(iteration_n, study_id, metrics)
            return None

        repair_apply = apply_edits(repair_result.edits)
        if repair_result.token_usage:
            repair_prompt_tokens += repair_result.token_usage.prompt_tokens
            repair_completion_tokens += repair_result.token_usage.completion_tokens
        if not repair_apply.applied:
            log_anomaly(
                study_id, iteration_n,
                repair_apply.reason or "repair_edit_apply_failed",
                {"offending_path": repair_apply.offending_path},
            )
            git_ops.rollback_playground()
            metrics["repair_attempts"] = repair_attempts
            metrics["anomaly"] = True
            artifact_writer.append_metrics(iteration_n, study_id, metrics)
            return None

    if not validation_ok:
        git_ops.rollback_playground()
        metrics["anomaly"] = True
        artifact_writer.append_metrics(iteration_n, study_id, metrics)
        return None

    metrics["repair_attempts"] = repair_attempts
    if repair_attempts > 0:
        metrics["repair_prompt_tokens"] = repair_prompt_tokens
        metrics["repair_completion_tokens"] = repair_completion_tokens
    artifact_writer.snapshot_playground(iteration_n, study_id)

    print(f"  Corpus run starting...")
    t0 = time.time()
    try:
        corpus_result = await asyncio.wait_for(
            corpus_runner.run_corpus(study_id),
            timeout=ITERATION_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        print(f"  Corpus TIMEOUT after {time.time()-t0:.0f}s")
        git_ops.rollback_playground()
        log_anomaly(study_id, iteration_n, "iteration_timeout", {})
        log_anomaly(study_id, iteration_n, "episode_discarded", {})
        metrics["anomaly"] = True
        artifact_writer.append_metrics(iteration_n, study_id, metrics)
        return None
    except Exception as e:
        print(f"  Corpus FAILED: {e}")
        git_ops.rollback_playground()
        log_anomaly(
            study_id, iteration_n,
            "scan_failure",
            {"error": str(e)},
        )
        log_anomaly(study_id, iteration_n, "episode_discarded", {})
        metrics["anomaly"] = True
        artifact_writer.append_metrics(iteration_n, study_id, metrics)
        return None

    elapsed = time.time() - t0
    print(f"  Corpus done in {elapsed:.0f}s: {len(corpus_result.results)} abstracts, {len(corpus_result.failures)} failures")

    ground_truth = _load_ground_truth()
    score_result = score_corpus(corpus_result.results, ground_truth)
    print(f"  Scores: P={score_result['macro_precision']:.3f} R={score_result['macro_recall']:.3f} F1={score_result['macro_f1']:.3f}")

    episode_store.append(study_id, iteration_n, agent_result.episode)
    metrics["episode_persisted"] = True

    metrics["scanned"] = True
    metrics["macro_precision"] = score_result["macro_precision"]
    metrics["macro_recall"] = score_result["macro_recall"]
    metrics["macro_f1"] = score_result["macro_f1"]
    metrics["micro_tp"] = score_result["micro_tp"]
    metrics["micro_fp"] = score_result["micro_fp"]
    metrics["micro_fn"] = score_result["micro_fn"]
    metrics["avg_claims_per_abstract"] = score_result["avg_claims_per_abstract"]
    metrics["scan_duration_seconds"] = corpus_result.duration_seconds
    metrics["corpus_total_prompt_tokens"] = (
        corpus_result.corpus_token_usage.total_prompt_tokens
    )
    metrics["corpus_total_completion_tokens"] = (
        corpus_result.corpus_token_usage.total_completion_tokens
    )
    metrics["corpus_avg_tokens_per_abstract"] = (
        corpus_result.corpus_token_usage.avg_tokens_per_abstract
    )
    metrics["corpus_avg_tokens_per_second"] = (
        corpus_result.corpus_token_usage.avg_tokens_per_second
    )

    artifact_writer.write_iteration_artifacts(iteration_n, study_id, corpus_result)
    artifact_writer.append_metrics(iteration_n, study_id, metrics)
    artifact_writer.append_rationale(iteration_n, study_id, agent_result.rationale)

    return agent_result.rationale


async def _run_study_async(study_id: str, n_iterations: int) -> None:
    _pre_run_checks(study_id)
    git_ops.reset_partial_iteration()

    last = git_ops.last_committed_iteration(study_id)
    start_iter = last + 1

    if start_iter == 0:
        print(f"[{study_id}] Running baseline (iteration 0)...")
        await _run_baseline(study_id)
        git_ops.commit_iteration(0, study_id, "Baseline run")
        start_iter = 1

    for i in range(start_iter, n_iterations + 1):
        print(f"[{study_id}] Running iteration {i}...")
        rationale = await _run_iteration(i, study_id)
        git_ops.commit_iteration(i, study_id, rationale or f"Iteration {i}")
        print(f"[{study_id}] Iteration {i} committed.")

    print(f"[{study_id}] Study complete. {n_iterations + 1} iterations total.")


def run_study(study_id: str = "study_001", n_iterations: int = 20) -> None:
    asyncio.run(_run_study_async(study_id, n_iterations))
