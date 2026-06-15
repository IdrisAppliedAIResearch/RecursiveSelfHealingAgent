import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from protected.attention.analyzer import AttentionAnalyzer
from protected.attention.scorer import RoutingScore
from protected.harness.shared.analyzer_registry import set_analyzer, get_analyzer
from protected.harness.shared.anomaly_logger import log_anomaly
from protected.harness.shared.artifact_writer import (
    append_assessment,
    append_metrics,
    append_rationale,
    snapshot_playground,
    write_iteration_artifacts,
)
from protected.harness.shared.corpus_runner import run_corpus
from protected.harness.shared.edit_applier import apply_edits
from protected.harness.shared.edit_protocol import AgentFailure, AssessmentResult
from protected.harness.shared.episode_store import append as append_episode
from protected.harness.shared.episode_store import load_all as load_episodes
from protected.harness.shared.git_ops import (
    commit_iteration,
    ensure_branch,
    last_committed_iteration,
    reset_partial_iteration,
    rollback_playground,
    verify_no_remote_push,
)
from protected.harness.shared.interface_validator import validate_interface
from protected.harness.shared.model_performance import append_after_iteration, summarize
from protected.harness.study_002.agent_caller import invoke_diagnostic, invoke_decision, invoke_repair
from protected.harness.study_002.routing_history import (
    append as append_routing,
    format_for_agent,
    format_routing_delta,
    load_all as load_routing,
)
from protected.harness.shared.allowlist import ALLOWED_FILE_EXACT
from protected.interface import ITERATION_TIMEOUT_S
from protected.scorer import score_corpus

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
STUDY_ID = "study_002"
N_ITERATIONS = 25


def _load_probe_set() -> list[str]:
    probe_path = PROJECT_ROOT / "experiments" / STUDY_ID / "probe_set.json"
    if not probe_path.exists():
        raise RuntimeError("probe_set.json not found — run select_probe_set.py first")
    data = json.loads(probe_path.read_text(encoding="utf-8"))
    return data["abstract_ids"]


def _load_impact_abstracts(probe_ids: list[str], n: int = 15) -> list[str]:
    abstracts_dir = PROJECT_ROOT / "corpus" / "abstracts"
    all_ids = [f.stem for f in sorted(abstracts_dir.glob("*.json"))]
    impact_ids = [aid for aid in all_ids if aid not in probe_ids]
    import random

    random.seed(42)
    return random.sample(impact_ids, min(n, len(impact_ids)))


def _load_abstract_text(abstract_id: str) -> str:
    af = PROJECT_ROOT / "corpus" / "abstracts" / f"{abstract_id}.json"
    data = json.loads(af.read_text(encoding="utf-8", errors="replace"))
    return data.get("abstract", data.get("text", ""))


def _get_mini_corpus_files() -> list[Path]:
    probe_ids = _load_probe_set()
    impact_ids = _load_impact_abstracts(probe_ids)
    all_ids = probe_ids + impact_ids
    abstracts_dir = PROJECT_ROOT / "corpus" / "abstracts"
    files = []
    for aid in all_ids:
        af = abstracts_dir / f"{aid}.json"
        if af.exists():
            files.append(af)
    return files


def _get_control_corpus_files() -> list[Path]:
    probe_ids = _load_probe_set()
    abstracts_dir = PROJECT_ROOT / "corpus" / "abstracts"
    files = []
    for aid in probe_ids:
        af = abstracts_dir / f"{aid}.json"
        if af.exists():
            files.append(af)
    return files


def _load_ground_truth() -> dict[str, list[str]]:
    gt_path = PROJECT_ROOT / "corpus" / "ground_truth.jsonl"
    gt = {}
    for line in gt_path.read_text(encoding="utf-8-sig").strip().splitlines():
        if line.strip():
            try:
                entry = json.loads(line)
                gt[entry["abstract_id"]] = entry["claims"]
            except json.JSONDecodeError:
                pass
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

    try:
        records = json.loads(best_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], 0
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
        "assessment_available": None,
        "assessment_routing_trend": None,
        "call_1_prompt_tokens": None,
        "call_1_completion_tokens": None,
        "call_1_tokens_per_second": None,
        "call_2_prompt_tokens": None,
        "call_2_completion_tokens": None,
        "call_2_tokens_per_second": None,
        "corpus_total_prompt_tokens": None,
        "corpus_total_completion_tokens": None,
        "corpus_avg_tokens_per_abstract": None,
        "corpus_avg_tokens_per_second": None,
        "repair_prompt_tokens": None,
        "repair_completion_tokens": None,
        "pre_routing_score": None,
        "post_routing_score": None,
        "routing_delta": None,
        "routing_direction": None,
        "control_abstracts_improved": 0,
        "control_abstracts_declined": 0,
        "code_changes_attempted": False,
    }


def _read_system_prompt() -> str:
    prompts_dir = PROJECT_ROOT / "prompts"
    sp = prompts_dir / "system_prompt.md"
    examples = prompts_dir / "examples.md"
    prompt = sp.read_text(encoding="utf-8") if sp.exists() else ""
    ex = examples.read_text(encoding="utf-8").strip() if examples.exists() else ""
    if ex:
        prompt = prompt + "\n\n" + ex
    return prompt


def _run_attention_subprocess(study_id: str, iteration_n: int, analyzer: AttentionAnalyzer) -> list[dict]:
    output_path = PROJECT_ROOT / "experiments" / study_id / f"attention_scores_{iteration_n}.json"
    cmd = [
        sys.executable, "-m", "protected.attention.forward_pass_runner",
        "--study", study_id,
        "--iteration", str(iteration_n),
        "--output", str(output_path),
    ]

    print(f"  [attention] Releasing main-process VRAM before subprocess...", flush=True)
    analyzer.close()
    torch.cuda.empty_cache()

    try:
        print(f"  [attention] Running subprocess: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"Attention subprocess exited with code {result.returncode}")
    finally:
        print(f"  [attention] Reloading model after subprocess...", flush=True)
        analyzer.load()
        set_analyzer(analyzer)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    return data["scores"]


def _dicts_to_routing_scores(score_dicts: list[dict]) -> list[RoutingScore]:
    return [
        RoutingScore(
            abstract_id=s["abstract_id"],
            score=s.get("score"),
            results_attention_fraction=s.get("results_attention_fraction", 0.0),
            methods_attention_fraction=s.get("methods_attention_fraction", 0.0),
            background_attention_fraction=s.get("background_attention_fraction", 0.0),
            n_results_tokens=s.get("n_results_tokens", 0),
            n_methods_tokens=s.get("n_methods_tokens", 0),
            n_background_tokens=s.get("n_background_tokens", 0),
            n_layers_used=s.get("n_layers_used", 0),
        )
        for s in score_dicts
    ]


def _check_code_changes(edits) -> bool:
    from protected.harness.shared.edit_protocol import Edit as EditType

    for edit in edits:
        if isinstance(edit, EditType):
            path = edit.file_path
        else:
            path = edit.get("file_path", "")
        if path.startswith("playground/") and path.endswith(".py"):
            return True
    return False


def _pre_run_checks(study_id: str) -> None:
    pre_reg = PROJECT_ROOT / "experiments" / study_id / "pre-registration.md"
    if not pre_reg.exists():
        raise RuntimeError(f"Pre-registration not found: {pre_reg}")

    probe_path = PROJECT_ROOT / "experiments" / study_id / "probe_set.json"
    if not probe_path.exists():
        raise RuntimeError("probe_set.json not found — run select_probe_set.py first")

    gt_path = PROJECT_ROOT / "corpus" / "ground_truth.jsonl"
    if not gt_path.exists():
        raise RuntimeError("Ground truth not found")

    manifest = PROJECT_ROOT / "corpus" / "corpus_manifest.md"
    if not manifest.exists():
        raise RuntimeError("Corpus manifest not found")

    for fp in ALLOWED_FILE_EXACT:
        full = PROJECT_ROOT / fp
        if not full.exists():
            raise RuntimeError(f"Required file missing: {fp}")

    branch_name = f"experiment/{study_id}"
    ensure_branch(branch_name)
    verify_no_remote_push(branch_name)

    metrics_path = PROJECT_ROOT / "experiments" / study_id / "metrics.jsonl"
    if metrics_path.exists():
        count = 0
        for line in metrics_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    pass
        if count >= 21:
            raise RuntimeError(
                f"Study {study_id} already complete ({count} metrics entries). "
                f"Delete experiments/{study_id}/ to re-run."
            )

    transformers_path = os.environ.get("TRANSFORMERS_MODEL_PATH")
    if not transformers_path:
        raise RuntimeError("TRANSFORMERS_MODEL_PATH must be set (e.g., C:\\Users\\muzaf\\.cache\\huggingface\\hub\\models--Qwen--Qwen3.6-27B)")

    harness_diff = subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "protected/harness/", "protected/attention/"],
        cwd=PROJECT_ROOT,
        capture_output=True,
    )
    if harness_diff.returncode != 0:
        raise RuntimeError(
            "protected/harness/ or protected/attention/ has uncommitted modifications relative to HEAD"
        )


async def _run_baseline(study_id: str) -> None:
    ground_truth = _load_ground_truth()
    probe_ids = _load_probe_set()

    mini_files = _get_mini_corpus_files()
    print(f"  Mini-corpus ({len(mini_files)} abstracts) run starting...", flush=True)
    t0 = time.time()
    try:
        corpus_result = await asyncio.wait_for(
            run_corpus(study_id, mini_files),
            timeout=ITERATION_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        print(f"  Corpus TIMEOUT after {time.time()-t0:.0f}s")
        log_anomaly(study_id, 0, "iteration_timeout", {})
        metrics = _make_metrics_base(0)
        metrics["anomaly"] = True
        append_metrics(0, study_id, metrics)
        return
    except Exception as e:
        print(f"  Corpus FAILED: {e}")
        log_anomaly(study_id, 0, "scan_failure", {"error": str(e)})
        metrics = _make_metrics_base(0)
        metrics["anomaly"] = True
        append_metrics(0, study_id, metrics)
        return
    elapsed = time.time() - t0
    print(f"  Mini-corpus done in {elapsed:.0f}s: {len(corpus_result.results)} abstracts, {len(corpus_result.failures)} failures", flush=True)

    score_result = score_corpus(corpus_result.results, ground_truth)
    print(f"  Baseline scores: P={score_result['macro_precision']:.3f} R={score_result['macro_recall']:.3f} F1={score_result['macro_f1']:.3f}", flush=True)

    pre_scores: list[RoutingScore] = []
    post_scores: list[RoutingScore] = []
    pre_agg = None
    post_agg = None

    print("  Running attention analysis on control abstracts (subprocess)...", flush=True)
    analyzer = get_analyzer()
    score_dicts = _run_attention_subprocess(study_id, 0, analyzer)
    post_scores = _dicts_to_routing_scores(score_dicts)
    post_agg = (
        sum(s.score for s in post_scores if s.score is not None)
        / max(1, len(post_scores))
    )

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
    metrics["pre_routing_score"] = pre_agg
    metrics["post_routing_score"] = post_agg

    append_routing(study_id, 0, pre_scores, post_scores)
    write_iteration_artifacts(0, study_id, corpus_result)
    snapshot_playground(0, study_id)
    append_metrics(0, study_id, metrics)
    append_after_iteration(study_id, 0)


async def _run_iteration(
    iteration_n: int,
    study_id: str,
) -> str | None:
    prior_output, prior_iter = _load_prior_output(study_id)
    if not prior_output:
        log_anomaly(study_id, iteration_n, "no_prior_output", {})
        metrics = _make_metrics_base(iteration_n)
        metrics["anomaly"] = True
        append_metrics(iteration_n, study_id, metrics)
        return None

    prior_episodes = load_episodes(study_id)
    current_files = _current_files()

    routing_history = load_routing(study_id)
    routing_history_text = format_for_agent(routing_history)
    routing_delta_text = format_routing_delta(study_id, iteration_n)

    metrics = _make_metrics_base(iteration_n)

    # b1 — Diagnostic Assessment (Call 1)
    print(f"  [Call 1] Diagnostic assessment (output from iteration {prior_iter}, {len(prior_episodes)} prior episodes)...")
    assessment = await invoke_diagnostic(
        prior_output=prior_output,
        prior_output_iteration=prior_iter,
        routing_history_text=routing_history_text,
        routing_delta_text=routing_delta_text,
        prior_episodes=prior_episodes,
    )

    if isinstance(assessment, AgentFailure):
        print(f"  [Call 1] Assessment FAILED: {assessment.reason}")
        log_anomaly(
            study_id, iteration_n,
            "assessment_malformed",
            {"reason": assessment.reason},
        )
        metrics["assessment_available"] = False
        assessment = None

    if assessment is not None:
        metrics["assessment_available"] = True
        metrics["assessment_routing_trend"] = assessment.routing_trend
        trend = assessment.routing_trend
        if trend not in ("improving", "declining", "flat"):
            log_anomaly(
                study_id, iteration_n,
                "assessment_field_invalid",
                {"field": "routing_trend", "value": trend},
            )
        append_assessment(iteration_n, study_id, assessment)
        if assessment.token_usage is not None:
            tu1 = assessment.token_usage
            metrics["call_1_prompt_tokens"] = tu1.prompt_tokens
            metrics["call_1_completion_tokens"] = tu1.completion_tokens
            metrics["call_1_tokens_per_second"] = tu1.tokens_per_second

    # b2 — Modification Decision (Call 2)
    print(f"  [Call 2] Modification decision...")
    agent_result = await invoke_decision(
        assessment=assessment,
        current_files=current_files,
    )

    if isinstance(agent_result, AgentFailure):
        print(f"  [Call 2] Agent FAILED: {agent_result.reason}")
        log_anomaly(
            study_id, iteration_n,
            "agent_response_malformed",
            {"reason": agent_result.reason},
        )
        metrics["anomaly"] = True
        metrics["scanned"] = False
        append_metrics(iteration_n, study_id, metrics)
        return None

    if agent_result.token_usage:
        tu2 = agent_result.token_usage
        metrics["call_2_prompt_tokens"] = tu2.prompt_tokens
        metrics["call_2_completion_tokens"] = tu2.completion_tokens
        metrics["call_2_tokens_per_second"] = tu2.tokens_per_second

        metrics["agent_prompt_tokens"] = tu2.prompt_tokens
        metrics["agent_completion_tokens"] = tu2.completion_tokens
        metrics["agent_total_tokens"] = tu2.total_tokens
        metrics["agent_tokens_per_second"] = tu2.tokens_per_second
        metrics["agent_context_window"] = tu2.context_window

    metrics["agent_edits_proposed"] = len(agent_result.edits)
    metrics["code_changes_attempted"] = _check_code_changes(agent_result.edits)

    print(f"  Agent: hypothesis={agent_result.episode.hypothesis[:100]}...")
    print(f"  Agent: expectation={agent_result.episode.expectation[:100]}...")
    print(f"  Agent proposed {len(agent_result.edits)} edits")

    apply_result = apply_edits(agent_result.edits)
    if not apply_result.applied:
        print(f"  Edits REJECTED: {apply_result.reason} ({apply_result.offending_path})")
        log_anomaly(
            study_id, iteration_n,
            apply_result.reason or "edit_apply_failed",
            {"offending_path": apply_result.offending_path},
        )
        metrics["anomaly"] = True
        append_metrics(iteration_n, study_id, metrics)
        return None

    metrics["agent_edits_applied"] = len(apply_result.files_changed or [])
    metrics["playground_files_changed"] = apply_result.files_changed or []
    print(f"  Edits applied: {apply_result.files_changed}")

    repair_attempts = 0
    repair_prompt_tokens = 0
    repair_completion_tokens = 0
    validation_ok = False
    for attempt in range(1, 4):
        val_result = await validate_interface()
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
            rollback_playground()
            log_anomaly(study_id, iteration_n, "repair_exhausted", {})
            log_anomaly(study_id, iteration_n, "episode_discarded", {})
            metrics["repair_attempts"] = 3
            metrics["anomaly"] = True
            append_metrics(iteration_n, study_id, metrics)
            return None

        repair_files = _current_files()
        print(f"  Repair attempt {attempt}: calling agent...")
        repair_result = await invoke_repair(
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
            rollback_playground()
            metrics["repair_attempts"] = repair_attempts
            metrics["anomaly"] = True
            append_metrics(iteration_n, study_id, metrics)
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
            rollback_playground()
            metrics["repair_attempts"] = repair_attempts
            metrics["anomaly"] = True
            append_metrics(iteration_n, study_id, metrics)
            return None

    if not validation_ok:
        rollback_playground()
        metrics["anomaly"] = True
        append_metrics(iteration_n, study_id, metrics)
        return None

    metrics["repair_attempts"] = repair_attempts
    if repair_attempts > 0:
        metrics["repair_prompt_tokens"] = repair_prompt_tokens
        metrics["repair_completion_tokens"] = repair_completion_tokens

    pre_scores: list[RoutingScore] = []
    post_scores: list[RoutingScore] = []
    pre_agg = None
    post_agg = None
    routing_delta = None

    print(f"  Running POST-MODIFICATION attention pass (subprocess)...")
    analyzer = get_analyzer()
    score_dicts = _run_attention_subprocess(study_id, iteration_n, analyzer)
    post_scores = _dicts_to_routing_scores(score_dicts)

    routing_history = load_routing(study_id)
    if routing_history:
        last_entry = routing_history[-1]
        last_post = last_entry.get("post_scores", [])
        pre_scores = post_scores
        prev_scores_map = {s["abstract_id"]: s for s in last_post}
        pre_agg_list = []
        post_agg_list = []
        control_improved = 0
        control_declined = 0
        for s in post_scores:
            prev = prev_scores_map.get(s.abstract_id, {})
            prev_score = prev.get("score")
            if prev_score is not None and s.score is not None:
                pre_agg_list.append(prev_score)
                post_agg_list.append(s.score)
                d = s.score - prev_score
                if d > 0.02:
                    control_improved += 1
                elif d < -0.02:
                    control_declined += 1

        pre_agg = sum(pre_agg_list) / len(pre_agg_list) if pre_agg_list else None
        post_agg = sum(post_agg_list) / len(post_agg_list) if post_agg_list else None
        routing_delta = post_agg - pre_agg if (pre_agg is not None and post_agg is not None) else None
        metrics["control_abstracts_improved"] = control_improved
        metrics["control_abstracts_declined"] = control_declined
    else:
        pre_agg = (
            sum(s.score for s in post_scores if s.score is not None)
            / max(1, len(post_scores))
        )
        post_agg = pre_agg
        pre_scores = post_scores

    snapshot_playground(iteration_n, study_id)

    print(f"  Mini-corpus run starting...")
    t0 = time.time()
    mini_files = _get_mini_corpus_files()
    try:
        corpus_result = await asyncio.wait_for(
            run_corpus(study_id, mini_files),
            timeout=ITERATION_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        print(f"  Corpus TIMEOUT after {time.time()-t0:.0f}s")
        rollback_playground()
        log_anomaly(study_id, iteration_n, "iteration_timeout", {})
        log_anomaly(study_id, iteration_n, "episode_discarded", {})
        metrics["anomaly"] = True
        append_metrics(iteration_n, study_id, metrics)
        return None
    except Exception as e:
        print(f"  Corpus FAILED: {e}")
        rollback_playground()
        log_anomaly(
            study_id, iteration_n,
            "scan_failure",
            {"error": str(e)},
        )
        log_anomaly(study_id, iteration_n, "episode_discarded", {})
        metrics["anomaly"] = True
        append_metrics(iteration_n, study_id, metrics)
        return None

    elapsed = time.time() - t0
    print(f"  Mini-corpus done in {elapsed:.0f}s: {len(corpus_result.results)} abstracts, {len(corpus_result.failures)} failures")

    ground_truth = _load_ground_truth()
    score_result = score_corpus(corpus_result.results, ground_truth)
    print(f"  Scores: P={score_result['macro_precision']:.3f} R={score_result['macro_recall']:.3f} F1={score_result['macro_f1']:.3f}")

    append_episode(study_id, iteration_n, agent_result.episode)
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
    metrics["pre_routing_score"] = pre_agg
    metrics["post_routing_score"] = post_agg
    metrics["routing_delta"] = routing_delta
    if routing_delta is not None:
        if routing_delta > 0.02:
            metrics["routing_direction"] = "positive"
        elif routing_delta < -0.02:
            metrics["routing_direction"] = "negative"
        else:
            metrics["routing_direction"] = "neutral"
    else:
        metrics["routing_direction"] = "neutral"

    append_routing(study_id, iteration_n, pre_scores, post_scores)
    write_iteration_artifacts(iteration_n, study_id, corpus_result)
    append_metrics(iteration_n, study_id, metrics)
    append_rationale(iteration_n, study_id, agent_result.rationale)
    append_after_iteration(study_id, iteration_n)

    return agent_result.rationale


async def _run_study_async(study_id: str, n_iterations: int) -> None:
    _pre_run_checks(study_id)

    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    last = last_committed_iteration(study_id)
    start_iter = last + 1

    # Load model for agent completions only — forward pass runs in subprocess
    print("  Loading AttentionAnalyzer model for agent calls...")
    model_path = os.environ.get("TRANSFORMERS_MODEL_PATH", "")
    if not model_path:
        raise RuntimeError("TRANSFORMERS_MODEL_PATH not set")
    analyzer = AttentionAnalyzer(model_path)
    analyzer.load()
    set_analyzer(analyzer)
    print("  AttentionAnalyzer loaded.")

    if start_iter == 0:
        print(f"[{study_id}] Running baseline (iteration 0)...", flush=True)
        await _run_baseline(study_id)
        commit_iteration(0, study_id, "Baseline run")
        start_iter = 1

    for i in range(start_iter, n_iterations + 1):
        print(f"[{study_id}] Running iteration {i}...", flush=True)
        rationale = await _run_iteration(i, study_id)
        commit_iteration(i, study_id, rationale or f"Iteration {i}")
        print(f"[{study_id}] Iteration {i} committed.")

    analyzer.close()
    set_analyzer(None)
    print(f"[{study_id}] Study complete. {n_iterations + 1} iterations total.")
    summarize(study_id)


def run_study(study_id: str = STUDY_ID, n_iterations: int = N_ITERATIONS) -> None:
    asyncio.run(_run_study_async(study_id, n_iterations))


if __name__ == "__main__":
    run_study()
