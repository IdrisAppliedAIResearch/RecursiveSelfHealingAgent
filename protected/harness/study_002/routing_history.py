import json
from datetime import datetime, timezone
from pathlib import Path

from protected.attention.scorer import RoutingScore

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _routing_history_path(study_id: str) -> Path:
    return _PROJECT_ROOT / "experiments" / study_id / "routing_history.jsonl"


def _entry_scores(entry: dict) -> list[dict]:
    """A004-12: read the single `scores` field, falling back to the legacy
    `post_scores` for any records written before the schema change."""
    return entry.get("scores", entry.get("post_scores", []))


def _serialize(scores: list[RoutingScore]) -> list[dict]:
    result = []
    for s in scores:
        result.append({
            "abstract_id": s.abstract_id,
            "score": s.score,
            "score_start": s.score_start,
            "score_end": s.score_end,
            "intra_generation_delta": s.intra_generation_delta,
            "results_attention_fraction": s.results_attention_fraction,
            "methods_attention_fraction": s.methods_attention_fraction,
            "background_attention_fraction": s.background_attention_fraction,
            "n_results_tokens": s.n_results_tokens,
            "n_methods_tokens": s.n_methods_tokens,
            "n_background_tokens": s.n_background_tokens,
            "n_layers_used": s.n_layers_used,
        })
    return result


def append(
    study_id: str,
    iteration_n: int,
    scores: list[RoutingScore],
) -> None:
    # A004-12: a single end-of-generation `scores` list per iteration. The prior
    # pre_scores/post_scores pair was always identical and is removed; the
    # consequence signal is the inter-iteration delta computed against the previous
    # entry's aggregate.
    path = _routing_history_path(study_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = _serialize(scores)
    valid = [s["score"] for s in serialized if s["score"] is not None]
    aggregate = sum(valid) / len(valid) if valid else None

    prior = load_all(study_id)
    prev_agg = prior[-1].get("aggregate_score") if prior else None
    inter_delta = (
        aggregate - prev_agg
        if (aggregate is not None and prev_agg is not None)
        else None
    )

    record = {
        "iteration_n": iteration_n,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scores": serialized,
        "aggregate_score": aggregate,
        "inter_iteration_delta": inter_delta,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_all(study_id: str) -> list[dict]:
    path = _routing_history_path(study_id)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    records.sort(key=lambda r: r.get("iteration_n", 0))
    return records


def format_for_agent(
    history: list[dict],
    current_pre_scores: list[RoutingScore] | None = None,
) -> str:
    if not history:
        return "ROUTING HISTORY — No prior routing data available."

    abstract_ids = set()
    for entry in history:
        for s in _entry_scores(entry):
            abstract_ids.add(s["abstract_id"])
    abstract_ids = sorted(abstract_ids)

    iter_scores: dict[str, dict[int, float | None]] = {}
    iter_starts: dict[str, dict[int, float | None]] = {}
    iter_ends: dict[str, dict[int, float | None]] = {}
    iter_intras: dict[str, dict[int, float | None]] = {}
    for aid in abstract_ids:
        iter_scores[aid] = {}
        iter_starts[aid] = {}
        iter_ends[aid] = {}
        iter_intras[aid] = {}

    for entry in history:
        it = entry.get("iteration_n", 0)
        for s in _entry_scores(entry):
            aid = s["abstract_id"]
            if aid in iter_scores:
                iter_scores[aid][it] = s.get("score")
                iter_starts[aid][it] = s.get("score_start")
                iter_ends[aid][it] = s.get("score_end")
                iter_intras[aid][it] = s.get("intra_generation_delta")

    max_iter = max(entry.get("iteration_n", 0) for entry in history)

    current_scores: dict[str, float | None] = {}
    current_starts: dict[str, float | None] = {}
    current_ends: dict[str, float | None] = {}
    current_intras: dict[str, float | None] = {}
    if current_pre_scores:
        for s in current_pre_scores:
            current_scores[s.abstract_id] = s.score
            current_starts[s.abstract_id] = s.score_start
            current_ends[s.abstract_id] = s.score_end
            current_intras[s.abstract_id] = s.intra_generation_delta

    lines = []
    lines.append("ROUTING HISTORY — Attention to Results Sentences\n")

    header = f"{'Abstract':<14} | Start | End   | Intra Δ"
    for it in range(max_iter + 1):
        header += f" | Iter {it}"
    if current_pre_scores:
        header += " | Current"
    lines.append(header)
    lines.append("-" * len(header))

    def _fmt(v: float | None) -> str:
        if v is None:
            return "  N/A"
        return f"  {v:.2f}"

    def _fmt_delta(v: float | None) -> str:
        if v is None:
            return "    N/A"
        return f"  {v:+.2f}"

    for aid in abstract_ids:
        start_avg = (sum(iter_starts[aid].values()) / len(iter_starts[aid]) if iter_starts[aid] else None)
        end_avg = (sum(iter_ends[aid].values()) / len(iter_ends[aid]) if iter_ends[aid] else None)
        intra_avg = (sum(iter_intras[aid].values()) / len(iter_intras[aid]) if iter_intras[aid] else None)
        row = f"{aid:<14} | {_fmt(start_avg)} | {_fmt(end_avg)} | {_fmt_delta(intra_avg)}"
        for it in range(max_iter + 1):
            val = iter_scores[aid].get(it)
            row += f" | {_fmt(val)}"
        if aid in current_scores:
            row += f" | {_fmt(current_scores[aid])}"
        lines.append(row)

    agg_scores = []
    agg_starts = []
    agg_ends = []
    agg_intras = []
    for it in range(max_iter + 1):
        vals = []
        start_vals = []
        end_vals = []
        intra_vals = []
        for aid in abstract_ids:
            v = iter_scores[aid].get(it)
            if v is not None:
                vals.append(v)
            sv = iter_starts[aid].get(it)
            if sv is not None:
                start_vals.append(sv)
            ev = iter_ends[aid].get(it)
            if ev is not None:
                end_vals.append(ev)
            iv = iter_intras[aid].get(it)
            if iv is not None:
                intra_vals.append(iv)
        agg_scores.append(sum(vals) / len(vals) if vals else None)
        agg_starts.append(sum(start_vals) / len(start_vals) if start_vals else None)
        agg_ends.append(sum(end_vals) / len(end_vals) if end_vals else None)
        agg_intras.append(sum(intra_vals) / len(intra_vals) if intra_vals else None)

    agg_row = f"{'AGGREGATE':<14} | {_fmt(agg_starts[-1] if agg_starts else None)} | {_fmt(agg_ends[-1] if agg_ends else None)} | {_fmt_delta(agg_intras[-1] if agg_intras else None)}"
    for it in range(max_iter + 1):
        agg_row += f" | {_fmt(agg_scores[it])}"
    if current_pre_scores:
        cur_vals = [s for s in current_scores.values() if s is not None]
        cur_agg = sum(cur_vals) / len(cur_vals) if cur_vals else None
        agg_row += f" | {_fmt(cur_agg)}"
    lines.append(agg_row)

    lines.append("")
    lines.append("Interpretation notes:")
    lines.append("- Start score: where the model's attention is grounded before generating")
    lines.append("- End score: where it is grounded as it completes generation")
    lines.append("- Intra Δ: positive means attention improved during generation;")
    lines.append("            negative means it drifted away from results content")
    lines.append("- Iter delta: change from prior iteration's end score to this iteration's")

    if max_iter >= 0:
        prev_iter = max_iter
        prev_agg = agg_scores[prev_iter] if prev_iter < len(agg_scores) else None
        if current_pre_scores and prev_agg is not None:
            cur_vals = [s for s in current_scores.values() if s is not None]
            cur_agg = sum(cur_vals) / len(cur_vals) if cur_vals else None
            if cur_agg is not None:
                delta = cur_agg - prev_agg
                prev_map = {aid: {"score": iter_scores[aid].get(prev_iter)} for aid in abstract_ids}
                curr_map = {aid: {"score": current_scores[aid]} for aid in abstract_ids if aid in current_scores}
                improved, _, largest_inc, largest_dec, largest_inc_aid, largest_dec_aid = _compute_delta_stats(prev_map, curr_map)

                lines.append("")
                lines.append(f"Current iteration routing signal:")
                lines.append(
                    f"After your iteration {prev_iter} modification, routing toward results "
                    f"sentence{'s' if improved != 1 else ''} "
                    f"increased on {improved} of {len(abstract_ids)} control abstracts "
                    f"(aggregate: {delta:+.2f})."
                )
                if largest_inc_aid:
                    lines.append(
                        f"The largest increase was on abstract {largest_inc_aid} "
                        f"(+{largest_inc:.2f})."
                    )
                if largest_dec_aid:
                    lines.append(
                        f"The largest decrease was on abstract {largest_dec_aid} "
                        f"({largest_dec:+.2f})."
                    )

                if len(history) >= 3:
                    recent = agg_scores[-3:]
                    if all(v is not None for v in recent):
                        net_shift = recent[-1] - recent[0]
                        lines.append("")
                        if abs(net_shift) > 0.02:
                            direction = "positive" if net_shift > 0 else "negative"
                            lines.append(
                                f"Trend: Your last 3 modifications have produced a net "
                                f"{direction} routing shift of {net_shift:+.2f}."
                            )
                            if cur_agg and cur_agg < 0.5:
                                lines.append(
                                    "The aggregate routing score has not yet crossed 0.5 "
                                    "(the threshold at which the model attends more to "
                                    "results than to non-results content on average)."
                                )
                        else:
                            lines.append(
                                "Trend: Your last 3 modifications have produced no "
                                "meaningful routing change. Consider a different "
                                "modification approach."
                            )

    return "\n".join(lines)


def _compute_delta_stats(
    prev_map: dict, curr_map: dict
) -> tuple:
    """Return (improved, deltas, largest_inc, largest_dec, largest_inc_aid, largest_dec_aid)."""
    improved = 0
    deltas = []
    largest_inc = 0.0
    largest_dec = 0.0
    largest_inc_aid = ""
    largest_dec_aid = ""
    for aid in curr_map:
        prev_score = prev_map.get(aid, {}).get("score")
        curr_score = curr_map.get(aid, {}).get("score")
        if prev_score is not None and curr_score is not None:
            d = curr_score - prev_score
            deltas.append(d)
            if d > 0.02:
                improved += 1
                if d > largest_inc:
                    largest_inc = d
                    largest_inc_aid = aid
            elif d < -0.02:
                if d < largest_dec:
                    largest_dec = d
                    largest_dec_aid = aid
    return improved, deltas, largest_inc, largest_dec, largest_inc_aid, largest_dec_aid


def format_routing_delta(study_id: str, iteration_n: int) -> str:
    if iteration_n <= 1:
        return "No prior modification has been made."

    history = load_all(study_id)
    if len(history) < 2:
        return "No prior modification has been made."

    prev = history[-2]
    curr = history[-1]
    prev_post = _entry_scores(prev)
    curr_post = _entry_scores(curr)

    prev_map = {s["abstract_id"]: s for s in prev_post}
    curr_map = {s["abstract_id"]: s for s in curr_post}

    improved, deltas, largest_inc, largest_dec, largest_inc_aid, largest_dec_aid = _compute_delta_stats(prev_map, curr_map)

    total = len(deltas)
    agg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    prev_iter_n = prev.get("iteration_n", iteration_n - 1)

    parts = [
        f"After your iteration {prev_iter_n} modification, routing toward results "
        f"sentence{'s' if improved != 1 else ''} "
        f"increased on {improved} of {total} control abstracts "
        f"(aggregate delta: {agg_delta:+.2f})."
    ]
    if largest_inc_aid:
        parts.append(
            f"The largest increase was on abstract {largest_inc_aid} (+{largest_inc:.2f})."
        )
    if largest_dec_aid:
        parts.append(
            f"The largest decrease was on abstract {largest_dec_aid} ({largest_dec:+.2f})."
        )

    return "\n".join(parts)
