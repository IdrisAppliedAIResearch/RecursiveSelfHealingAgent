import json
from pathlib import Path

from protected.attention.scorer import RoutingScore

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _routing_history_path(study_id: str) -> Path:
    return _PROJECT_ROOT / "experiments" / study_id / "routing_history.jsonl"


def append(
    study_id: str,
    iteration_n: int,
    pre_scores: list[RoutingScore],
    post_scores: list[RoutingScore],
) -> None:
    path = _routing_history_path(study_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(scores: list[RoutingScore]) -> list[dict]:
        result = []
        for s in scores:
            result.append({
                "abstract_id": s.abstract_id,
                "score": s.score,
                "results_attention_fraction": s.results_attention_fraction,
                "methods_attention_fraction": s.methods_attention_fraction,
                "background_attention_fraction": s.background_attention_fraction,
                "n_results_tokens": s.n_results_tokens,
                "n_methods_tokens": s.n_methods_tokens,
                "n_background_tokens": s.n_background_tokens,
                "n_layers_used": s.n_layers_used,
            })
        return result

    record = {
        "iteration_n": iteration_n,
        "pre_scores": _serialize(pre_scores),
        "post_scores": _serialize(post_scores),
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
        for s in entry.get("post_scores", []):
            abstract_ids.add(s["abstract_id"])
    abstract_ids = sorted(abstract_ids)

    iter_scores: dict[str, dict[int, float | None]] = {}
    for aid in abstract_ids:
        iter_scores[aid] = {}

    for entry in history:
        it = entry.get("iteration_n", 0)
        for s in entry.get("post_scores", []):
            aid = s["abstract_id"]
            if aid in iter_scores:
                iter_scores[aid][it] = s.get("score")

    max_iter = max(entry.get("iteration_n", 0) for entry in history)

    current_scores: dict[str, float | None] = {}
    if current_pre_scores:
        for s in current_pre_scores:
            current_scores[s.abstract_id] = s.score

    lines = []
    lines.append("ROUTING HISTORY — Attention to Results Sentences\n")

    header = f"{'Abstract':<14}"
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

    for aid in abstract_ids:
        row = f"{aid:<14}"
        for it in range(max_iter + 1):
            val = iter_scores[aid].get(it)
            row += f" | {_fmt(val)}"
        if aid in current_scores:
            row += f" | {_fmt(current_scores[aid])}"
        lines.append(row)

    agg_scores = []
    for it in range(max_iter + 1):
        vals = []
        for aid in abstract_ids:
            v = iter_scores[aid].get(it)
            if v is not None:
                vals.append(v)
        agg_scores.append(sum(vals) / len(vals) if vals else None)

    agg_row = f"{'AGGREGATE':<14}"
    for it in range(max_iter + 1):
        agg_row += f" | {_fmt(agg_scores[it])}"
    if current_pre_scores:
        cur_vals = [s for s in current_scores.values() if s is not None]
        cur_agg = sum(cur_vals) / len(cur_vals) if cur_vals else None
        agg_row += f" | {_fmt(cur_agg)}"
    lines.append(agg_row)

    if max_iter >= 0:
        prev_iter = max_iter
        prev_agg = agg_scores[prev_iter] if prev_iter < len(agg_scores) else None
        if current_pre_scores and prev_agg is not None:
            cur_vals = [s for s in current_scores.values() if s is not None]
            cur_agg = sum(cur_vals) / len(cur_vals) if cur_vals else None
            if cur_agg is not None:
                delta = cur_agg - prev_agg
                improved = 0
                declined = 0
                largest_inc = 0.0
                largest_dec = 0.0
                largest_inc_aid = ""
                largest_dec_aid = ""
                for aid in abstract_ids:
                    prev_val = iter_scores[aid].get(prev_iter)
                    cur_val = current_scores.get(aid)
                    if prev_val is not None and cur_val is not None:
                        d = cur_val - prev_val
                        if d > 0.02:
                            improved += 1
                            if d > largest_inc:
                                largest_inc = d
                                largest_inc_aid = aid
                        elif d < -0.02:
                            declined += 1
                            if d < largest_dec:
                                largest_dec = d
                                largest_dec_aid = aid

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
