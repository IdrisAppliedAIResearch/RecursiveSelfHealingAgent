# Recursive Self-Healing Agent

**Idris Applied AI Research** | May 2026

A pre-registered study investigating whether an autonomous agent can iteratively improve a scientific claim extractor through self-directed modification of its own code and prompts, operating only on the signal of its prior outputs with no score feedback.

## Research Question

Can an autonomous agent improve the precision and recall of a naive scientific claim extractor by modifying its own extraction code and prompt layer, operating only on the signal of its prior extraction outputs against a fixed corpus?

## Key Finding

**No.** Across 14 iterations (0-13), macro-F1 degraded from 0.467 to 0.142 — a 69.6% decline. The agent exclusively modified prompts (never Python code), pursued progressively over-constraining extraction criteria, and became trapped in a local minimum. The hidden-score constraint, intended to prevent metric gaming, proved sufficient to prevent improvement.

## Repository Structure

```
dev-history/          # Sprint specs and design decisions (001-003)
experiments/
  study_001/          # Completed study: pre-registration, artifacts, results, paper
    pre-registration.md    # Locked study design (commit 7f1acdd)
    paper.md               # Full analysis and findings
    metrics.jsonl          # Per-iteration metrics (14 iterations)
    episodes.jsonl         # Agent episodic memory (9 episodes)
    anomalies.jsonl        # Structured anomaly events
    iterations/            # Per-iteration playground snapshots
    ...
```

## Study Design

- **Corpus:** 200 neuroscience abstracts from NeuroSynth, annotated with ground truth claims
- **Agent:** Qwen 3.6 27B MTP 6-bit GGUF, running locally via llama.cpp on RTX 5090
- **Constraint:** Agent receives no scores, no ground truth, no evaluation metrics
- **Signal:** Only the prior iteration's raw extraction output (predicted claim lists)
- **Memory:** Episodic, append-only ledger of observation/hypothesis/action/expectation
- **Scope:** Agent may modify Python code and prompt files in a scoped playground

## Three Observed Phases

1. **Initial Degradation (iterations 1-3):** Few-shot additions reduced F1 by 44%
2. **Total Collapse (iterations 4-6, 8):** Empty prompts and malformed responses produced zero output
3. **Degraded Floor (iterations 7-13):** Recovery to stable F1 0.13-0.18, never approaching baseline

## Conclusion

Self-healing agents without score feedback degrade rather than improve. Future work requires at least approximate score feedback, a calibration mechanism, or a dual-agent architecture to navigate the precision-recall tradeoff.

See `experiments/study_001/paper.md` for the full analysis.
