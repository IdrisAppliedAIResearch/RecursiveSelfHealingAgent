# Recursive Self-Healing Agent

**Idris Applied AI Research** | Updated July 2026

A pre-registered research program investigating whether an autonomous agent can iteratively improve a scientific claim extractor through self-directed modification of its own code and prompts, operating **blind to its own score**. The program isolates one candidate remedy per study, so that any effect can be attributed to a single mechanism.

## Research Question

Can an autonomous agent improve the precision and recall of a naive scientific claim extractor by modifying its own extraction code and prompt layer, operating only on the signal of its prior extraction outputs against a fixed corpus, with no access to F1, ground truth, or evaluation metrics?

## The Isolation Arc

Study 001 established the failure baseline. Each subsequent study adds exactly one structural component to that baseline, holding everything else constant, so the arc reads as a controlled ablation of what a self-modifying agent needs.

| Study | Component under test | Status | Result |
|-------|----------------------|--------|--------|
| **001** | (baseline — no consequence signal) | Complete | **Negative** — F1 0.467 → 0.142; agent edited only prompts, never code |
| **002** | Intrinsic cost signal (attention routing fidelity) + baseline correction | Complete | **Soft negative** — correction unlocked code edits; routing signal did not guide them |
| 003–006 | (reserved: configurator, calibration, dual-agent, …) | Planned | — |

## Key Findings So Far

**Study 001 — No.** Across fourteen iterations (0–13), macro-F1 degraded from 0.467 to 0.142, a 69.6% decline. The agent exclusively modified prompts, never Python code, pursued progressively over-constraining extraction criteria, and became trapped in a local minimum without ever detecting its own decline. The hidden-score constraint, intended to prevent metric gaming, proved sufficient to prevent improvement.

**Study 002 — A signal can unlock code self-modification without guiding it.** Adding an intrinsic cost signal (attention routing fidelity) plus a baseline capability correction, the agent modified `playground/extractor.py` at four iterations — clearing the behavioral-differentiation criterion Study 001 met at zero. But the code edits are most parsimoniously attributed to the baseline correction, not the routing signal, which was inert as a guide: it declined to a frozen 0.0058 and never once read as improving. At iteration 6 the agent refactored the extractor into a contradictory two-call pipeline that produced **zero claims on every abstract from iteration 6 through 12** — an extraction death the routing signal, which measures attention over the model's *input* rather than the health of the downstream pipeline, reported as flat and unremarkable. Reported as a **soft negative**: attention routing fidelity, as instantiated here, is structurally decoupled from the quantity the agent was trying to improve.

## Repository Structure

```
dev-history/          # Sprint specs and design decisions
experiments/
  study_001/          # Complete (negative)
    pre-registration.md    # Locked study design (commit 7f1acdd)
    paper.md               # Full analysis and findings
    metrics.jsonl          # Per-iteration metrics
    episodes.jsonl         # Agent episodic memory
    anomalies.jsonl        # Structured anomaly events
    iterations/            # Per-iteration playground snapshots
  study_002/          # Complete (soft negative)
    README.md              # Full findings paper (renders on GitHub)
    pre-registration.md    # Locked study design + amendment addendum
    make_figures.py        # Reproducible figure generation from metrics.jsonl
    figures/               # Publication figures (F1/routing, precision/recall, replication)
    brief.html             # Interactive visual research brief
    metrics.jsonl          # Per-iteration metrics (iters 0–12)
    routing_history.jsonl  # Per-control-abstract routing scores
    episodes.jsonl         # Agent episodic memory
    anomalies.jsonl        # Structured anomaly events
    assessments.jsonl      # Per-iteration assessments
    probe_set.json         # Fixed 10-abstract control set
```

## Study Design (shared across the program)

- **Corpus:** 200 neuroscience abstracts from NeuroSynth, annotated with ground truth claims
- **Agent:** Qwen 3.6 27B, running locally
- **Constraint:** Agent receives no scores, no ground truth, no evaluation metrics
- **Signal:** Only the prior iteration's raw extraction output — plus, from Study 002 on, the study's one added component
- **Memory:** Episodic, append-only ledger of observation / hypothesis / action / falsifiable expectation
- **Scope:** Agent may modify Python code and prompt files in a scoped playground
- **Scorer:** `rapidfuzz` token-sort-ratio matching at threshold 80, macro-averaged (in `protected/`)

## Reading the Studies

- **Study 001** — full analysis in [`experiments/study_001/paper.md`](experiments/study_001/paper.md).
- **Study 002** — full findings paper in [`experiments/study_002/README.md`](experiments/study_002/README.md), with figures under `figures/` and an interactive brief in `brief.html`.
