# Autonomous Architectural Self-Modification Without Score Feedback Degrades Extraction Quality: A Pre-Registered Study

**Muzaffer Ozen**

*Idris Applied AI Research*

*May 2026*

---

## Abstract

We investigate whether an autonomous agent can iteratively improve the precision and recall of a scientific claim extractor through self-directed modification of its own code and prompt layer, operating only on the signal of its prior extraction outputs against a fixed corpus. The agent, powered by Qwen 3.6 27B running locally, was given full access to a Python playground and prompt files over 200 neuroscience abstracts from NeuroSynth. Evaluation metrics (precision, recall, F1) and ground truth were hidden from the agent by design. The study was pre-registered at commit SHA `7f1acdd` (`experiments/study_001/pre-registration-proof.txt`, line 1) and terminated early at iteration 13 of 20 based on convergence evidence. Across 14 iterations (0–13), macro-F1 degraded from a baseline of 0.467 to 0.142 — a 69.6% decline. Three distinct phases emerged: (1) initial degradation (iterations 1–3), where the agent's first few-shot prompt modifications immediately reduced F1 by 44%; (2) total collapse (iterations 4–6, 8), where the agent's architectural changes produced zero valid claims across all 200 abstracts; and (3) partial recovery to a degraded floor (iterations 7–13), where the agent recovered to a stable band of F1 0.13–0.18 but never approached baseline. The agent exclusively modified prompt files, never touching Python code, and pursued a strategy of progressively over-constraining extraction criteria with increasingly restrictive negative rules. We find that autonomous self-modification without score feedback tends to degrade extraction quality and become trapped in local minima. The hidden-score constraint, intended to prevent metric gaming, proved sufficient to prevent improvement.

**Keywords:** self-healing agents, recursive self-modification, autonomous AI, prompt evolution, scientific claim extraction, negative results

---

## 1. Introduction

Large language models (LLMs) have demonstrated remarkable capacity for few-shot learning and instruction following, leading to active research into whether LLM-powered agents can autonomously improve their own performance through iterative self-modification. Prior work on self-healing agents has largely focused on code repair scenarios where the agent receives explicit error signals (test failures, compiler errors) or on prompt optimization where reward signals are provided through automated scoring.

This study addresses a more constrained and novel setting: can an autonomous agent improve an extraction pipeline through architectural self-modification when it receives **no score feedback**, **no ground truth access**, and **no evaluation metric visibility** — only the raw output of its prior iteration? This setting is motivated by practical deployment scenarios where evaluation infrastructure is costly, where ground truth is unavailable, or where the agent must reason about quality from first principles rather than optimizing against a known signal.

The research question is: *Can an autonomous agent iteratively improve the precision and recall of a naive scientific claim extractor by modifying its own extraction code and prompt layer, operating only on the signal of its prior outputs against a fixed corpus?*

We pre-registered our study design before any harness code existed or any iteration ran, committing the hypothesis, metrics, evaluation rubric, and execution constraints to version control. This pre-registration prevents post-hoc design drift and ensures the study's findings are defensible regardless of direction (`experiments/study_001/pre-registration.md`, lines 29–44).

---

## 2. Methods

### 2.1. Corpus

The fixed target corpus consists of 200 neuroscience abstracts sourced from NeuroSynth version 0.7 (`idris-neuro-extract/corpus/corpus_manifest.md`, line 1). Abstracts were filtered by the following criteria (lines 11–14):
- Minimum 150 words in length
- Contains results/conclusion language
- English language only
- Human subjects only (animal model papers excluded)

Each abstract was annotated with ground truth claims using a SciFact-derived claim definition (`idris-neuro-extract/corpus/annotation_prompt.md`, lines 1–8): a scientific claim is a declarative sentence that (1) asserts a specific, testable finding, (2) is explicitly stated in the text, (3) is supported by the abstract's own reported results, and (4) is atomic (one assertion per claim).

A quality review of 30 abstracts from the ground truth found zero background statements incorrectly extracted, zero compound claims that should have been split, zero duplicates, and zero missed findings within the non-empty set (`idris-neuro-extract/corpus/review_notes.md`, lines 11–17). However, 20 abstracts (10% of corpus) had zero annotated claims due to extraction-layer failures during corpus construction — all 11 sampled were confirmed as genuine misses containing extractable findings (lines 27–42). The corpus was frozen and does not change across iterations (`experiments/study_001/pre-registration.md`, lines 58–69).

### 2.2. Baseline Extractor

The baseline (iteration 0) extractor is a single-pass LLM call with a minimal system prompt (`idris-neuro-extract/experiments/study_001/iterations/iteration_00_playground/prompts/system_prompt.md`, lines 1–11):

> "You are a scientific claim extractor. Given a neuroscience abstract, extract all scientific claims the abstract explicitly makes."

The prompt defines a claim as "a declarative sentence asserting a specific, testable finding" and instructs the model to exclude background, prior work, and methodological descriptions. No few-shot examples are provided. The extraction code (`idris-neuro-extract/experiments/study_001/iterations/iteration_00_playground/playground/extractor.py`, lines 1–21) reads the system prompt and optional examples file, concatenates them, and issues a single synchronous completion call via a local llama.cpp provider.

### 2.3. Evaluation

Evaluation uses fuzzy string matching via `rapidfuzz.fuzz.token_sort_ratio` with a threshold of 80 (`idris-neuro-extract/protected/scorer.py`, lines 1–37). For each abstract, predicted claims are matched to ground truth claims using a greedy best-match strategy: each predicted claim is scored against all unmatched ground truth claims, and the best match above threshold counts as a true positive. Macro-averaged precision, recall, and F1 are computed across all 200 abstracts (lines 40–81). The scorer is in the protected zone — the agent cannot read or modify it (`experiments/study_001/pre-registration.md`, lines 142–165).

### 2.4. Agent and Self-Modification Loop

The modifier agent is Qwen 3.6 27B MTP 6-bit GGUF, running locally via llama.cpp server on an RTX 5090 32GB (`experiments/study_001/pre-registration.md`, lines 250–261). The 262k context window accommodates full episodic memory without truncation at N=20.

The agent operates with **episodic memory**: an append-only record of prior iterations containing observation, hypothesis, action, and expectation fields (`experiments/study_001/pre-registration.md`, lines 168–198). The expectation field commits a falsifiable prediction before the next iteration's results are seen.

The agent has access to a **playground** containing mutable Python files and prompt files (`idris-neuro-extract/playground/`, `idris-neuro-extract/prompts/`). The only constraint is the immutable entry point contract: `async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult` (`idris-neuro-extract/protected/interface.py`, lines 1–6; `experiments/study_001/pre-registration.md`, lines 120–138).

Critically, the agent receives **no scores, no ground truth, and no evaluation metrics** (`experiments/study_001/pre-registration.md`, lines 160–165). The only signal is the prior iteration's per-abstract extraction output (predicted claim lists). The agent's objective, as stated in the harness invocation prompt, is: "Improve the precision and recall of the scientific claim extractor by modifying its Python code and/or prompt files. You receive the prior iteration's extraction output. You cannot see scores, ground truth, or evaluation metrics" (`idris-neuro-extract/protected/harness/agent_caller.py`, lines 19–24).

A **3-attempt repair loop** handles validation failures (`experiments/study_001/pre-registration.md`, lines 202–220). If the agent's code fails to produce a valid `extract` function after 3 attempts, the playground is rolled back and the iteration is skipped. Each corpus run has a 14,400-second hard timeout (lines 227–246).

### 2.5. Pre-Registration and Early Termination

The study design was pre-registered at commit SHA `7f1acdd258178cd2af19dc1aa770a0712bdc93f9` on 2026-05-29 (`experiments/study_001/pre-registration-proof.txt`, lines 1–2). The original design specified N=20 unconditional iterations (`experiments/study_001/pre-registration.md`, lines 265–272). The run was terminated at iteration 13 based on three observed failure phases and evidence of convergence to a local minimum, as documented in the amendment section (lines 389–452).

---

## 3. Results

### 3.1. F1 Trajectory

Table 1 summarizes per-iteration performance across all 14 iterations. The baseline F1 of 0.467 at iteration 0 declined monotonically through iteration 3 to 0.268, collapsed to zero in iterations 4–6 and 8, and recovered to a floor of 0.13–0.18 in iterations 7–13. The final iteration (13) achieved F1=0.142, 69.6% below baseline.

| Iteration | Macro-P | Macro-R | Macro-F1 | Avg Claims | TP | FP | FN | Scanned |
|-----------|---------|---------|----------|------------|-----|------|------|---------|
| 0 | 0.513 | 0.430 | **0.467** | 4.52 | 464 | 439 | 641 | Yes |
| 1 | 0.418 | 0.347 | 0.379 | 4.45 | 368 | 522 | 737 | Yes |
| 2 | 0.301 | 0.235 | 0.264 | 4.19 | 247 | 591 | 858 | Yes |
| 3 | 0.323 | 0.229 | 0.268 | 3.88 | 252 | 523 | 853 | Yes |
| 4 | — | — | — | — | — | — | — | No |
| 5 | — | — | — | — | — | — | — | No |
| 6 | — | — | — | — | — | — | — | No |
| 7 | 0.230 | 0.128 | 0.164 | 2.74 | 130 | 418 | 975 | Yes |
| 8 | — | — | — | — | — | — | — | No |
| 9 | 0.244 | 0.142 | 0.179 | 2.77 | 150 | 404 | 955 | Yes |
| 10 | 0.251 | 0.144 | **0.183** | 2.69 | 143 | 394 | 962 | Yes |
| 11 | 0.223 | 0.113 | 0.150 | 2.33 | 116 | 349 | 989 | Yes |
| 12 | 0.200 | 0.097 | 0.130 | 2.24 | 101 | 346 | 1004 | Yes |
| 13 | 0.221 | 0.104 | 0.142 | 2.28 | 104 | 351 | 1001 | Yes |

*Table 1: Per-iteration performance metrics. Data source: `idris-neuro-extract/experiments/study_001/metrics.jsonl`, lines 1–14. Dashes indicate anomalous iterations with no corpus scan.*

Of the 14 iterations, 10 produced extraction output (scanned=True) and 4 were anomalous. The peak F1 of the recovery phase was 0.183 at iteration 10, which remains 60.8% below baseline.

### 3.2. Three Phases

The trajectory decomposes into three distinct phases:

**Phase 1 — Initial Degradation (Iterations 1–3).** The agent's first edits introduced few-shot examples and stricter extraction criteria. F1 dropped from 0.467 to 0.264 — a 43.5% decline in three iterations. The agent observed that the baseline "occasionally fails on complex abstracts" and "includes background context, methodological details, or broad speculative interpretations" (`idris-neuro-extract/experiments/study_001/episodes.jsonl`, line 1). Its response was to add three few-shot examples and constrain the definition of valid claims. However, the agent could not distinguish between genuinely low-quality baseline output and output that was acceptable under the ground truth's claim definition. The over-constraining reduced recall (0.430 → 0.229) more than it improved precision (0.513 → 0.323).

**Phase 2 — Total Collapse (Iterations 4–6, 8).** Four iterations produced zero extraction output. Two were caused by the agent replacing prompt files with empty content (`idris-neuro-extract/experiments/study_001/anomalies.jsonl`, lines 3–4: `empty_file_replacement` anomalies at iterations 4 and 5). Iteration 6 and iteration 8 produced malformed (empty) JSON responses from the agent itself (lines 5–6: `agent_response_malformed`). These iterations were rolled back and skipped. Notably, the agent's own reasoning degraded alongside its output — by iteration 6, the agent was unable to produce a valid JSON response at all.

**Phase 3 — Partial Recovery and Floor (Iterations 7–13).** The agent recovered to a stable but severely degraded performance band of F1 0.13–0.18. Despite 7 iterations of continued self-modification, the agent could not exceed the iteration 10 peak of 0.183. The trajectory from iteration 9 onward shows minor oscillation without improvement signal. Average claims per abstract declined from 2.77 to 2.28, indicating the agent continued to over-constrain extraction criteria even during this phase. True positives declined from 150 to 104 over iterations 10–13, while false positives remained relatively stable (394 to 351), suggesting the agent was filtering valid claims without correspondingly reducing invalid ones.

### 3.3. Prompt Evolution Analysis

The agent never modified the Python extractor code across all 14 iterations. The final extractor code at iteration 13 (`idris-neuro-extract/playground/extractor.py`) is identical to the baseline (`idris-neuro-extract/experiments/study_001/iterations/iteration_00_playground/playground/extractor.py`). All agent activity was directed at `prompts/system_prompt.md` and `prompts/examples.md`.

The system prompt grew from 11 lines at baseline to a dense wall of negative constraints by iteration 13. The final prompt (`idris-neuro-extract/prompts/system_prompt.md`) contains the following categories of constraints not present at baseline:

1. **Functional interpretation bans** (27 banned phrases including "plays a role", "subserves", "mediates", "constitute a network for", "work in concert for")
2. **Negative neural claim bans** ("no activation", "does not show activation", "does not depend on", "no significant difference")
3. **Behavioral correlation bans** (claims correlating neural activation with reaction time, pain intensity, accuracy)
4. **Meta-conclusion bans** ("is not due to", "cannot be attributed to", "persists when controlled")
5. **Methodological observation bans** (subject percentages, parameter correlations, standard deviations)
6. **Interpretive clause bans** ("suggesting", "indicating", "consistent with", "may reflect")
7. **Synthesis directive** (group anatomical regions by condition)

The examples file grew from zero examples at baseline to 11 few-shot examples by iteration 13.

### 3.4. Contradictory Few-Shot Examples

A critical finding is that the agent's few-shot examples increasingly contradicted its own system prompt rules. At iteration 13:

- Example 2 includes "no activation of the visual cortex" (a negative neural claim) and "is not due to plasticity" (a meta-conclusion) — both explicitly banned in the system prompt (`idris-neuro-extract/prompts/examples.md`, lines 18–20)
- Example 3 includes "constitute a network for orthographic processing" and "work in concert for phonological processing" — both explicitly banned phrases (`idris-neuro-extract/prompts/examples.md`, lines 24–26)

The agent recognized this contradiction at iteration 12, noting: "Example 2 in examples.md directly contradicts system rules by including 'no activation of the visual cortex' (negative neural claim) and 'is not due to plasticity' (meta-conclusion), confusing the model" (`idris-neuro-extract/experiments/study_001/episodes.jsonl`, line 8). At iteration 13, it attempted to fix Example 3 but the contradiction persisted in Example 2. This pattern illustrates how the agent, operating without score feedback, could detect contradictions in principle but could not determine which examples were actually harming performance versus helping it.

### 3.5. Resource Consumption

Token consumption grew monotonically across iterations as the agent accumulated episodic memory and expanded its prompts. Agent-side token usage ranged from 87,196 to 102,469 total tokens per invocation (`idris-neuro-extract/experiments/study_001/metrics.jsonl`, lines 1–14, field `agent_total_tokens`). Corpus-side average tokens per abstract grew from 3,007 at baseline to 5,858 at iteration 13 (field `corpus_avg_tokens_per_abstract`), a 94.8% increase driven by the expanding system prompt and examples. Scan duration per iteration ranged from 3,849 seconds (64 minutes) at baseline to 4,875 seconds (81 minutes) at iteration 12.

---

## 4. Discussion

### 4.1. The Over-Constraint Trap

The agent's behavior reveals a systematic tendency toward over-constraining extraction criteria. Starting from the observation that the baseline occasionally included non-empirical content, the agent adopted a strategy of adding progressively more negative rules: ban interpretive language, ban functional roles, ban negative claims, ban behavioral correlations, ban methodological observations. Each round of constraint addition reduced the set of accepted claims, lowering recall faster than it improved precision.

This trap is structurally inevitable in a hidden-score setting. The agent cannot distinguish between: (a) a claim that is correctly extracted but happens to contain interpretive language (false positive under the agent's internal rubric, but a true positive under the ground truth), and (b) a claim that is genuinely spurious. Without score feedback to signal that recall has dropped too far, the agent optimizes for an internal quality standard that is stricter than the evaluation rubric requires.

### 4.2. Few-Shot Example Contradictions

The agent's difficulty maintaining consistency between its system rules and its few-shot examples represents a second structural failure mode. At iteration 12, the agent identified that Example 2 contradicted system rules (`idris-neuro-extract/experiments/study_001/episodes.jsonl`, line 8). At iteration 13, it identified the same problem with Example 3 (`idris-neuro-extract/experiments/study_001/episodes.jsonl`, line 9). Despite recognizing these contradictions across two consecutive iterations, the agent could not resolve them completely — the fixed examples still contained banned phrases.

This is consistent with the known finding that LLMs learn from few-shot examples more strongly than from negative constraints in system prompts. The agent effectively created a situation where its examples taught one behavior while its rules taught another, and the examples won. In a score-feedback setting, the agent would observe the resulting performance degradation and know to revise the examples. Without scores, it can only reason about contradictions in principle, which proved insufficient.

### 4.3. Code vs. Prompt Modification

The agent never modified the Python code across all 14 iterations. Despite having full access to the playground's Python files (`playground/extractor.py`, `playground/pipeline.py`, `playground/utils.py`), every edit was directed at `prompts/system_prompt.md` and `prompts/examples.md`. This suggests that prompt modification is the path of least resistance for LLM agents tasked with improving extraction pipelines — it requires less structural reasoning about code, fewer dependencies, and lower risk of breaking the interface contract.

This is not necessarily suboptimal for this particular task, as the bottleneck is likely the LLM's extraction quality rather than pipeline architecture. However, it means the study observed only prompt-space self-modification, not the broader architectural evolution that was permitted by the design.

### 4.4. Recovery Without Escaping the Minimum

The agent's recovery from total collapse (Phase 2) to the F1 0.13–0.18 floor (Phase 3) demonstrates that episodic memory enabled basic failure recovery — the agent could detect that its output was broken and roll back to a working state. However, it could not guide itself back to baseline performance or beyond. The 7 iterations of Phase 3 produced only minor oscillation around the floor, with true positives declining from 150 to 104 (iterations 10 to 13).

This suggests that the hidden-score constraint created an information asymmetry: enough signal to recover from catastrophic failure (the output was obviously empty), but insufficient signal to navigate the fine-grained tradeoff between precision and recall. The agent could see that it was extracting too many or too few claims, but could not determine whether its internal quality standard was too strict or too lenient relative to the ground truth.

### 4.5. Implications for Self-Healing Research

Our findings suggest that self-healing agents operating without score feedback are likely to degrade rather than improve extraction quality. The structural reasons are clear:

1. **No gradient signal:** Without scores, the agent cannot determine whether a change improved or harmed performance. It must rely on qualitative reasoning about output characteristics, which is insufficient for navigating precision-recall tradeoffs.

2. **Over-constraint bias:** LLM agents have a natural tendency toward adding restrictions when tasked with improving quality. Without a score to signal "you've gone too far," this tendency is unbounded.

3. **Example-rubric contradiction:** Few-shot examples and system prompt rules can diverge, and the agent cannot determine which is dominant without score feedback.

These findings do not rule out self-healing as a concept. They suggest that the next generation of self-healing systems will need either: (a) score feedback (even approximate or noisy), (b) a calibration mechanism that lets the agent compare its internal quality standard against external criteria, or (c) a dual-agent architecture where one agent proposes changes and another evaluates them against held-out criteria.

---

## 5. Limitations

1. **Single run (n=1).** Results are from a single execution with a single model configuration. Model nondeterminism means the trajectory may differ under repetition. However, the structural failures (over-constraint, example contradictions) are likely reproducible as they stem from the information asymmetry inherent in the hidden-score design.

2. **Model capacity.** Qwen 3.6 27B at 6-bit quantization may not have sufficient reasoning capacity for this task. A more capable model might navigate the precision-recall tradeoff differently. The study isolates the self-modification loop as the variable under study, not model capacity (`experiments/study_001/pre-registration.md`, lines 250–261).

3. **Corpus quality.** 20 abstracts (10%) have zero ground truth claims due to extraction failures during corpus construction (`idris-neuro-extract/corpus/review_notes.md`, lines 22–42). This creates a systematic recall ceiling that applies equally to all iterations.

4. **Solo evaluation.** The qualitative rubric was applied by a single rater (`experiments/study_001/pre-registration.md`, lines 278–296). Multi-rater evaluation is planned contingent on follow-up studies.

5. **Early termination.** The run was terminated at iteration 13 rather than the planned 20. The rationale for termination — convergence to a stable floor with no improvement signal over 5+ iterations — is documented in the amendment (`experiments/study_001/pre-registration.md`, lines 389–452). The remaining 7 iterations would likely have replicated the oscillatory behavior observed in Phase 3.

---

## 6. Conclusion

This pre-registered study demonstrates that autonomous architectural self-modification without score feedback tends to degrade extraction quality and become trapped in local minima. Over 14 iterations, the agent reduced macro-F1 from 0.467 to 0.142 — a 69.6% decline. The agent's strategy of progressively over-constraining extraction criteria, combined with contradictory few-shot examples, produced a trajectory of initial degradation, total collapse, and partial recovery to a degraded floor.

The hidden-score constraint, intended to prevent metric gaming, proved sufficient to prevent improvement. This negative result is informative: it establishes that self-healing agents need at least approximate score feedback to navigate the precision-recall tradeoff, and that qualitative reasoning about output characteristics is insufficient for iterative quality improvement.

Future work should investigate whether score feedback, even when approximate or delayed, enables self-improving trajectories, and whether architectural constraints (e.g., bounding prompt growth, enforcing example-rubric consistency checks) can mitigate the over-constraint bias observed here.

---

## References

1. `experiments/study_001/pre-registration.md` — Pre-registered study design, decisions 002-A through 002-K, and early termination amendment. Organization: Idris Applied AI Research, May 2026.

2. `experiments/study_001/pre-registration-proof.txt` — Pre-registration proof: commit SHA and locked date.

3. `idris-neuro-extract/experiments/study_001/metrics.jsonl` — Per-iteration programmatic metrics (14 iterations). Precision, recall, F1, token counts, scan durations.

4. `idris-neuro-extract/experiments/study_001/episodes.jsonl` — Agent episodic memory: observation, hypothesis, action, expectation per iteration (9 episodes).

5. `idris-neuro-extract/experiments/study_001/anomalies.jsonl` — Structured anomaly events: 6 anomalies across iterations -1, 4, 5, 6, 8.

6. `idris-neuro-extract/experiments/study_001/agent-rationale.jsonl` — Agent free-form rationale per iteration (9 entries).

7. `idris-neuro-extract/corpus/corpus_manifest.md` — Corpus specification: source, SHA-256, filter criteria, 200 abstract IDs.

8. `idris-neuro-extract/corpus/annotation_prompt.md` — Ground truth claim annotation definition and prompt.

9. `idris-neuro-extract/corpus/review_notes.md` — Ground truth quality review: 30-abstract sample assessment and empty-abstract analysis.

10. `idris-neuro-extract/protected/scorer.py` — Evaluation implementation: fuzzy matching scorer with macro-averaged metrics.

11. `idris-neuro-extract/protected/interface.py` — Immutable entry point contract.

12. `idris-neuro-extract/protected/harness/agent_caller.py` — Agent invocation protocol, response schema, and prompt construction.

13. `idris-neuro-extract/experiments/study_001/iterations/iteration_00_playground/` — Baseline iteration 0 playground snapshot (extractor code, system prompt, empty examples).

14. `idris-neuro-extract/prompts/system_prompt.md` — Final system prompt at iteration 13 (git commit `aaa0cc8`).

15. `idris-neuro-extract/prompts/examples.md` — Final few-shot examples at iteration 13 (git commit `aaa0cc8`).

16. `idris-neuro-extract/playground/extractor.py` — Final extractor code at iteration 13 (git commit `aaa0cc8`; identical to baseline).
