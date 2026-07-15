# Study 003 — Logit-Lens Workspace Probe (Feasibility)

**Idris Applied AI Research** · July 2026
**Status:** Exploratory probe, run once. Not a pre-registered study.
**Verdict:** **Negative — do not pursue workspace grounding via the generation-position logit lens as an intrinsic cost signal.**

---

## What this probe asked

Study 002 tested one intrinsic cost signal — *attention routing fidelity* — and
found it structurally decoupled from extraction quality. Before designing a study
around an alternative, this probe checks the feasibility of a second candidate
drawn from Anthropic's "Verbalizable Representations Form a Global Workspace in
Language Models" (Transformer Circuits, 2026): **workspace grounding**, read
cheaply with the *logit lens*.

Narrow question: **when Qwen 3.6 27B is about to extract claims from a
neuroscience abstract, do result-relevant concepts appear in the logit-lens
readout at middle-band layers?** If yes, workspace grounding could be compared
against attention routing fidelity in a later study. If no, we drop it.

Full spec: [`dev-history/s3-001-logit-lens-workspace-probe.md`](../../dev-history/s3-001-logit-lens-workspace-probe.md).

## Method

`logit_lens_probe.py` runs one forward pass per abstract with
`output_hidden_states=True`, then applies the model's final RMSNorm + unembedding
(`lm_head`) to the residual stream at the **generation-start position** (the last
token, where the model is about to begin its answer), sweeping seven depths
(20–90%). It scores the top-30 tokens at each layer against a result-concept and
a method-concept word list. Run on 3 Study-002 probe abstracts plus two synthetic
controls (results-only, methods-only). Model: Qwen 3.6 27B, 4-bit nf4, 64 layers,
loaded exactly as the Study 002 attention analyzer. GPU: RTX 5090.

Reproduce:

```bash
python experiments/study_003/logit_lens_probe.py
# writes probe_output.txt (human-readable sweep) and probe_output.json (structured)
```

## Result — negative on all three success criteria

Aggregated concept hits across all seven layers (see `probe_output.json`):

| Abstract | result-concept hits (all layers) | method-concept hits |
|---|---|---|
| 10355678 | **0** | 3 |
| 11590114 | **0** | 3 |
| 12727182 | **0** | 3 |
| synthetic results-only | **0** | 1 |
| synthetic methods-only | **0** | 2 |

1. **No result-concepts, at any layer.** Zero result-concept words (region names,
   activation/correlation/directional terms) appear in the top-30 readout at any
   depth for any abstract. The only concept-list matches are the token
   `analysis` — the model's own task-planning word, not the abstract's content.

2. **No content differentiation.** The synthetic results-only and methods-only
   abstracts produce essentially **indistinguishable** readouts; the results-only
   control shows *fewer* concept hits, not more. The readout does not reflect
   content type — failing the probe's directional check.

3. **There is a readable band, but it holds the wrong thing.** The layer
   structure is clean and consistent across all five inputs:
   - **early (20–40%)** — punctuation and high-frequency tokens (noise / input echo);
   - **middle (50–70%)** — the model's *procedure* for doing the task:
     `analysis`, `分析`, `首先`/`first`, `抽取`/`提取`/`extract`, `following`,
     `第一步` ("first step"), `按照` ("according to");
   - **late (80–90%)** — the *output format*: `` ``` ``, `{`, `"`, `json`, `JSON`.

   This matches the workspace paper's noise→concepts→prediction shape, but the
   "concepts" in the middle band are the model's **plan for executing the
   extraction and the JSON it will emit**, not the domain concepts it has loaded
   from the abstract.

## Interpretation

At the generation-start position, the residual stream encodes *what the model is
about to do and in what format* — not *what it read*. That is the honest reading
of the workspace here: the verbalizable content at that position is procedural
and format-bearing. Domain concepts from the abstract, if they are logit-lens
readable at all, would have to be read at positions **over the abstract tokens**,
not at the generation position this probe (per its spec) targeted.

Two incidental observations: the model plans the extraction partly in Chinese
(`分析`/`首先`/`抽取` recur across every input), and `<think>` appears in early-layer
readouts even with `enable_thinking=False` — neither affects the verdict.

## Decision

**Do not pursue** workspace grounding as instantiated here (logit lens at the
generation position, whole-vocabulary concept-word matching) as the Study-004
intrinsic signal. It does not surface result concepts and does not discriminate
content type — the same decoupling failure mode Study 002 documented, reached by a
different route. A future probe *could* revisit the idea by reading the lens over
abstract-token positions rather than the generation position; that is a new spec,
written only if motivated, not a continuation of this one.

## Files

- `logit_lens_probe.py` — the probe (self-contained; reuses the Study-002 4-bit load)
- `probe_output.txt` — full human-readable layer sweep for all five inputs
- `probe_output.json` — structured per-layer top-tokens and concept counts
