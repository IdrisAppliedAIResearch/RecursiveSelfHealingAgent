 # Sprint 001 — MVP Neuroscience Claim Extractor

**Organization:** Idris Applied AI Research
**Status:** Specced
**Date:** May 2026
**Author:** Muzaffer Ozen
**Depends on:** Nothing. This is the founding sprint.

---

## Context

This sprint builds the minimum viable extraction tool that the self-healing
detection study (Sprint 002) will run against. The extractor is intentionally
naive: a single prompt, one abstract in, a structured list of scientific claims
out. No agents, no orchestration, no tool use. The simplicity is load-bearing —
the research question in Sprint 002 is whether the self-healing loop can lift a
simple extractor, not whether a sophisticated system can be marginally improved.

The corpus this extractor runs against is a fixed set of 200 neuroscience
abstracts from NeuroSynth, annotated with ground truth claims using the
SciFact-derived claim definition. Corpus build and annotation are
**prerequisites** completed before any extractor code runs. They are described
in this spec and treated as Sprint 001 setup tasks, not a separate sprint,
because the extractor has no acceptance bar without ground truth to score
against.

**Critical design commitment:** The prompt layer lives in `.md` data files from
day one, not embedded in Python modules. This is the founding architectural
decision that makes the Sprint 002 allowlist enforceable by path without a
prerequisite refactor. Any future pressure to inline prompts into Python for
convenience must be resisted — the path-isolation property is load-bearing for
the entire research program.

---

## Corpus: NeuroSynth + SciFact-Derived Annotation

### Source

NeuroSynth (Yarkoni et al., 2011) is a large-scale neuroscience meta-analysis
platform built from PubMed abstracts. The raw database archive is publicly
available at `https://github.com/neurosynth/neurosynth-data` and contains
approximately 14,000 abstracts with associated neuroimaging activation metadata.

The fixed corpus for this study is a subset of 200 abstracts drawn from the
NeuroSynth archive, filtered by the criteria below. Once selected, the corpus
is locked: abstract IDs are recorded in `corpus/corpus_manifest.md` with a
SHA-256 of the source archive, and the set is never changed.

### Filtering criteria

- Abstract length ≥ 150 words (excludes methods-only and stub entries)
- Abstract contains at least one RESULTS or CONCLUSION section or equivalent
  language (excludes pure methodology introductions)
- English language only
- Human subjects only (filters out animal model papers where claim structure
  differs substantially)

The filtering pass is run by `scripts/build_corpus.py`, which pulls from the
NeuroSynth archive, applies criteria, and writes 200 qualifying abstracts to
`corpus/abstracts/` as individual JSON files keyed by PubMed ID.

### Claim definition (SciFact-derived)

A **scientific claim** for this study is a declarative sentence that:

1. **Asserts a specific, testable finding** the abstract is making — not
   background, not prior work, not a method description.
2. **Is explicitly stated** — not implied or inferable. The words must be in
   the abstract.
3. **Is supported** by the abstract's own reported results. Hedged background
   statements ("previous studies suggest...") are not claims the abstract is
   making.
4. **Is atomic** — one assertion per claim. Compound findings are split.

This definition maps directly onto the SUPPORTED label class in SciFact
(Wadden et al., EMNLP 2020), which is the annotation standard this study
inherits. Anchoring the definition to a 500+ citation benchmark makes the
ground truth methodology defensible without requiring novel annotation
framework development.

### Ground truth annotation process

1. **Model annotation pass** — `scripts/annotate_corpus.py` runs a prompted
   Qwen pass over all 200 abstracts using the claim definition above. Output
   per abstract: a JSON list of candidate claims. Prompt used for annotation
   is committed to `corpus/annotation_prompt.md` and is immutable after the
   annotation pass runs.
2. **Spot-check review** — The researcher reviews a random sample of 30
   abstracts (15%) to validate annotation consistency and correct systematic
   errors. Corrections are applied directly to `corpus/ground_truth.jsonl`.
   The review process is documented in `corpus/review_notes.md`.
3. **Lock** — `corpus/ground_truth.jsonl` is committed and frozen before any
   extractor code runs. No post-hoc adjustments to ground truth after Sprint
   001 is closed.

Ground truth format (one line per abstract):

```json
{"abstract_id": "12345678", "claims": ["Dorsolateral PFC activity increased during working memory encoding.", "Hippocampal volume negatively correlated with cortisol levels."]}
```

---

## Architecture

```
idris-neuro-extract/
  README.md
  corpus/
    abstracts/                    # 200 JSON files, one per abstract, keyed by PubMed ID
    ground_truth.jsonl            # locked ground truth (one line per abstract)
    corpus_manifest.md            # locked corpus definition: IDs, source SHA, filter criteria
    annotation_prompt.md          # immutable prompt used for ground truth annotation
    review_notes.md               # spot-check review log
  extractor/
    __init__.py
    provider.py                   # LlamaCppProvider: OpenAI-compatible client
    extractor.py                  # main extraction logic: abstract in → claims out
    schema.py                     # Claim, ExtractionResult pydantic models
  prompts/
    system_prompt.md              # mutable — the prompt layer the self-healing loop targets
    examples.md                   # mutable few-shot examples (starts empty)
  evaluation/
    scorer.py                     # precision, recall, F1 against ground truth
    runner.py                     # runs extractor over full corpus, writes results
  scripts/
    build_corpus.py               # NeuroSynth archive pull + filtering + output to corpus/abstracts/
    annotate_corpus.py            # model annotation pass → candidate ground truth
  experiments/
    .gitkeep                      # reserved for Sprint 002 self-healing harness
```

---

## Decisions

---

**DECISION 001-A — Single prompt, no agents, no orchestration.**

The extractor is a single LLM call per abstract. System prompt from
`prompts/system_prompt.md`, user message is the abstract text, response is a
JSON object with a `claims` array. No chaining, no tool use, no multi-step
pipeline. The naivety is intentional — Sprint 002 studies whether this floor
can be raised by recursive prompt refinement. A sophisticated baseline would
obscure the signal.

---

**DECISION 001-B — Prompt layer in `.md` files, loaded at construction time.**

`prompts/system_prompt.md` and `prompts/examples.md` are the mutable prompt
layer. The `Extractor` class loads them in `__init__`, not at module import.
Construction-time loading is what makes each extraction run pick up any edits
applied by the self-healing harness in the same Python process. Import-time
loading would require a process restart between iterations, which breaks the
Sprint 002 loop.

This also means the Sprint 002 allowlist can enforce path-based isolation
immediately — no prerequisite refactor required.

---

**DECISION 001-C — llama.cpp server via OpenAI-compatible endpoint.**

The provider uses the `openai` Python SDK pointed at the llama.cpp server's
`/v1/chat/completions` endpoint. `LLAMA_CPP_BASE_URL` env var sets the host.
`LLAMA_CPP_API_KEY` env var is optional (llama.cpp can run without auth).
Model is declared as `LLAMA_CPP_MODEL_ID` env var, defaulting to
`qwen3-27b-mtp-6bit`.

`response_format={"type": "json_object"}` is passed on every call to
instruct the model to return valid JSON. Schema validation happens at the
harness parse layer — a response that is valid JSON but fails schema
validation is a parse error, not an API error. This is the correct split
given that llama.cpp does not support OpenAI's strict JSON schema enforcement.

---

**DECISION 001-D — Output schema is flat and minimal.**

```python
class Claim(BaseModel):
    claim_text: str

class ExtractionResult(BaseModel):
    abstract_id: str
    claims: list[Claim]
```

No confidence scores, no claim typing, no provenance. The self-healing loop
evolves the prompt; it does not evolve the schema. If a future study needs
richer output structure, that is a new sprint with a new schema, not an
amendment to Sprint 001.

---

**DECISION 001-E — Fuzzy string matching for claim evaluation.**

Exact string matching is too strict — a paraphrase of the same claim scores
as a false positive and a false negative simultaneously, artificially
depressing recall without representing a real extraction failure. The
evaluation uses RapidFuzz token sort ratio with a configurable threshold
(default: 80). A predicted claim matches a ground truth claim if their token
sort ratio exceeds the threshold.

The threshold is a parameter declared in the pre-registration for Sprint 002
and held constant across all 20 iterations. It is not tuned post-hoc.

Embedding-based semantic matching is deferred to a future sprint — it
introduces an additional model dependency and the threshold calibration
problem is harder to defend without empirical evidence that the fuzzy match
threshold is miscalibrated.

---

**DECISION 001-F — Corpus is frozen before extractor code runs.**

Ground truth annotation is a prerequisite, not a concurrent task. The extractor
is not written until `corpus/ground_truth.jsonl` is committed. This prevents
the annotation process from being implicitly shaped by knowledge of what the
extractor produces. The discipline is the same as Sprint 036's pre-registration
commitment: design before build.

---

## Component Specifications

### `extractor/schema.py`

```python
from pydantic import BaseModel

class Claim(BaseModel):
    claim_text: str

class ExtractionResult(BaseModel):
    abstract_id: str
    claims: list[Claim]
```

### `extractor/provider.py`

`LlamaCppProvider` wraps the `openai` SDK:

```python
class LlamaCppProvider:
    def __init__(self):
        self.client = openai.OpenAI(
            base_url=os.environ["LLAMA_CPP_BASE_URL"],
            api_key=os.environ.get("LLAMA_CPP_API_KEY", "no-key"),
        )
        self.model = os.environ.get("LLAMA_CPP_MODEL_ID", "qwen3-27b-mtp-6bit")

    def complete(self, system_prompt: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
```

No retry logic in Sprint 001. A failed call raises and is surfaced to the
caller. The evaluation runner logs failures and continues.

### `extractor/extractor.py`

```python
class Extractor:
    def __init__(self):
        self.provider = LlamaCppProvider()
        prompts_dir = Path(__file__).parent.parent / "prompts"
        system_prompt = (prompts_dir / "system_prompt.md").read_text()
        examples = (prompts_dir / "examples.md").read_text().strip()
        if examples:
            system_prompt = system_prompt + "\n\n" + examples
        self.system_prompt = system_prompt

    def extract(self, abstract_id: str, abstract_text: str) -> ExtractionResult:
        raw = self.provider.complete(self.system_prompt, abstract_text)
        data = json.loads(raw)
        claims = [Claim(claim_text=c) for c in data.get("claims", [])]
        return ExtractionResult(abstract_id=abstract_id, claims=claims)
```

Construction-time loading is the critical property (DECISION 001-B). The
`system_prompt` attribute is set in `__init__` from the `.md` files on disk.
A fresh `Extractor()` constructed after a prompt edit will reflect the edit.

### `evaluation/scorer.py`

```python
def score(
    predicted: list[str],
    ground_truth: list[str],
    threshold: int = 80,
) -> dict:
    """
    Returns {"precision": float, "recall": float, "f1": float,
             "tp": int, "fp": int, "fn": int}
    Matching: RapidFuzz token_sort_ratio >= threshold.
    Each ground truth claim matched at most once (greedy, order-independent).
    """
```

The scorer operates on string lists. `runner.py` feeds it claims extracted
from `ExtractionResult` and claims from `ground_truth.jsonl`.

Corpus-level aggregation: macro-average precision/recall/F1 across all 200
abstracts, plus micro-aggregate TP/FP/FN counts. Both are written to the
results file.

### `evaluation/runner.py`

Runs the extractor over all 200 abstracts, scores each against ground truth,
writes results to `evaluation/results/run_{ISO8601}.json`. Output format:

```json
{
  "run_timestamp": "...",
  "model": "...",
  "match_threshold": 80,
  "macro_precision": 0.0,
  "macro_recall": 0.0,
  "macro_f1": 0.0,
  "micro_tp": 0,
  "micro_fp": 0,
  "micro_fn": 0,
  "per_abstract": [
    {
      "abstract_id": "...",
      "predicted_claims": [...],
      "ground_truth_claims": [...],
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "tp": 0, "fp": 0, "fn": 0
    }
  ]
}
```

This is the file Sprint 002's self-healing harness reads as the
"prior iteration's report." The format is locked; Sprint 002 depends on it.

### `prompts/system_prompt.md` (initial content)

The initial prompt is intentionally plain. The self-healing study's baseline
value is established by this prompt's performance — it should not be
pre-optimized. A reasonable starting point:

```
You are a scientific claim extractor. Given a neuroscience abstract, extract
all scientific claims the abstract explicitly makes.

A scientific claim is a declarative sentence asserting a specific, testable
finding that the abstract supports. Do not include background statements,
prior work references, or methodological descriptions.

Respond with a JSON object in this exact format:
{"claims": ["claim one", "claim two"]}

If no claims are present, return: {"claims": []}
```

### `prompts/examples.md` (initial content)

Empty. The self-healing loop may populate this over iterations.

---

## Prerequisites (Ordered)

These must be completed before extractor implementation begins:

1. Clone or download the NeuroSynth data archive. Record source SHA-256 in
   `corpus/corpus_manifest.md`.
2. Run `scripts/build_corpus.py` to produce 200 filtered abstracts in
   `corpus/abstracts/`.
3. Commit `corpus/annotation_prompt.md` (the prompt used for ground truth
   annotation — must be committed before the annotation pass runs).
4. Run `scripts/annotate_corpus.py` to produce candidate ground truth.
5. Spot-check 30 abstracts. Correct errors. Document in
   `corpus/review_notes.md`.
6. Commit `corpus/ground_truth.jsonl` and `corpus/corpus_manifest.md`.
   These are now frozen.
7. Begin extractor implementation.

---

## Tasks

- Set up repo: `idris-neuro-extract`, MIT license, `README.md`, `.gitignore`,
  `requirements.txt` (openai, pydantic, rapidfuzz, requests)
- Implement `scripts/build_corpus.py`: NeuroSynth archive pull, filtering,
  output 200 abstracts to `corpus/abstracts/`
- Commit `corpus/annotation_prompt.md`
- Implement `scripts/annotate_corpus.py`: Qwen annotation pass over 200
  abstracts → candidate `corpus/ground_truth.jsonl`
- Researcher spot-check: review 30 abstracts, correct, commit final
  `corpus/ground_truth.jsonl` and `corpus/review_notes.md`
- Commit `corpus/corpus_manifest.md` with source SHA and filter criteria
- Implement `extractor/schema.py` with `Claim` and `ExtractionResult`
- Implement `extractor/provider.py` with `LlamaCppProvider`
- Implement `extractor/extractor.py` with construction-time prompt loading
- Write `prompts/system_prompt.md` (initial plain prompt) and empty
  `prompts/examples.md`
- Implement `evaluation/scorer.py` with fuzzy match, per-abstract and
  corpus-level aggregation
- Implement `evaluation/runner.py` with results JSON output
- Run baseline evaluation: full corpus pass, record
  `evaluation/results/run_{ISO8601}.json` as the Sprint 001 baseline artifact
- Verify construction-time loading: write a sentinel string into
  `prompts/system_prompt.md`, construct a fresh `Extractor()`, assert the
  sentinel is in `self.system_prompt` — confirm the property Sprint 002 depends on

---

## Acceptance Criteria

- `corpus/ground_truth.jsonl` committed and frozen (200 entries)
- `corpus/corpus_manifest.md` committed with source SHA and filter criteria
- `extractor/`, `evaluation/`, and `prompts/` directories exist with all
  specified modules
- `LlamaCppProvider` successfully calls the llama.cpp server and returns a
  valid JSON string
- `Extractor.extract()` returns a schema-valid `ExtractionResult` for a sample
  abstract
- `scorer.py` correctly computes precision, recall, F1 for a manually verified
  test case
- `runner.py` produces a well-formed results JSON over the full 200-abstract
  corpus
- Construction-time loading is verified: sentinel written to
  `prompts/system_prompt.md` is present in a freshly constructed `Extractor`
  instance's `system_prompt` attribute
- Baseline evaluation results file exists in `evaluation/results/`
- Baseline macro-F1 is recorded (value is not a pass/fail criterion — it is
  the Sprint 002 iteration-0 anchor)
- `experiments/.gitkeep` exists (Sprint 002 harness reserved)

---

## Scope Boundaries

Explicitly out of scope:

- Self-healing harness (Sprint 002)
- Multi-agent or orchestrated extraction
- Embedding-based claim matching (deferred)
- Claim typing or confidence scoring
- Web UI or API layer
- Docker containerization
- Any modification to `corpus/ground_truth.jsonl` after it is committed