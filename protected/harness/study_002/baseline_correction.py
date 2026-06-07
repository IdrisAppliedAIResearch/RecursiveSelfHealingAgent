CAPABILITY_FRAMING = """YOUR TOOLS

You have four types of files you can change. Here is what each one 
does and how changing it affects the extraction system:

prompts/system_prompt.md
  This is the instruction the model reads before seeing each abstract.
  Changing it changes how the model interprets the extraction task —
  what it looks for, what it excludes, and how it formats its output.

prompts/examples.md
  These are example abstracts with correct extractions shown alongside.
  Changing them teaches the model by demonstration. An empty file means
  no examples are shown.

playground/extractor.py
  This is the Python function that runs for every abstract. It is called
  as: result = await extract(abstract_id, abstract_text)
  Changing it changes the architecture of extraction — you can add
  preprocessing that runs before the model call, post-processing that
  filters results after, or multiple model calls in sequence.

playground/ (new .py files you create)
  You can create any Python file in playground/ and import it from
  extractor.py. This lets you build modular components — a preprocessor,
  a validator, a claim filter — as separate modules.

WHAT YOU CANNOT CHANGE
The evaluation system, the corpus, the ground truth, the harness, and
the scoring infrastructure are off limits. You can only change what is
listed above."""

WORKED_EXAMPLE_A = """EXAMPLE A — Changing the system prompt

Situation: The extractor is including methodology descriptions as claims.
Abstracts that describe fMRI protocols are producing claims like
"The study used a 3T scanner with TR=2000ms" which are not findings.

Change: Add an explicit exclusion to the system prompt.

Edit instruction:
{
  "file_path": "prompts/system_prompt.md",
  "operation": "replace_string",
  "old_string": "Do not include background statements, prior work references,
or methodological descriptions.",
  "new_string": "Do not include background statements, prior work references,
methodological descriptions, scanner parameters, participant counts,
or any statement that describes how the study was conducted rather than
what it found.",
  "new_content": null
}"""

WORKED_EXAMPLE_B = """EXAMPLE B — Adding a preprocessing step

Situation: The model is attending to methodology sentences when extracting
claims. The routing signal shows low scores (below 0.4) on abstracts with
long methodology sections. You want to filter the abstract to results-only
content before passing it to the extraction model.

Change: Create a preprocessor module and rewire extractor.py to use it.

Step 1 — Create playground/preprocessor.py:
{
  "file_path": "playground/preprocessor.py",
  "operation": "create_file",
  "old_string": null,
  "new_string": null,
  "new_content": "import re\\n\\nRESULTS_PATTERNS = [\\n    r'\\\\b(showed?|demonstrated?|revealed?|found|observed)\\\\b',\\n    r'\\\\b(significantly|greater than|less than|compared to)\\\\b',\\n    r'\\\\b(activation|deactivation|correlation|increase|decrease)\\\\b',\\n    r'\\\\b(p\\\\s*[<>=]\\\\s*0\\\\.\\\\d+|t\\\\s*\\\\(\\\\d+\\\\))\\\\b',\\n]\\n\\nMETHODS_PATTERNS = [\\n    r'\\\\b(participants?|subjects?|were recruited|were scanned)\\\\b',\\n    r'\\\\b(fMRI|scanner|TR|voxel|mm|tesla)\\\\b',\\n    r'\\\\b(we used|study examined|designed to)\\\\b',\\n]\\n\\ndef score_sentence(sentence: str) -> str:\\n    results_hits = sum(1 for p in RESULTS_PATTERNS if re.search(p, sentence, re.IGNORECASE))\\n    methods_hits = sum(1 for p in METHODS_PATTERNS if re.search(p, sentence, re.IGNORECASE))\\n    if results_hits >= methods_hits and results_hits > 0:\\n        return 'RESULTS'\\n    elif methods_hits > results_hits:\\n        return 'METHODS'\\n    return 'BACKGROUND'\\n\\ndef filter_to_results(abstract_text: str) -> str:\\n    sentences = re.split(r'(?<=[.!?])\\\\s+', abstract_text)\\n    results = [s for s in sentences if score_sentence(s) == 'RESULTS']\\n    return ' '.join(results) if results else abstract_text\\n"
}

Step 2 — Rewire extractor.py to use the preprocessor:
{
  "file_path": "playground/extractor.py",
  "operation": "replace_string",
  "old_string": "async def extract(abstract_id: str, abstract_text: str) -> ExtractionResult:",
  "new_string": "from playground.preprocessor import filter_to_results\\n\\nasync def extract(abstract_id: str, abstract_text: str) -> ExtractionResult:\\n    abstract_text = filter_to_results(abstract_text)",
  "new_content": null
}"""

ROUTING_SIGNAL_EXPLANATION = """YOUR ROUTING SIGNAL

When the model reads an abstract to extract claims, it distributes attention
across the abstract's sentences. Some of that attention goes to sentences
reporting results — findings, activations, correlations. Some goes to
methodology sentences — how the study was run, what equipment was used.
Some goes to background — prior work, objectives, context.

The routing score measures what fraction of the model's attention goes to
results sentences when it is deciding what to extract. A score of 1.0 means
all attention is on results sentences. A score of 0.0 means none is.

A higher routing score does not guarantee better extractions. But a model
that attends primarily to results sentences when extracting claims is more
likely to extract actual findings than one attending to methodology or
background. The signal tells you about the model's processing, not its output.

You see this signal in two forms:
- ROUTING HISTORY: your routing scores across all prior iterations
- ROUTING DELTA: what your last modification did to routing scores

If your modifications are not moving routing scores, that is information.
It may mean the prompt is not the right lever for changing where the model
attends. It may be worth considering architectural changes instead."""


def compose_baseline_correction() -> str:
    parts = [
        CAPABILITY_FRAMING,
        "",
        WORKED_EXAMPLE_A,
        "",
        WORKED_EXAMPLE_B,
        "",
        ROUTING_SIGNAL_EXPLANATION,
    ]
    return "\n\n".join(parts)
