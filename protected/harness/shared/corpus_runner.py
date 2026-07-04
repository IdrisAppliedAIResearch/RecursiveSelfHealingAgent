import asyncio
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from protected.harness.shared.anomaly_logger import log_anomaly
from protected.harness.shared.interface_validator import reload_playground
from protected.schema import ExtractionResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class CorpusAbstractFailure:
    abstract_id: str
    error: str


@dataclass
class CorpusTokenUsage:
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_tokens_per_abstract: float
    avg_tokens_per_second: float


@dataclass
class CorpusRunResult:
    results: list[ExtractionResult]
    failures: list[CorpusAbstractFailure]
    duration_seconds: float
    corpus_token_usage: CorpusTokenUsage
    abstract_texts: dict[str, str] | None = None
    zero_claim_count: int = 0
    n_extracted: int = 0


@contextmanager
def provider_injected(provider):
    """A004-9: set the extractor's module-global provider for the duration of a
    block and restore the prior value afterward. Shared by the corpus run and the
    interface smoke test so that any path invoking extract() has a live provider,
    even immediately after reload_playground() reset the module to _provider=None.

    If provider is None, the existing module provider is left untouched (used by
    backends that establish the provider through another mechanism)."""
    import playground.extractor as pg
    if provider is None:
        yield
        return
    old = getattr(pg, "_provider", None)
    pg._provider = provider
    try:
        yield
    finally:
        pg._provider = old


async def run_corpus(study_id: str, abstract_files: list[Path] = None) -> CorpusRunResult:
    reload_playground()

    from playground.extractor import extract as extract_fn

    if abstract_files is None:
        abstracts_dir = PROJECT_ROOT / "corpus" / "abstracts"
        abstract_files = sorted(abstracts_dir.glob("*.json"))

    # Use the shared model instance so the corpus run uses the same weights as the
    # attention pass and the agent calls.
    from protected.harness.shared.analyzer_registry import get_analyzer
    provider = get_analyzer()

    results: list[ExtractionResult] = []
    failures: list[CorpusAbstractFailure] = []
    abstract_texts: dict[str, str] = {}

    print(f"  Corpus: running {len(abstract_files)} abstracts, provider={provider is not None}...", flush=True)
    start = time.monotonic()

    with provider_injected(provider):
        for idx, af in enumerate(abstract_files, 1):
            abstract_id = af.stem
            abstract_data = json.loads(af.read_text(encoding="utf-8", errors="replace"))
            abstract_text = abstract_data.get("abstract", abstract_data.get("text", ""))
            abstract_texts[abstract_id] = abstract_text
            try:
                result = await extract_fn(abstract_id, abstract_text)
                results.append(result)
                elapsed = time.monotonic() - start
                print(f"  Corpus: {idx}/{len(abstract_files)} done ({abstract_id}) "
                      f"[{len(result.claims)} claims] ({elapsed:.0f}s)", flush=True)
            except Exception as e:
                log_anomaly(
                    study_id, -1,
                    "corpus_abstract_failure",
                    {"abstract_id": abstract_id, "error": str(e)},
                )
                failures.append(CorpusAbstractFailure(abstract_id=abstract_id, error=str(e)))

    duration = time.monotonic() - start

    # A004-4: extraction observability — count abstracts that produced zero claims.
    zero_claim_count = sum(1 for r in results if len(r.claims) == 0)

    return CorpusRunResult(
        results=results,
        failures=failures,
        duration_seconds=duration,
        corpus_token_usage=CorpusTokenUsage(0, 0, 0.0, 0.0),
        abstract_texts=abstract_texts,
        zero_claim_count=zero_claim_count,
        n_extracted=len(results),
    )
