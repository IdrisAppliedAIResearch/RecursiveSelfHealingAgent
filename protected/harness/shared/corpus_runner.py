import asyncio
import json
import time
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


class _CountingProviderProxy:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens_per_second_sum = 0.0
        self.call_count = 0

    def complete(self, system_prompt: str, user_message: str) -> str:
        content, usage = self._wrapped.complete_with_usage(
            system_prompt, user_message
        )
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens_per_second_sum += usage.tokens_per_second
        self.call_count += 1
        return content

    def get_usage(self) -> CorpusTokenUsage:
        attempted = self.call_count if self.call_count > 0 else 1
        total = self.total_prompt_tokens + self.total_completion_tokens
        return CorpusTokenUsage(
            total_prompt_tokens=self.total_prompt_tokens,
            total_completion_tokens=self.total_completion_tokens,
            avg_tokens_per_abstract=total / attempted,
            avg_tokens_per_second=(
                self.total_tokens_per_second_sum / attempted
            ),
        )


async def run_corpus(study_id: str, abstract_files: list[Path] = None) -> CorpusRunResult:
    reload_playground()

    from playground.extractor import extract as extract_fn

    if abstract_files is None:
        abstracts_dir = PROJECT_ROOT / "corpus" / "abstracts"
        abstract_files = sorted(abstracts_dir.glob("*.json"))

    import playground.extractor as pg_extractor_mod

    # Inject shared analyzer if available, so corpus runs use the same model
    from protected.harness.study_002.study_runner import _analyzer_instance

    if _analyzer_instance is not None:
        old_provider = pg_extractor_mod._provider
        pg_extractor_mod._provider = _analyzer_instance
        proxy = None
    else:
        playground_mod = __import__("playground.extractor", fromlist=["extract"])
        import inspect
        old_provider = None
        for name, obj in inspect.getmembers(playground_mod):
            if hasattr(obj, "_provider") and hasattr(obj, "complete"):
                old_provider = obj._provider
                break
        if old_provider is None:
            old_provider = getattr(pg_extractor_mod, "_provider", None)

        proxy = None
        if old_provider is not None:
            proxy = _CountingProviderProxy(old_provider)
            pg_extractor_mod._provider = proxy

    results: list[ExtractionResult] = []
    failures: list[CorpusAbstractFailure] = []
    abstract_texts: dict[str, str] = {}

    print(f"  Corpus: running {len(abstract_files)} abstracts...")
    start = time.monotonic()

    for idx, af in enumerate(abstract_files, 1):
        abstract_id = af.stem
        abstract_data = json.loads(af.read_text(encoding="utf-8", errors="replace"))
        abstract_text = abstract_data.get("abstract", abstract_data.get("text", ""))
        abstract_texts[abstract_id] = abstract_text
        try:
            result = await extract_fn(abstract_id, abstract_text)
            results.append(result)
            elapsed = time.monotonic() - start
            print(f"  Corpus: {idx}/{len(abstract_files)} done (abstract {abstract_id}) ({elapsed:.0f}s)")
        except Exception as e:
            log_anomaly(
                study_id, -1,
                "corpus_abstract_failure",
                {"abstract_id": abstract_id, "error": str(e)},
            )
            failures.append(CorpusAbstractFailure(abstract_id=abstract_id, error=str(e)))

    duration = time.monotonic() - start

    if _analyzer_instance is not None:
        pg_extractor_mod._provider = old_provider
        token_usage = CorpusTokenUsage(0, 0, 0.0, 0.0)
    elif proxy:
        token_usage = proxy.get_usage()
        pg_extractor_mod._provider = old_provider
    else:
        token_usage = CorpusTokenUsage(0, 0, 0.0, 0.0)

    return CorpusRunResult(
        results=results,
        failures=failures,
        duration_seconds=duration,
        corpus_token_usage=token_usage,
        abstract_texts=abstract_texts,
    )
