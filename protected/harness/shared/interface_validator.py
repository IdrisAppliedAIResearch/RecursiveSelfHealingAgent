import asyncio
import inspect
import sys
from dataclasses import dataclass

from protected.interface import ENTRY_POINT_MODULE, ENTRY_POINT_FUNCTION

_SMOKE_TEST_ABSTRACT = (
    "Bilateral hippocampal activation was significantly increased during "
    "encoding compared to baseline (p < 0.001). Left prefrontal cortex showed "
    "greater activation for novel stimuli than repeated stimuli."
)


@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None
    smoke_test_passed: bool = False
    smoke_test_claim_count: int = 0


def reload_playground() -> None:
    to_remove = [key for key in sys.modules if key.startswith("playground")]
    for key in to_remove:
        del sys.modules[key]


class _BoundedProvider:
    """A004-14: wraps the shared provider to force a small generation budget for the
    smoke test, which only verifies the pipeline runs — not extraction quality —
    and must not risk the 60 s timeout on a full-length generation."""

    def __init__(self, inner, cap: int = 64):
        self._inner = inner
        self._cap = cap

    def complete_with_usage(self, system_prompt, user_message, max_tokens=None, *args, **kwargs):
        capped = min(max_tokens or self._cap, self._cap)
        return self._inner.complete_with_usage(system_prompt, user_message, capped, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _check_extraction_contract(result) -> str | None:
    """A007-1: return an error string if `result` violates the ExtractionResult
    contract the corpus scan depends on (`.claims` is a list of Claim), else None.
    Imported lazily so a broken playground import cannot break this module."""
    from protected.schema import Claim, ExtractionResult

    if not isinstance(result, ExtractionResult):
        return (
            f"{ENTRY_POINT_FUNCTION}() must return an ExtractionResult, got "
            f"{type(result).__name__}. Return "
            f"ExtractionResult(abstract_id=..., claims=[Claim(claim_text=...), ...])."
        )
    if not isinstance(result.claims, list):
        return (
            f"{ENTRY_POINT_FUNCTION}() returned ExtractionResult.claims of type "
            f"{type(result.claims).__name__}; it must be a list of Claim."
        )
    for i, c in enumerate(result.claims):
        if not isinstance(c, Claim):
            return (
                f"{ENTRY_POINT_FUNCTION}() returned a non-Claim element at "
                f"claims[{i}] (type {type(c).__name__}); every element must be a "
                f"Claim(claim_text=...)."
            )
    return None


def _validate_extract_fn(fn) -> ValidationResult:
    if fn is None or not callable(fn):
        return ValidationResult(
            valid=False,
            error=f"{ENTRY_POINT_FUNCTION} not found or not callable",
        )
    if not asyncio.iscoroutinefunction(fn):
        return ValidationResult(
            valid=False,
            error=f"{ENTRY_POINT_FUNCTION} is not an async function",
        )
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    positional_params = [
        p for p in params
        if p.default == inspect.Parameter.empty
           and p.kind in (
               inspect.Parameter.POSITIONAL_ONLY,
               inspect.Parameter.POSITIONAL_OR_KEYWORD,
           )
    ]
    if len(positional_params) < 2:
        return ValidationResult(
            valid=False,
            error=f"{ENTRY_POINT_FUNCTION} requires at least 2 positional parameters, got {len(positional_params)}",
        )
    return ValidationResult(valid=True)


async def validate_interface() -> ValidationResult:
    reload_playground()

    try:
        module = await asyncio.get_running_loop().run_in_executor(
            None, __import__, ENTRY_POINT_MODULE
        )
        mod = module
        for part in ENTRY_POINT_MODULE.split(".")[1:]:
            mod = getattr(mod, part)
    except (ImportError, SyntaxError, Exception) as e:
        return ValidationResult(valid=False, error=str(e))

    extract_fn = getattr(mod, ENTRY_POINT_FUNCTION, None)
    return _validate_extract_fn(extract_fn)


async def run_smoke_test() -> ValidationResult:
    reload_playground()

    try:
        module = await asyncio.get_running_loop().run_in_executor(
            None, __import__, ENTRY_POINT_MODULE
        )
        mod = module
        for part in ENTRY_POINT_MODULE.split(".")[1:]:
            mod = getattr(mod, part)
    except (ImportError, SyntaxError, Exception) as e:
        return ValidationResult(
            valid=False,
            error=f"Import failed in smoke test: {e}",
            smoke_test_passed=False,
        )

    extract_fn = getattr(mod, ENTRY_POINT_FUNCTION, None)
    sig_result = _validate_extract_fn(extract_fn)
    if not sig_result.valid:
        return ValidationResult(
            valid=False,
            error=sig_result.error,
            smoke_test_passed=False,
        )

    # A004-9: inject the shared provider so the extractor has a live provider even
    # though reload_playground() reset it. A004-14: bound the generation budget.
    from protected.harness.shared.corpus_runner import provider_injected
    from protected.harness.shared.analyzer_registry import get_analyzer

    provider = get_analyzer()
    bounded = _BoundedProvider(provider) if provider is not None else None

    try:
        with provider_injected(bounded):
            result = await asyncio.wait_for(
                extract_fn("smoke_test_001", _SMOKE_TEST_ABSTRACT),
                timeout=60,
            )
        # A007-1: enforce the return contract instead of masking a wrong return type
        # as "0 claims, passed". The corpus scan accesses result.claims per abstract
        # (corpus_runner.py:93 / :105); a type/shape violation that only surfaces there
        # crashes all 25 abstracts and gives the agent no feedback. Fail here so the
        # existing smoke-repair loop hands the agent the exact contract error, before
        # the attention pass and full scan.
        contract_error = _check_extraction_contract(result)
        if contract_error is not None:
            return ValidationResult(
                valid=False,
                error=contract_error,
                smoke_test_passed=False,
            )
        return ValidationResult(
            valid=True,
            smoke_test_passed=True,
            smoke_test_claim_count=len(result.claims),
        )
    except asyncio.TimeoutError:
        return ValidationResult(
            valid=False,
            error="Smoke test timed out after 60s",
            smoke_test_passed=False,
        )
    except Exception as e:
        return ValidationResult(
            valid=False,
            error=f"Smoke test invocation failed: {e}",
            smoke_test_passed=False,
        )
