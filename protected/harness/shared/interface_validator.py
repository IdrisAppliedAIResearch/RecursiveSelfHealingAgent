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

    try:
        result = await asyncio.wait_for(
            extract_fn("smoke_test_001", _SMOKE_TEST_ABSTRACT),
            timeout=60,
        )
        claim_count = len(result.claims) if hasattr(result, 'claims') else 0
        return ValidationResult(
            valid=True,
            smoke_test_passed=True,
            smoke_test_claim_count=claim_count,
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
