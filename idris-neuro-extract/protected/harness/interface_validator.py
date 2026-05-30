import asyncio
import inspect
import sys
from dataclasses import dataclass

from protected.interface import ENTRY_POINT_MODULE, ENTRY_POINT_FUNCTION


@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None


def reload_playground() -> None:
    to_remove = [key for key in sys.modules if key.startswith("playground")]
    for key in to_remove:
        del sys.modules[key]


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
    if extract_fn is None or not callable(extract_fn):
        return ValidationResult(
            valid=False,
            error=f"{ENTRY_POINT_FUNCTION} not found or not callable",
        )

    if not asyncio.iscoroutinefunction(extract_fn):
        return ValidationResult(
            valid=False,
            error=f"{ENTRY_POINT_FUNCTION} is not an async function",
        )

    sig = inspect.signature(extract_fn)
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
