from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

ALLOWED_FILE_EXACT = [
    "prompts/system_prompt.md",
    "prompts/examples.md",
    "playground/extractor.py",
    "playground/__init__.py",
]

ALLOWED_DIR_PREFIX = "playground/"

EXCLUDED_PREFIXES = [
    "protected/",
    "corpus/",
    "experiments/",
    "evaluation/",
    "scripts/",
]

CORE_FILES = [
    "playground/extractor.py",
    "playground/__init__.py",
]


def is_allowed(file_path: str, operation: str) -> bool:
    root = _PROJECT_ROOT
    for prefix in EXCLUDED_PREFIXES:
        if file_path.startswith(prefix):
            return False

    if operation == "create_file":
        if not file_path.startswith(ALLOWED_DIR_PREFIX):
            return False
        if not file_path.endswith(".py"):
            return False
        full_path = root / file_path
        if full_path.exists():
            return False
        return True

    if operation == "delete_file":
        if not file_path.startswith(ALLOWED_DIR_PREFIX):
            return False
        if file_path in CORE_FILES:
            return False
        return True

    if file_path in ALLOWED_FILE_EXACT:
        return True

    if file_path.startswith(ALLOWED_DIR_PREFIX) and file_path.endswith(".py"):
        return True

    return False
