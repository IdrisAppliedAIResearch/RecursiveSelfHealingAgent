from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from protected.harness.shared.edit_protocol import Edit

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class ApplyResult:
    applied: bool
    reason: str | None = None
    offending_path: str | None = None
    files_changed: list[str] | None = None


def apply_edits(edits: list[Edit]) -> ApplyResult:
    root = _PROJECT_ROOT
    if not edits:
        return ApplyResult(applied=True, files_changed=[])

    from protected.harness.shared.allowlist import is_allowed

    for edit in edits:
        if not is_allowed(edit.file_path, edit.operation):
            return ApplyResult(
                applied=False,
                reason="allowlist_violation",
                offending_path=edit.file_path,
            )

        full_path = root / edit.file_path

        if edit.operation == "replace_string":
            if edit.old_string is None:
                return ApplyResult(applied=False, reason="missing_old_string", offending_path=edit.file_path)
            if edit.new_string is None:
                return ApplyResult(applied=False, reason="missing_new_string", offending_path=edit.file_path)
            if edit.new_content is not None:
                return ApplyResult(applied=False, reason="unexpected_new_content", offending_path=edit.file_path)
            if not full_path.exists():
                return ApplyResult(applied=False, reason="file_not_found", offending_path=edit.file_path)
            content = full_path.read_text(encoding="utf-8", errors="replace")
            count = content.count(edit.old_string)
            if count == 0:
                return ApplyResult(applied=False, reason="no_match", offending_path=edit.file_path)
            if count > 1:
                return ApplyResult(applied=False, reason="ambiguous_match", offending_path=edit.file_path)

        elif edit.operation == "replace_file":
            if edit.new_content is None or edit.new_content == "":
                return ApplyResult(applied=False, reason="empty_file_replacement", offending_path=edit.file_path)
            if edit.old_string is not None or edit.new_string is not None:
                return ApplyResult(applied=False, reason="unexpected_old_or_new_string", offending_path=edit.file_path)
            if not full_path.exists():
                return ApplyResult(applied=False, reason="file_not_found", offending_path=edit.file_path)

        elif edit.operation == "create_file":
            if edit.new_content is None:
                return ApplyResult(applied=False, reason="missing_new_content", offending_path=edit.file_path)
            if full_path.exists():
                return ApplyResult(applied=False, reason="create_file_exists", offending_path=edit.file_path)

        elif edit.operation == "delete_file":
            if not full_path.exists():
                return ApplyResult(applied=False, reason="delete_file_missing", offending_path=edit.file_path)

    for edit in edits:
        full_path = root / edit.file_path

        if edit.operation == "replace_string":
            content = full_path.read_text(encoding="utf-8", errors="replace")
            new_content = content.replace(edit.old_string, edit.new_string, 1)
            tmp_path = full_path.with_suffix(full_path.suffix + ".tmp")
            tmp_path.write_text(new_content, encoding="utf-8")
            tmp_path.replace(full_path)

        elif edit.operation == "replace_file":
            tmp_path = full_path.with_suffix(full_path.suffix + ".tmp")
            tmp_path.write_text(edit.new_content, encoding="utf-8")
            tmp_path.replace(full_path)

        elif edit.operation == "create_file":
            full_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = full_path.with_suffix(full_path.suffix + ".tmp")
            tmp_path.write_text(edit.new_content, encoding="utf-8")
            tmp_path.replace(full_path)

        elif edit.operation == "delete_file":
            full_path.unlink()

    return ApplyResult(applied=True, files_changed=[e.file_path for e in edits])
