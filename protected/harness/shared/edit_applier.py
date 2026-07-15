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


_DELETED = object()


def apply_edits(edits: list[Edit]) -> ApplyResult:
    """A009-7 (audit #6): validate and stage every edit against an in-memory working set,
    then write the whole batch to disk only once it all validates. This (a) lets overlapping
    same-file edits see each other's effect during validation — the old code validated each
    edit against the *original* file, so a second edit whose match the first consumed was
    silently dropped while still reported as applied — and (b) never lets an apply exception
    escape and abort the study; on failure the caller rolls back the playground (git
    checkout + clean, A009-6), which repairs any partial write."""
    root = _PROJECT_ROOT
    if not edits:
        return ApplyResult(applied=True, files_changed=[])

    from protected.harness.shared.allowlist import is_allowed

    buffers: dict[str, object] = {}  # file_path -> current content (str) or _DELETED

    def _exists(path: str) -> bool:
        if path in buffers:
            return buffers[path] is not _DELETED
        return (root / path).exists()

    def _read(path: str) -> str | None:
        if path in buffers:
            c = buffers[path]
            return None if c is _DELETED else c  # type: ignore[return-value]
        full = root / path
        return full.read_text(encoding="utf-8", errors="replace") if full.exists() else None

    try:
        for edit in edits:
            if not is_allowed(edit.file_path, edit.operation):
                return ApplyResult(applied=False, reason="allowlist_violation", offending_path=edit.file_path)
            p = edit.file_path

            if edit.operation == "replace_string":
                if edit.old_string is None:
                    return ApplyResult(applied=False, reason="missing_old_string", offending_path=p)
                if edit.new_string is None:
                    return ApplyResult(applied=False, reason="missing_new_string", offending_path=p)
                if edit.new_content is not None:
                    return ApplyResult(applied=False, reason="unexpected_new_content", offending_path=p)
                if not _exists(p):
                    return ApplyResult(applied=False, reason="file_not_found", offending_path=p)
                content = _read(p)
                count = content.count(edit.old_string)
                if count == 0:
                    return ApplyResult(applied=False, reason="no_match", offending_path=p)
                if count > 1:
                    return ApplyResult(applied=False, reason="ambiguous_match", offending_path=p)
                buffers[p] = content.replace(edit.old_string, edit.new_string, 1)

            elif edit.operation == "replace_file":
                if edit.new_content is None or edit.new_content == "":
                    return ApplyResult(applied=False, reason="empty_file_replacement", offending_path=p)
                if edit.old_string is not None or edit.new_string is not None:
                    return ApplyResult(applied=False, reason="unexpected_old_or_new_string", offending_path=p)
                if not _exists(p):
                    return ApplyResult(applied=False, reason="file_not_found", offending_path=p)
                buffers[p] = edit.new_content

            elif edit.operation == "create_file":
                if edit.new_content is None:
                    return ApplyResult(applied=False, reason="missing_new_content", offending_path=p)
                if _exists(p):
                    return ApplyResult(applied=False, reason="create_file_exists", offending_path=p)
                buffers[p] = edit.new_content

            elif edit.operation == "delete_file":
                if not _exists(p):
                    return ApplyResult(applied=False, reason="delete_file_missing", offending_path=p)
                buffers[p] = _DELETED

        # All edits validated against the running buffer — commit to disk. A009-11: write with
        # newline="" so agent edits stay LF (no CRLF translation on Windows) — closes the
        # long-standing CRLF/hash-gate landmine at the write site.
        for path, content in buffers.items():
            full = root / path
            if content is _DELETED:
                if full.exists():
                    full.unlink()
                continue
            full.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = full.with_suffix(full.suffix + ".tmp")
            tmp_path.write_text(content, encoding="utf-8", newline="")  # type: ignore[arg-type]
            tmp_path.replace(full)

    except Exception as e:
        return ApplyResult(applied=False,
                           reason=f"apply_exception: {type(e).__name__}: {e}",
                           offending_path=None)

    return ApplyResult(applied=True, files_changed=[e.file_path for e in edits])
