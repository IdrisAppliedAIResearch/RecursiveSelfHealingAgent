from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from extractor.provider import TokenUsage


@dataclass
class Edit:
    file_path: str
    operation: str
    old_string: str | None
    new_string: str | None
    new_content: str | None

    def __post_init__(self):
        # A004-5: normalize path separators to POSIX before any allowlist or
        # filesystem operation. The model sometimes emits Windows-style
        # separators (playground\extractor.py), which the POSIX-slash allowlist
        # would otherwise reject as a false violation.
        if self.file_path:
            self.file_path = self.file_path.replace("\\", "/")


@dataclass
class Episode:
    observation: str
    hypothesis: str
    action: str
    expectation: str
    edits_applied: bool = False
    field_failures: list = field(default_factory=list)


@dataclass
class AgentResponse:
    episode: Episode
    rationale: str
    edits: list[Edit]
    token_usage: TokenUsage | None = None


@dataclass
class RepairResponse:
    edits: list[Edit]
    token_usage: TokenUsage | None = None


@dataclass
class AgentFailure:
    reason: str
    raw_response: str | None = None


@dataclass
class AssessmentResult:
    routing_trend: str
    last_action_effect: str
    pattern_observed: str
    hypothesis: str
    raw_response: str
    token_usage: TokenUsage | None = None
    field_failures: list = field(default_factory=list)
