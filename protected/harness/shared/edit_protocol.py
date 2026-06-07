from __future__ import annotations
from dataclasses import dataclass
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


@dataclass
class Episode:
    observation: str
    hypothesis: str
    action: str
    expectation: str


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
