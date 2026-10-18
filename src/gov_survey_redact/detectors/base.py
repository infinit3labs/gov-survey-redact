"""Detector protocol and base classes for deterministic and NLP recognisers."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from ..types import Category, Finding, Span, Stage


@runtime_checkable
class Detector(Protocol):
    """Protocol implemented by every span detector in the pipeline."""

    name: str

    def detect(self, text: str) -> Iterator[Finding]:
        ...


class RegexDetector:
    """Base class for detectors backed by one or more regular expressions."""

    name = "regex"
    stage = Stage.DETERMINISTIC
    confidence = 1.0
    rule_id: str | None = None

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        raise NotImplementedError

    def category_for(self, match: re.Match[str]) -> Category:
        raise NotImplementedError

    def confidence_for(self, match: re.Match[str]) -> float:
        return self.confidence

    def metadata_for(self, match: re.Match[str]) -> dict[str, object]:
        meta: dict[str, object] = {}
        if self.rule_id:
            meta["recognizer"] = self.rule_id
        return meta

    def detect(self, text: str) -> Iterator[Finding]:
        for pattern in self.patterns():
            for match in pattern.finditer(text):
                if "span" in pattern.groupindex:
                    start, end = match.span("span")
                else:
                    start, end = match.span()
                if start < 0:
                    continue
                meta = self.metadata_for(match)
                if self.rule_id and "recognizer" not in meta:
                    meta["recognizer"] = self.rule_id
                yield Finding(
                    span=Span(start, end),
                    category=self.category_for(match),
                    detector=self.name,
                    stage=self.stage,
                    confidence=self.confidence_for(match),
                    value=text[start:end],
                    metadata=tuple(meta.items()),
                )


class SinglePatternDetector(RegexDetector):
    """Detector backed by a single regex pattern string and category."""

    pattern: str
    category: Category
    flags: int = 0

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (re.compile(self.pattern, self.flags),)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category


class CompiledPatternDetector(RegexDetector):
    """Detector backed by a pre-compiled regex and category."""

    regex: re.Pattern[str]
    category: Category

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return (self.regex,)

    def category_for(self, match: re.Match[str]) -> Category:
        return self.category
