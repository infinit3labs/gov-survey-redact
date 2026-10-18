"""OCR geometry and confidence integration (Azure AI Document Intelligence Read/Layout).

Separates OCR from PII recognition while retaining word-level bounding polygons
and confidence scores so:
1. Identified character spans ``[start_char, end_char)`` can be mapped directly
   back into bounding polygons for visual PDF/image redaction.
2. Low OCR confidence on a span or field increases review probability and
   prevents OCR-corrupted identifiers from bypassing review.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .types import Category, Finding, Span


@dataclass(frozen=True, slots=True)
class BoundingPolygon:
    """Page-indexed bounding polygon ``[(x0, y0), (x1, y1), (x2, y2), (x3, y3)]``."""

    page: int
    points: tuple[tuple[float, float], ...]

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """Axis-aligned ``(min_x, min_y, max_x, max_y)`` box."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def as_dict(self) -> dict[str, object]:
        return {
            "page": self.page,
            "points": [list(p) for p in self.points],
            "bbox": list(self.bounding_box),
        }


@dataclass(frozen=True, slots=True)
class OCRWord:
    """One word extracted by Azure Document Intelligence Read/Layout."""

    text: str
    span: Span
    confidence: float
    polygon: BoundingPolygon | None = None


@dataclass(frozen=True, slots=True)
class VisualRedactionBox:
    """Visual redaction region on a scanned page corresponding to a PII finding."""

    category: Category
    span: Span
    polygons: tuple[BoundingPolygon, ...]
    min_ocr_confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category.value,
            "start_char": self.span.start,
            "end_char": self.span.end,
            "min_ocr_confidence": round(self.min_ocr_confidence, 4),
            "polygons": [p.as_dict() for p in self.polygons],
        }


@dataclass(frozen=True, slots=True)
class OCRFieldLayout:
    """OCR word geometry and confidence for a single survey text field."""

    text: str
    words: tuple[OCRWord, ...] = ()
    field_confidence: float | None = None

    @property
    def mean_confidence(self) -> float:
        if self.field_confidence is not None:
            return self.field_confidence
        if not self.words:
            return 1.0
        return sum(w.confidence for w in self.words) / len(self.words)

    @property
    def min_confidence(self) -> float:
        if not self.words:
            return self.field_confidence if self.field_confidence is not None else 1.0
        low = min(w.confidence for w in self.words)
        if self.field_confidence is not None:
            return min(low, self.field_confidence)
        return low

    def span_ocr_confidence(self, span: Span) -> float:
        """Return the minimum OCR confidence of words overlapping ``span``."""
        overlapping = [w.confidence for w in self.words if w.span.overlaps(span)]
        if not overlapping:
            return self.mean_confidence
        return min(overlapping)

    def visual_boxes_for(self, findings: Sequence[Finding]) -> list[VisualRedactionBox]:
        """Map character-offset findings back to page bounding polygons."""
        boxes: list[VisualRedactionBox] = []
        for finding in findings:
            overlapping = [w for w in self.words if w.span.overlaps(finding.span)]
            polygons = tuple(w.polygon for w in overlapping if w.polygon is not None)
            min_conf = (
                min(w.confidence for w in overlapping)
                if overlapping
                else self.mean_confidence
            )
            boxes.append(
                VisualRedactionBox(
                    category=finding.category,
                    span=finding.span,
                    polygons=polygons,
                    min_ocr_confidence=min_conf,
                )
            )
        return boxes


__all__ = [
    "BoundingPolygon",
    "OCRFieldLayout",
    "OCRWord",
    "VisualRedactionBox",
]
