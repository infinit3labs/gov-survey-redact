"""Offset-preserving text normalisation for survey free text.

Survey responses arrive with fullwidth digits/letters pasted from phones, smart
quotes and dashes, inconsistent spacing, diacritics, and leetspeak obfuscation.
Two normalisation levels are provided:

* ``normalise``: conservative case/Unicode/punctuation/whitespace folding.
* ``normalise_aligned``: strictly 1-to-1 character length preserving
  (``len(normalise_aligned(text)) == len(text)``), allowing regex detectors to
  match normalised text while emitting exact offsets into the caller's string.
* ``FoldedText``: aggressive alphanumeric + leetspeak folding with an index
  map back to original character offsets.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_SEPARATORS = re.compile(r"[\s\u00a0_/\\|]+")

_LEET = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "8": "b",
        "@": "a",
        "$": "s",
        "!": "i",
        "|": "l",
        "+": "t",
    }
)

_PUNCT_RE = re.compile("[‘’‛ʼ′＇`´]|[“”‟″＂]|[–—―−]|[\u00a0\u200b\u200c\u2060]")
_PUNCT_MAP = {
    **{c: "'" for c in "‘’‛ʼ′＇`´"},
    **{c: '"' for c in "“”‟″＂"},
    **{c: "-" for c in "–—―−"},
    **{c: " " for c in "\u00a0\u200b\u200c\u2060"},
}

_EXPANSIONS = {"…": "..."}
_EXPANSIONS_RE = re.compile("|".join(map(re.escape, _EXPANSIONS)))


@dataclass(frozen=True, slots=True)
class FoldedText:
    """Aggressively normalised view of ``source`` with an offset back-map."""

    source: str
    text: str
    source_index: tuple[int, ...]

    @classmethod
    def build(cls, source: str, *, leet: bool = True) -> FoldedText:
        folded_chars: list[str] = []
        indices: list[int] = []
        for source_pos, char in enumerate(source):
            decomposed = unicodedata.normalize("NFKD", char)
            base = "".join(
                c
                for c in decomposed
                if not unicodedata.combining(c) and not unicodedata.category(c).startswith("M")
            )
            base = base.casefold()
            if leet:
                base = base.translate(_LEET)
            for out_char in base:
                if out_char.isalnum():
                    folded_chars.append(out_char)
                    indices.append(source_pos)
        return cls(source=source, text="".join(folded_chars), source_index=tuple(indices))

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        """Translate a span in folded space back to original-string offsets.

        Out-of-range folded offsets are clamped so callers holding a stale span
        get a best-effort mapping instead of an exception. Spans that start past
        the end of the folded text are returned unchanged.
        """
        if start >= end or not self.source_index:
            return (start, end)
        folded_length = len(self.source_index)
        if start >= folded_length:
            return (start, end)
        end = min(end, folded_length)
        if start >= end:
            return (start, end)
        lo = min(self.source_index[start:end])
        hi = max(self.source_index[start:end]) + 1
        return (lo, hi)

    def contains(self, needle: str) -> bool:
        return needle in self.text

    def __bool__(self) -> bool:
        return bool(self.text)


def normalise(text: str) -> str:
    """Conservative normalisation for general matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _EXPANSIONS_RE.sub(lambda m: _EXPANSIONS[m.group()], text)
    text = _PUNCT_RE.sub(lambda m: _PUNCT_MAP.get(m.group(), m.group()), text)
    text = _SEPARATORS.sub(" ", text)
    return text.strip()


def normalise_aligned(text: str) -> str:
    """Length-preserving normalisation, including fullwidth folding.

    Guarantees ``len(result) == len(text)`` so any regex match span on the
    aligned string is an exact character span on ``text``.
    """
    if not text:
        return ""
    out: list[str] = []
    for char in text:
        if char in _PUNCT_MAP:
            out.append(_PUNCT_MAP[char])
            continue
        composed = unicodedata.normalize("NFKC", char)
        if len(composed) == 1:
            out.append(composed)
            continue
        decomposed = unicodedata.normalize("NFKD", char)
        base = "".join(
            c
            for c in decomposed
            if not unicodedata.combining(c) and not unicodedata.category(c).startswith("M")
        )
        out.append(base if len(base) == 1 else char)
    return "".join(out)


def tokenise(text: str) -> list[tuple[str, int, int]]:
    """Split text into ``(token, start, end)`` triples against ``text``."""
    out: list[tuple[str, int, int]] = []
    for match in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'’&.\-]*", text):
        token = match.group()
        start, end = match.span()
        while end > start and token[-1] in ".-&'’":
            token = token[:-1]
            end -= 1
        if token:
            out.append((token, start, end))
    return out


def token_key(token: str) -> str:
    """Alphanumeric case-folded comparison key for a token."""
    return "".join(c for c in token.casefold() if c.isalnum())


__all__ = [
    "FoldedText",
    "normalise",
    "normalise_aligned",
    "token_key",
    "tokenise",
]
