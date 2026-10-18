"""Term lexicon with SymSpell-style O(1) delete-index fuzzy matching.

Provides fast phrase and single-token lookup for organisational attribution
(internal systems, projects, legislation, compensation schemes, and awards)
with 1-edit typo tolerance for tokens of 6+ characters.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .normalise import token_key, tokenise
from .types import Category

MIN_FUZZY_LENGTH = 6
DEFAULT_MAX_DISTANCE = 1
DENSE_MENTION_RATIO = 0.05


@dataclass(frozen=True, slots=True)
class Term:
    """One lexicon entry for organisational or regulatory attribution."""

    term: str
    category: Category
    aliases: tuple[str, ...] = ()
    label: str = "internal term"
    exact_only: bool = False

    def __init__(
        self,
        term: str,
        category: Category = Category.INTERNAL_TERM,
        aliases: Sequence[str] = (),
        label: str = "internal term",
        exact_only: bool = False,
    ) -> None:
        object.__setattr__(self, "term", term)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "aliases", tuple(aliases))
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "exact_only", exact_only)

    @property
    def key(self) -> str:
        return token_key(self.term)

    def all_forms(self) -> tuple[str, ...]:
        return (self.term, *self.aliases)


@dataclass(frozen=True, slots=True)
class LexiconMatch:
    """A lexicon hit resolved to original-string character offsets."""

    span: tuple[int, int]
    matched_text: str
    canonical: str
    category: Category
    label: str
    fuzzy: bool

    @property
    def distance(self) -> int:
        return 1 if self.fuzzy else 0


@dataclass(frozen=True)
class Lexicon:
    """Immutable pre-indexed term set with SymSpell delete-index lookup."""

    terms: tuple[Term, ...] = ()
    max_distance: int = DEFAULT_MAX_DISTANCE

    _phrase_index: dict[str, Term] = field(default=None, compare=False, repr=False)  # type: ignore[assignment]
    _delete_index: dict[str, Term] = field(default=None, compare=False, repr=False)  # type: ignore[assignment]
    _max_phrase_tokens: int = field(default=1, compare=False, repr=False)
    _built: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def from_terms(
        cls,
        terms: Iterable[Term],
        *,
        max_distance: int = DEFAULT_MAX_DISTANCE,
    ) -> Lexicon:
        return cls(terms=tuple(terms), max_distance=max_distance)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        max_distance: int = DEFAULT_MAX_DISTANCE,
    ) -> Lexicon:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_spec(data, max_distance=max_distance)

    @classmethod
    def from_spec(
        cls,
        spec: Mapping[str, object],
        *,
        max_distance: int | None = None,
    ) -> Lexicon:
        distance = spec.get("max_distance", max_distance)
        raw_terms = spec.get("terms", ())
        terms = [
            Term(
                term=str(entry["term"]),
                category=Category(str(entry.get("category", Category.INTERNAL_TERM.value))),
                aliases=tuple(str(a) for a in entry.get("aliases", ())),  # type: ignore[union-attr]
                label=str(entry.get("label", "internal term")),
                exact_only=bool(entry.get("exact_only", False)),
            )
            for entry in raw_terms  # type: ignore[union-attr]
        ]
        return cls(
            terms=tuple(terms),
            max_distance=int(distance) if distance is not None else DEFAULT_MAX_DISTANCE,  # type: ignore[arg-type]
        )

    def merged(self, other: Lexicon) -> Lexicon:
        return Lexicon(
            terms=self.terms + other.terms,
            max_distance=max(self.max_distance, other.max_distance),
        )

    def _build(self) -> None:
        if self._built:
            return
        phrase: dict[str, Term] = {}
        delete: dict[str, Term] = {}
        max_tokens = 1
        for term in self.terms:
            for form in term.all_forms():
                key = token_key(form)
                if not key:
                    continue
                if key not in phrase:
                    phrase[key] = term
                max_tokens = max(max_tokens, len(form.split()))
                if _is_single_token(form) and not term.exact_only:
                    for variant in _deletions(key, self.max_distance):
                        delete.setdefault(variant, term)
        object.__setattr__(self, "_phrase_index", phrase)
        object.__setattr__(self, "_delete_index", delete)
        object.__setattr__(self, "_max_phrase_tokens", max_tokens)
        object.__setattr__(self, "_built", True)

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[Term]:
        return iter(self.terms)

    def match(self, text: str) -> list[LexiconMatch]:
        """Return every lexicon hit in ``text`` in ascending offset order."""
        self._build()
        if not text.strip():
            return []

        tokens = tokenise(text)
        if not tokens:
            return []

        hits: list[LexiconMatch] = []
        claimed: list[tuple[int, int]] = []

        # Pass 1: multi-word phrases, longest first
        if self._max_phrase_tokens > 1:
            for width in range(self._max_phrase_tokens, 1, -1):
                for start_index in range(len(tokens) - width + 1):
                    window = tokens[start_index : start_index + width]
                    key = "".join(token_key(tok) for tok, _, _ in window)
                    term = self._phrase_index.get(key)
                    if term is None:
                        continue
                    span = (window[0][1], window[-1][2])
                    if _any_overlap(span, claimed):
                        continue
                    claimed.append(span)
                    hits.append(
                        LexiconMatch(
                            span=span,
                            matched_text=text[span[0] : span[1]],
                            canonical=term.term,
                            category=term.category,
                            label=term.label,
                            fuzzy=False,
                        )
                    )

        # Pass 2: single tokens, exact then 1-edit delete-index fuzzy
        for token, start, end in tokens:
            if _any_overlap((start, end), claimed):
                continue
            key = token_key(token)
            if not key:
                continue
            term = self._phrase_index.get(key)
            if term is not None:
                claimed.append((start, end))
                hits.append(
                    LexiconMatch(
                        span=(start, end),
                        matched_text=token,
                        canonical=term.term,
                        category=term.category,
                        label=term.label,
                        fuzzy=False,
                    )
                )
                continue
            fuzzy = self._fuzzy_term(key)
            if fuzzy is not None and len(key) >= MIN_FUZZY_LENGTH:
                claimed.append((start, end))
                hits.append(
                    LexiconMatch(
                        span=(start, end),
                        matched_text=token,
                        canonical=fuzzy.term,
                        category=fuzzy.category,
                        label=fuzzy.label,
                        fuzzy=True,
                    )
                )

        hits.sort(key=lambda h: h.span[0])
        return hits

    def _fuzzy_term(self, key: str) -> Term | None:
        if len(key) < MIN_FUZZY_LENGTH:
            return None
        best: tuple[int, int, Term] | None = None
        for distance in range(0, self.max_distance + 1):
            for variant in _deletions(key, distance):
                term = self._delete_index.get(variant)
                if term is None:
                    continue
                candidate = (distance, -len(term.key), term)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
            if best is not None:
                return best[2]
        return None

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "term": t.term,
                "category": t.category.value,
                "label": t.label,
                "exact_only": t.exact_only,
                "aliases": list(t.aliases),
            }
            for t in self.terms
        ]


def _is_single_token(form: str) -> bool:
    return bool(form) and " " not in form


def _deletions(key: str, distance: int) -> list[str]:
    if distance <= 0:
        return [key]
    seen = {key}
    frontier = [key]
    for _ in range(distance):
        next_frontier: list[str] = []
        for variant in frontier:
            for index in range(len(variant)):
                candidate = variant[:index] + variant[index + 1 :]
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    next_frontier.append(candidate)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(seen)


def _any_overlap(span: tuple[int, int], claimed: Sequence[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in claimed)


def load_builtin_lexicon() -> Lexicon:
    """Load the bundled Australian Government & WHS seed lexicon."""
    from importlib.resources import files

    resource = files("gov_survey_redact").joinpath("resources/lexicon.json")
    return Lexicon.from_json(resource)  # type: ignore[arg-type]


__all__ = [
    "DEFAULT_MAX_DISTANCE",
    "MIN_FUZZY_LENGTH",
    "Lexicon",
    "LexiconMatch",
    "Term",
    "load_builtin_lexicon",
]
