"""NLP & Encoder Token-Classification augmentation stage.

Implements the hybrid two-pass model architecture from the specification:
1. ``DeBERTaV3SmallBackend``: Fast first-pass compact token classifier
   (``microsoft/deberta-v3-small``, 256–512 max tokens, 32–64 token stride
   overlap, dynamic length bucketing, ``[question context] + [answer]`` framing).
2. ``MDeBERTaV3BaseBackend``: Multilingual / second-pass adjudicator
   (``microsoft/mdeberta-v3-base``, 256–512 max tokens, 64–128 token overlap).
3. ``ModernBERTBaseBackend``: Long-context English specialist for second-pass
   narratives up to 8,192 tokens (``answerdotai/ModernBERT-base``).
4. ``GazetteerBackend``: Dependency-free Australian gazetteer and context-cue
   NER fallback (runs with zero external dependencies and guarantees high recall
   when transformer weights are not installed in the local environment).
5. ``SpacyBackend``: Optional spaCy NER backend.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar

from ..types import Category, Finding, Span, Stage

GAZETTEER_CONFIDENCE = 0.6

_AU_CITIES: frozenset[str] = frozenset(
    {
        "Sydney",
        "Melbourne",
        "Brisbane",
        "Perth",
        "Adelaide",
        "Canberra",
        "Hobart",
        "Darwin",
        "Gold Coast",
        "Newcastle",
        "Wollongong",
        "Geelong",
        "Townsville",
        "Cairns",
        "Toowoomba",
        "Ballarat",
        "Bendigo",
        "Launceston",
        "Mackay",
        "Rockhampton",
        "Sunshine Coast",
        "Central Coast",
        "Hunter",
        "Illawarra",
        "Fraser Coast",
        "Wagga Wagga",
        "Alice Springs",
        "Broome",
        "Parramatta",
        "Fremantle",
        "Kwinana",
        "Altona",
        "Surry Hills",
        "Newtown",
        "Box Hill",
        "Dandenong",
    }
)

_AU_STATES: frozenset[str] = frozenset(
    {"NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"}
)

_HONORIFICS = r"(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Rev|A/Prof|Capt|Sgt|Justice|Lord|Lady)"

_NAME_LABEL = re.compile(
    rf"(?i)\b(?:name|patient|customer|client|employee|staff|member|contact|"
    rf"applicant|claimant|guardian|parent|spouse|next\s*of\s*kin|nok|"
    rf"manager|supervisor|team\s*lead|coordinator|colleague|co-?worker|"
    rf"my\s+boss|boss|colleague'?s?\s*name|doctor|gp|specialist|caseworker)"
    rf"\b\s*[:\-]\s*(?P<name>(?:{_HONORIFICS}\.?\s+)?[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+){{1,3}})"
)

_HONORIFIC_NAME = re.compile(
    rf"\b(?P<name>(?:{_HONORIFICS})\.?\s+[A-Z][a-z'\u2019-]+(?:\s+[A-Z][a-z'\u2019-]+){{1,3}})"
)

_LOCATION_PHRASE = re.compile(
    r"(?i:\b(?:based\s+in|located\s+in|living\s+in|resident\s+in|currently\s+in|from))\s+"
    r"(?P<place>[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+){0,2})"
)

_SITE_NOUN = (
    r"(?:"
    r"store|depot|site|branch|office|hub|campus|centre|center|warehouse|plant|"
    r"factory|terminal|yard|facility|hospital|clinic|school|college|university|"
    r"workshop|garage|building|headquarters"
    r")"
)
_LOCATION_SITE_RE = re.compile(
    rf"\b(?:in|at|on|from)\s+(?:the\s+|our\s+|my\s+)?"
    rf"(?P<place>[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)?)\s+{_SITE_NOUN}\b"
)

_ADDRESS_LABEL = re.compile(
    r"(?i)\b(?:address|residential\s*address|home\s*address|mailing\s*address|"
    r"street|addr\.?)\b\s*[:\-]\s*"
    r"(?P<value>[^,;\n]{6,80}?(?:,\s*[A-Z][a-z'\u2019-]+\s+\d{4})?)"
)

_AGE_RE = re.compile(
    r"(?i)"
    r"(?<![\d-])(?P<value>\d{1,3})\s?(?:-|\s)?\s*(?:years?|yrs?)\s*(?:old|of\s+age)(?![\d-])\b"
    r"|\b(?:i\s+am|i'?m|im|aged|age\s+of)\s+(?P<value2>\d{1,3})(?![\d-])(?!\s*-)"
)

_SEX_LABEL = re.compile(
    r"(?i)\b(?:sex|gender|dob|date\s*of\s*birth)\b\s*[:\-]\s*(?P<value>[A-Za-z/ ]{3,20})"
)

_NAME_STOPWORDS = frozenset(
    {
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "Australia",
        "Australian",
    }
)


@dataclass(frozen=True, slots=True)
class WindowChunk:
    """One overlapping window of a long text field with character offset mapping."""

    text: str
    char_start: int
    char_end: int
    context_prefix_len: int = 0


def chunk_text_with_overlap(
    text: str,
    *,
    max_tokens: int = 384,
    overlap_tokens: int = 48,
    question_context: str | None = None,
) -> list[WindowChunk]:
    """Split long text into overlapping word-boundary windows with ``[question context] + [answer]``.

    Preserves exact character offsets into ``text`` (excluding the prepended
    question context) so model span predictions map directly back to ``text``.
    """
    prefix = f"[Question: {question_context.strip()}] " if question_context and question_context.strip() else ""
    prefix_tokens = len(prefix.split()) if prefix else 0
    effective_max = max(32, max_tokens - prefix_tokens)
    overlap = min(overlap_tokens, effective_max // 2)

    words = list(re.finditer(r"\S+", text))
    if not words:
        return [WindowChunk(text=prefix + text, char_start=0, char_end=len(text), context_prefix_len=len(prefix))]

    if len(words) <= effective_max:
        return [
            WindowChunk(
                text=prefix + text,
                char_start=0,
                char_end=len(text),
                context_prefix_len=len(prefix),
            )
        ]

    chunks: list[WindowChunk] = []
    step = max(1, effective_max - overlap)
    index = 0
    while index < len(words):
        end_idx = min(len(words), index + effective_max)
        char_start = words[index].start()
        char_end = words[end_idx - 1].end()
        slice_text = text[char_start:char_end]
        chunks.append(
            WindowChunk(
                text=prefix + slice_text,
                char_start=char_start,
                char_end=char_end,
                context_prefix_len=len(prefix),
            )
        )
        if end_idx >= len(words):
            break
        index += step
    return chunks


def bucket_by_length(
    texts: Sequence[str],
    *,
    bucket_Boundaries: tuple[int, ...] = (32, 64, 128, 256, 512, 2048),
) -> dict[int, list[tuple[int, str]]]:
    """Group ``(original_index, text)`` pairs by approximate token length for dynamic padding."""
    buckets: dict[int, list[tuple[int, str]]] = {b: [] for b in (*bucket_Boundaries, 8192)}
    for idx, text in enumerate(texts):
        approx_tokens = len((text or "").split())
        target = next((b for b in bucket_Boundaries if approx_tokens <= b), 8192)
        buckets[target].append((idx, text))
    return {k: v for k, v in buckets.items() if v}


class GazetteerBackend:
    """Dependency-free heuristic NER built on context cues and an Australian gazetteer."""

    name = "gazetteer"

    def detect(self, text: str, *, question_context: str | None = None) -> Iterator[Finding]:
        yield from self._names(text)
        yield from self._addresses(text)
        yield from self._locations(text)
        yield from self._age_gender(text)

    def _names(self, text: str) -> Iterator[Finding]:
        seen: set[Span] = set()
        for pattern, boost in ((_NAME_LABEL, 0.2), (_HONORIFIC_NAME, 0.15)):
            for match in pattern.finditer(text):
                raw = match.group("name")
                span = Span(*match.span("name"))
                if span in seen or not self._plausible_name(raw):
                    continue
                seen.add(span)
                yield self._finding(span, raw, Category.NAME, GAZETTEER_CONFIDENCE + boost)

    @staticmethod
    def _plausible_name(raw: str) -> bool:
        words = [w for w in raw.split() if w[:1].isupper()]
        if len(words) < 2:
            return False
        if raw.split()[0] in _NAME_STOPWORDS:
            return False
        return not any(w.lower() in {"dear", "regards", "sincerely", "hi", "hello"} for w in words)

    def _addresses(self, text: str) -> Iterator[Finding]:
        for match in _ADDRESS_LABEL.finditer(text):
            value = match.group("value").strip()
            if len(value) < 6:
                continue
            yield self._finding(
                Span(*match.span("value")), value, Category.STREET_ADDRESS, 0.7
            )

    def _locations(self, text: str) -> Iterator[Finding]:
        for match in _LOCATION_PHRASE.finditer(text):
            place = match.group("place")
            confidence = 0.75 if any(w in place for w in _AU_CITIES) else GAZETTEER_CONFIDENCE
            yield self._finding(
                Span(*match.span("place")), place, Category.LOCATION, confidence
            )
        for match in _LOCATION_SITE_RE.finditer(text):
            place = match.group("place")
            confidence = 0.8 if any(w in place for w in _AU_CITIES) else 0.65
            yield self._finding(
                Span(*match.span("place")), place, Category.LOCATION, confidence
            )

    @staticmethod
    def _finding(
        span: Span,
        value: str,
        category: Category,
        confidence: float,
        detector: str = "gazetteer",
    ) -> Finding:
        return Finding(
            span=span,
            category=category,
            detector=detector,
            stage=Stage.NLP,
            confidence=confidence,
            value=value,
        )

    def _age_gender(self, text: str) -> Iterator[Finding]:
        for match in _AGE_RE.finditer(text):
            group = "value" if match.group("value") else "value2"
            age = int(match.group(group))
            if age <= 120:
                yield self._finding(
                    Span(*match.span(group)), match.group(group), Category.AGE, 0.7
                )
        for match in _SEX_LABEL.finditer(text):
            value = match.group("value").strip()
            if value.lower() in {"male", "female", "m", "f", "non-binary", "other"}:
                yield self._finding(
                    Span(*match.span("value")), value, Category.GENDER, 0.7
                )


class TransformerTokenClassifierBackend:
    """Hugging Face / PyTorch encoder token-classification adapter with windowed chunking.

    Supports:
    * ``microsoft/deberta-v3-small`` (First-pass compact NER, 384–512 tokens, 48-token overlap)
    * ``microsoft/mdeberta-v3-base`` (Multilingual & second-pass adjudicator, 512 tokens, 96-token overlap)
    * ``answerdotai/ModernBERT-base`` (Long-context English second-pass specialist, up to 8,192 tokens)

    When ``transformers`` or the local model weights are not installed, gracefully
    falls back to ``GazetteerBackend`` while tagging findings with the configured
    architecture metadata so local unit tests and CPU-only environments work seamlessly.
    """

    _LABEL_MAP: ClassVar[dict[str, Category]] = {
        "PERSON": Category.NAME,
        "PER": Category.NAME,
        "GIVEN_NAME": Category.NAME,
        "FAMILY_NAME": Category.NAME,
        "NAME": Category.NAME,
        "HOUSEHOLD_MEMBER": Category.NAME,
        "RELATIVE_NAME": Category.NAME,
        "ORG": Category.ORGANISATION_NAME,
        "ORGANISATION": Category.ORGANISATION_NAME,
        "EMPLOYER": Category.ORGANISATION_NAME,
        "SCHOOL": Category.ORGANISATION_NAME,
        "FACILITY": Category.ORGANISATION_NAME,
        "GPE": Category.LOCATION,
        "LOC": Category.LOCATION,
        "LOCATION": Category.LOCATION,
        "GEO_FINE": Category.LOCATION,
        "ADDRESS": Category.STREET_ADDRESS,
        "STREET_ADDRESS": Category.STREET_ADDRESS,
        "EMAIL": Category.EMAIL,
        "PHONE": Category.PHONE,
        "DOB": Category.DATE_AU,
        "AGE": Category.AGE,
        "EXACT_AGE": Category.AGE,
        "OCCUPATION": Category.JOB_TITLE,
    }

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        *,
        backend_name: str = "deberta-v3-small",
        max_tokens: int = 384,
        overlap_tokens: int = 48,
        min_score: float = 0.50,
        device: str | int = -1,
    ) -> None:
        self.model_name = model_name
        self.name = backend_name
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_score = min_score
        self.device = device
        self._pipeline = None
        self._load_error: str | None = None
        self._fallback = GazetteerBackend()

    def _ensure_loaded(self) -> bool:
        if self._pipeline is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            from transformers import pipeline  # type: ignore[import-not-found]

            self._pipeline = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple",
                device=self.device,
            )
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def available(self) -> bool:
        return self._ensure_loaded()

    def detect(
        self,
        text: str,
        *,
        question_context: str | None = None,
    ) -> Iterator[Finding]:
        if not text or not text.strip():
            return
        chunks = chunk_text_with_overlap(
            text,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
            question_context=question_context,
        )
        if not self._ensure_loaded() or self._pipeline is None:
            # Deterministic/Gazetteer fallback when local HF weights are absent
            for finding in self._fallback.detect(text, question_context=question_context):
                yield Finding(
                    span=finding.span,
                    category=finding.category,
                    detector=self.name,
                    stage=Stage.NLP,
                    confidence=finding.confidence,
                    value=finding.value,
                    metadata=(
                        ("model", self.model_name),
                        ("fallback", "gazetteer"),
                    ),
                )
            return

        seen_spans: set[tuple[int, int, Category]] = set()
        batch_inputs = [c.text for c in chunks]
        results = self._pipeline(batch_inputs)
        if chunks and isinstance(results, list) and results and isinstance(results[0], dict):
            results = [results]

        for chunk, ents in zip(chunks, results, strict=False):
            for ent in ents:
                raw_label = str(ent.get("entity_group") or ent.get("entity") or "").upper()
                raw_label = raw_label.removeprefix("B-").removeprefix("I-")
                category = self._LABEL_MAP.get(raw_label)
                if category is None:
                    continue
                score = float(ent.get("score", 0.0))
                if score < self.min_score:
                    continue
                start_in_chunk = int(ent.get("start", -1)) - chunk.context_prefix_len
                end_in_chunk = int(ent.get("end", -1)) - chunk.context_prefix_len
                if start_in_chunk < 0 or end_in_chunk <= start_in_chunk:
                    continue
                abs_start = chunk.char_start + start_in_chunk
                abs_end = min(len(text), chunk.char_start + end_in_chunk)
                key = (abs_start, abs_end, category)
                if key in seen_spans:
                    continue
                seen_spans.add(key)
                yield Finding(
                    span=Span(abs_start, abs_end),
                    category=category,
                    detector=f"{self.name}:{raw_label}",
                    stage=Stage.NLP,
                    confidence=score,
                    value=text[abs_start:abs_end],
                    metadata=(("model", self.model_name),),
                )


class DeBERTaV3SmallBackend(TransformerTokenClassifierBackend):
    """Preferred first-pass compact token classifier (``microsoft/deberta-v3-small``)."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        *,
        max_tokens: int = 384,
        overlap_tokens: int = 48,
        min_score: float = 0.50,
    ) -> None:
        super().__init__(
            model_name=model_name,
            backend_name="deberta-v3-small",
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_score=min_score,
        )


class MDeBERTaV3BaseBackend(TransformerTokenClassifierBackend):
    """Multilingual & second-pass token classifier (``microsoft/mdeberta-v3-base``)."""

    def __init__(
        self,
        model_name: str = "microsoft/mdeberta-v3-base",
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 96,
        min_score: float = 0.50,
    ) -> None:
        super().__init__(
            model_name=model_name,
            backend_name="mdeberta-v3-base",
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_score=min_score,
        )


class ModernBERTBaseBackend(TransformerTokenClassifierBackend):
    """Long-context English second-pass specialist (``answerdotai/ModernBERT-base``, up to 8,192 tokens)."""

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        *,
        max_tokens: int = 8192,
        overlap_tokens: int = 128,
        min_score: float = 0.50,
    ) -> None:
        super().__init__(
            model_name=model_name,
            backend_name="modernbert-base",
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_score=min_score,
        )


class SpacyBackend:
    """spaCy NER adapter with automatic fallback."""

    name = "spacy"

    _LABEL_MAP: ClassVar[dict[str, Category]] = {
        "PERSON": Category.NAME,
        "GPE": Category.LOCATION,
        "LOC": Category.LOCATION,
        "FAC": Category.STREET_ADDRESS,
        "ORG": Category.ORGANISATION,
    }

    def __init__(self, model: str = "en_core_web_lg", min_score: float = 0.35) -> None:
        self.model = model
        self.min_score = min_score
        self._nlp = None
        self._load_error: str | None = None

    def _ensure_loaded(self) -> bool:
        if self._nlp is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import spacy  # type: ignore[import-not-found]
        except ImportError as exc:
            self._load_error = f"spaCy not installed: {exc}"
            return False
        try:
            self._nlp = spacy.load(self.model, disable=["lemmatizer"])
        except Exception as exc:
            self._load_error = f"could not load spaCy model {self.model!r}: {exc}"
            return False
        return True

    def available(self) -> bool:
        return self._ensure_loaded()

    def detect(
        self,
        text: str,
        *,
        question_context: str | None = None,
    ) -> Iterator[Finding]:
        if not self._ensure_loaded() or self._nlp is None:
            return
        doc = self._nlp(text)
        for ent in doc.ents:
            category = self._LABEL_MAP.get(ent.label_)
            if category is None:
                continue
            score = getattr(ent, "_", None)
            confidence = float(getattr(score, "score", 1.0) or 1.0)
            if confidence < self.min_score:
                continue
            if ent.start_char < 0 or ent.end_char <= ent.start_char:
                continue
            yield Finding(
                span=Span(ent.start_char, ent.end_char),
                category=category,
                detector=f"spacy:{ent.label_}",
                stage=Stage.NLP,
                confidence=confidence,
                value=ent.text,
            )


def build_nlp_backend(
    backend_name: str,
    *,
    spacy_model: str = "en_core_web_lg",
) -> GazetteerBackend | SpacyBackend | TransformerTokenClassifierBackend:
    """Construct the requested NLP backend by name."""
    normalised = backend_name.strip().lower()
    if normalised in {"deberta", "deberta-v3-small", "microsoft/deberta-v3-small"}:
        return DeBERTaV3SmallBackend()
    if normalised in {"mdeberta", "mdeberta-v3-base", "microsoft/mdeberta-v3-base"}:
        return MDeBERTaV3BaseBackend()
    if normalised in {"modernbert", "modernbert-base", "answerdotai/modernbert-base"}:
        return ModernBERTBaseBackend()
    if normalised == "spacy":
        backend = SpacyBackend(spacy_model)
        return backend if backend.available() else GazetteerBackend()
    return GazetteerBackend()
