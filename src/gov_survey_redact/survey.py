"""Survey profile: detector assembly and policy for Australian Government surveys."""

from __future__ import annotations

from dataclasses import replace

from .detectors import au_ids
from .detectors.base import Detector
from .detectors.implication import default_implication_detector
from .detectors.org import default_organisation_detectors
from .detectors.quasi import default_quasi_detectors
from .engine import DEFAULT_DETECTORS, Policy, Redactor
from .lexicon import Lexicon, load_builtin_lexicon
from .types import Category, MaskStyle

DEFAULT_GENERALISED = frozenset(
    {
        Category.AGE,
        Category.DATE_AU,
        Category.DATE_ISO,
        Category.TENURE,
        Category.TEAM_SIZE,
    }
)

HEALTH_DETAIL_STYLE: dict[Category, MaskStyle] = {
    Category.HEALTH_DETAIL: MaskStyle.LABEL,
}

ATTRIBUTION_STYLE: dict[Category, MaskStyle] = {
    Category.ORGANISATION_NAME: MaskStyle.LABEL,
    Category.INTERNAL_TERM: MaskStyle.LABEL,
    Category.AWARD: MaskStyle.LABEL,
}

IMPLICATION_STYLE: dict[Category, MaskStyle] = {
    Category.IMPLICATION: MaskStyle.LABEL,
}

QUASI_STYLE: dict[Category, MaskStyle] = {
    Category.JOB_TITLE: MaskStyle.LABEL,
    Category.EMPLOYMENT_STATUS: MaskStyle.LABEL,
}


def survey_detectors(
    lexicon: Lexicon | None = None,
    *,
    include_builtin_lexicon: bool = True,
    ocr_mode: bool = False,
) -> tuple[Detector, ...]:
    """Assemble the deterministic + organisation + quasi + implication detector stack."""
    base_list: list[Detector] = []
    for cls in DEFAULT_DETECTORS:
        if issubclass(cls, au_ids.ChecksumDetector):
            base_list.append(cls(ocr_mode=ocr_mode))
        else:
            base_list.append(cls())  # type: ignore[call-arg]
    base = tuple(base_list)
    resolved = load_builtin_lexicon() if include_builtin_lexicon else Lexicon()
    if lexicon is not None:
        resolved = resolved.merged(lexicon)
    organisation = default_organisation_detectors(resolved)
    quasi = default_quasi_detectors()
    implication = default_implication_detector()
    return base + organisation + quasi + (implication,)


def survey_policy(
    *,
    salt: bytes | None = None,
    nlp_enabled: bool = True,
    nlp_backend: str = "deberta-v3-small",
    generalise_categories: frozenset[Category] = DEFAULT_GENERALISED,
    ocr_mode: bool = False,
    overlap_mode: str = "longest",
    default_style: MaskStyle = MaskStyle.MASK,
    **overrides: object,
) -> Policy:
    """Build the default policy for government survey free text."""
    styles: dict[Category | str, MaskStyle] = {
        "default": default_style,
        **ATTRIBUTION_STYLE,
        **QUASI_STYLE,
        **HEALTH_DETAIL_STYLE,
        **IMPLICATION_STYLE,
    }
    policy = Policy(
        styles=styles,  # type: ignore[arg-type]
        generalise_categories=generalise_categories,
        salt=salt,
        nlp_enabled=nlp_enabled,
        nlp_backend=nlp_backend,
        ocr_mode=ocr_mode,
        overlap_mode=overlap_mode,
        default_style=default_style,
    )
    if overrides:
        policy = replace(policy, **overrides)  # type: ignore[arg-type]
    return policy


def survey_redactor(
    *,
    lexicon: Lexicon | None = None,
    salt: bytes | None = None,
    nlp_enabled: bool = True,
    nlp_backend: str = "deberta-v3-small",
    generalise_categories: frozenset[Category] = DEFAULT_GENERALISED,
    include_builtin_lexicon: bool = True,
    ocr_mode: bool = False,
    overlap_mode: str = "longest",
    **policy_overrides: object,
) -> Redactor:
    """Construct a :class:`Redactor` wired for government survey free text."""
    return Redactor(
        survey_policy(
            salt=salt,
            nlp_enabled=nlp_enabled,
            nlp_backend=nlp_backend,
            generalise_categories=generalise_categories,
            ocr_mode=ocr_mode,
            overlap_mode=overlap_mode,
            **policy_overrides,
        ),
        detectors=survey_detectors(
            lexicon,
            include_builtin_lexicon=include_builtin_lexicon,
            ocr_mode=ocr_mode,
        ),
    )


__all__ = [
    "ATTRIBUTION_STYLE",
    "DEFAULT_GENERALISED",
    "HEALTH_DETAIL_STYLE",
    "IMPLICATION_STYLE",
    "QUASI_STYLE",
    "survey_detectors",
    "survey_policy",
    "survey_redactor",
]
