"""Arabic text normalization for retrieval.

The same normalizer, with the same config, must run on documents at index time and on
queries at search time. If the two differ, "أحمد" in a query won't match "احمد" in a
document, and nothing reports an error: retrieval just gets worse. Store
`NormalizerConfig.fingerprint()` with the index, and reindex when it changes.

Normalized text is for matching only. Citations and answers always show the original text.

Rules run in a fixed order, chosen so that the result is idempotent
(`normalize(normalize(x)) == normalize(x)`):

1. Unicode composition (NFKC by default, else NFC). Folds presentation-form glyphs
   (common in PDF extractions) to base letters, e.g. "ﻻ" -> "لا", and composes
   letter + combining hamza/madda into one code point, e.g. "ا" + U+0653 -> "آ".
2. Remove invisible format characters (Unicode category Cf): zero-width joiners and
   spaces, bidi marks and embeddings, byte order marks, soft hyphens, Arabic number signs.
3. Strip diacritics (tashkeel, superscript alef, Quranic annotation marks).
4. Strip tatweel (kashida).
5. Letter folds: Persian keheh/yeh, alef variants, alef maqsura, and the off-by-default
   teh marbuta and hamza-carrier folds.
6. Digits: Arabic-Indic and Persian digits (and Arabic decimal/thousands separators) to
   ASCII.
7. Arabic punctuation to ASCII (off by default).
8. Whitespace: collapse runs of spaces and tabs, strip each line, keep line breaks (the
   chunker needs them) but collapse three or more into one blank line.

Latin text is not changed, apart from whitespace and NFKC compatibility forms
(e.g. the "ﬁ" ligature becomes "fi").
"""

import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from functools import cache
from typing import Literal


@dataclass(frozen=True)
class NormalizerConfig:
    # NFKC folds compatibility characters, including Arabic presentation forms. With it
    # off, NFC is still applied so that composed and decomposed input match.
    compatibility_forms: bool = True
    remove_format_chars: bool = True
    strip_diacritics: bool = True
    strip_tatweel: bool = True
    # ک (keheh) -> ك, ی (Farsi yeh) -> ي. Persian keyboards produce these in Arabic text.
    fold_persian_letters: bool = True
    # أ إ آ ٱ ٲ ٳ -> ا
    fold_alef: bool = True
    # ى -> ي
    fold_alef_maqsura: bool = True
    # ة -> ه. Off by default: it merges different words (e.g. a feminine noun and a
    # possessive pronoun ending), so it can hurt precision. Turn it on if your documents
    # are inconsistent about writing teh marbuta.
    fold_teh_marbuta: bool = False
    # ؤ ئ -> ء. Off by default: it merges words that only differ in the hamza seat.
    fold_hamza_carriers: bool = False
    # ٠-٩ and ۰-۹ -> 0-9, ٫ -> ".", ٬ -> ",". On by default so numbers and IDs match
    # whichever digits the writer used.
    ascii_digits: bool = True
    # ، -> ",", ؛ -> ";", ؟ -> "?", ٪ -> "%", ۔ -> "."
    ascii_punctuation: bool = False
    normalize_whitespace: bool = True

    def fingerprint(self) -> str:
        """A short stable hash of the config. Store it with the index."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


DEFAULT_CONFIG = NormalizerConfig()


@cache
def _format_chars() -> str:
    """Every invisible format character (Unicode category Cf).

    Covers zero-width space/joiners, LRM/RLM and the Arabic letter mark, bidi embeddings,
    overrides and isolates, the byte order mark, soft hyphens (common in PDF and HTML
    extractions), and Arabic number signs such as U+0600. Zero-width space (U+200B) is
    category Cf too. ZWNJ carries meaning in Persian; this normalizer is tuned for Arabic.
    Computed once, on first use.
    """
    return "".join(
        chr(c) for c in range(sys.maxunicode + 1) if unicodedata.category(chr(c)) == "Cf"
    )


_DIACRITICS = (
    "".join(chr(c) for c in range(0x0610, 0x061B))  # Quranic honorifics and small marks
    + "".join(chr(c) for c in range(0x064B, 0x0660))  # tashkeel, combining hamza/madda
    + "ٰ"  # superscript alef
    + "".join(chr(c) for c in range(0x06D6, 0x06DD))  # Quranic annotation marks
    + "".join(chr(c) for c in range(0x06DF, 0x06E5))
    + "ۧۨ"
    + "".join(chr(c) for c in range(0x06EA, 0x06EE))
)

_TATWEEL = "ـ"

_PERSIAN_LETTERS = {"ک": "ك", "ی": "ي"}  # ک -> ك, ی -> ي
_ALEF_VARIANTS = {c: "ا" for c in "آأإٱٲٳ"}
_ALEF_MAQSURA = {"ى": "ي"}  # ى -> ي
_TEH_MARBUTA = {"ة": "ه"}  # ة -> ه
_HAMZA_CARRIERS = {"ؤ": "ء", "ئ": "ء"}  # ؤ ئ -> ء

_DIGITS = {
    **{chr(0x0660 + i): str(i) for i in range(10)},  # Arabic-Indic
    **{chr(0x06F0 + i): str(i) for i in range(10)},  # Extended Arabic-Indic (Persian/Urdu)
    "٫": ".",  # Arabic decimal separator
    "٬": ",",  # Arabic thousands separator
}

_PUNCTUATION = {
    "،": ",",  # ،
    "؛": ";",  # ؛
    "؟": "?",  # ؟
    "٪": "%",  # ٪
    "۔": ".",  # ۔
}

# Safety cap for the fixed-point loop in ArabicNormalizer.__call__. Fuzzing has not found
# input that needs more than a few passes.
_MAX_PASSES = 8

_LINE_BREAKS = re.compile(r"\r\n|[\r\x0b\x0c\x1c\x1d\x1e\x85  ]")
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


def _build_table(config: NormalizerConfig) -> dict[int, str | None]:
    table: dict[str, str | None] = {}
    if config.remove_format_chars:
        table.update(dict.fromkeys(_format_chars()))
    if config.strip_diacritics:
        table.update(dict.fromkeys(_DIACRITICS))
    if config.strip_tatweel:
        table[_TATWEEL] = None
    if config.fold_persian_letters:
        table.update(_PERSIAN_LETTERS)
    if config.fold_alef:
        table.update(_ALEF_VARIANTS)
    if config.fold_alef_maqsura:
        table.update(_ALEF_MAQSURA)
    if config.fold_teh_marbuta:
        table.update(_TEH_MARBUTA)
    if config.fold_hamza_carriers:
        table.update(_HAMZA_CARRIERS)
    if config.ascii_digits:
        table.update(_DIGITS)
    if config.ascii_punctuation:
        table.update(_PUNCTUATION)
    return str.maketrans(table)


def _normalize_whitespace(text: str) -> str:
    text = _LINE_BREAKS.sub("\n", text)
    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _EXTRA_BLANK_LINES.sub("\n\n", text)
    return text.strip()


class ArabicNormalizer:
    """Normalizes text with one fixed config. Cheap to call; build it once and reuse it."""

    def __init__(self, config: NormalizerConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._form: Literal["NFKC", "NFC"] = "NFKC" if config.compatibility_forms else "NFC"
        self._table = _build_table(config)

    def __call__(self, text: str) -> str:
        # One pass is not always enough: removing a character (e.g. a tatweel between two
        # marks) can leave marks in non-canonical order, or let a letter and a combining
        # hamza compose on the next pass. Each pass only removes characters or folds them
        # toward canonical forms, so this settles quickly, usually after one or two passes.
        for _ in range(_MAX_PASSES):
            result = self._pass(text)
            if result == text:
                return result
            text = result
        return text

    def _pass(self, text: str) -> str:
        text = unicodedata.normalize(self._form, text)
        text = text.translate(self._table)
        if self.config.normalize_whitespace:
            text = _normalize_whitespace(text)
        return text


_default = ArabicNormalizer()


def normalize(text: str, config: NormalizerConfig | None = None) -> str:
    """Normalize with `config`, or with the defaults when no config is given."""
    if config is None or config == DEFAULT_CONFIG:
        return _default(text)
    return ArabicNormalizer(config)(text)
