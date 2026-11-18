"""Turns raw text into words, and word-level BIO predictions into entity spans.

Pure Python with no model dependency, so it is unit-tested on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Words with internal hyphens, apostrophes or dots ("5-fluorouracil", "0.5"), or single punctuation marks.
_WORD_RE = re.compile(r"[^\W_]+(?:[-'.][^\W_]+)*|\S")
_GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "kappa", "mu", "omega"}


@dataclass(frozen=True)
class Word:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    start: int
    end: int
    score: float


def _keep_hyphen(left: str, right: str) -> bool:
    """Mirrors BC5CDR: chemical names like L-dopa, 5-FU, MK-801 or alpha-methyldopa stay one word,
    while ordinary compounds such as lithium-induced or dose-dependent are split around the hyphen."""
    return len(left) <= 2 or left.lower() in _GREEK or any(c.isdigit() for c in left + right)


def split_words(text: str) -> list[Word]:
    words: list[Word] = []
    for match in _WORD_RE.finditer(text):
        token, offset = match.group(), match.start()
        parts = token.split("-")
        start = 0  # start of the word being built, relative to the token
        position = len(parts[0])
        for left, right in zip(parts, parts[1:], strict=False):
            if not _keep_hyphen(left, right):
                words.append(Word(token[start:position], offset + start, offset + position))
                words.append(Word("-", offset + position, offset + position + 1))
                start = position + 1
            position += 1 + len(right)
        words.append(Word(token[start:], offset + start, offset + len(token)))
    return words


def decode_bio(text: str, words: list[Word], labels: list[str], scores: list[float]) -> list[Entity]:
    """Groups consecutive B-X / I-X words into entities.

    An I-X that does not continue an X entity starts a new one (lenient decoding),
    since dropping it would hide a mention the model did detect. The entity score
    is the mean of its word scores.
    """
    if not (len(words) == len(labels) == len(scores)):
        raise ValueError("words, labels and scores must have the same length")

    entities: list[Entity] = []
    current: tuple[str, int, int, list[float]] | None = None  # label, start, end, scores

    def close() -> None:
        if current is not None:
            label, start, end, word_scores = current
            entities.append(Entity(text[start:end], label, start, end, sum(word_scores) / len(word_scores)))

    for word, tag, score in zip(words, labels, scores, strict=True):
        prefix, _, entity_type = tag.partition("-")
        if prefix == "I" and current is not None and current[0] == entity_type:
            current = (entity_type, current[1], word.end, [*current[3], score])
        elif prefix in ("B", "I"):
            close()
            current = (entity_type, word.start, word.end, [score])
        else:
            close()
            current = None
    close()
    return entities
