"""Kit class vocabulary: 16 video labels, 98 verbs, 301 nouns (EPIC-100 keys).

Inference must run without ``kit/`` (organizers re-run run.sh in isolation), so the
class list ships inside the package as ``rewind/resources/classes.json``, a byte-for-
byte copy of ``kit/supported_classes.json`` (a test enforces the match).
"""

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources


@dataclass(frozen=True)
class Vocab:
    version: str
    labels: frozenset[str]
    verbs: frozenset[str]
    nouns: frozenset[str]
    # Kit order: key i is EPIC class id i (`other` last), i.e. a detector's class index.
    verb_keys: tuple[str, ...]
    noun_keys: tuple[str, ...]


def vocab_from_dict(data: dict) -> Vocab:
    """Build a ``Vocab`` from the parsed classes JSON. Pure."""
    return Vocab(
        version=data["version"],
        labels=frozenset(data["labels"]),
        verbs=frozenset(data["verbs"]),
        nouns=frozenset(data["nouns"]),
        verb_keys=tuple(data["verbs"]),
        noun_keys=tuple(data["nouns"]),
    )


@cache
def load_vocab() -> Vocab:
    """Load the bundled class list once per process."""
    text = resources.files("rewind").joinpath("resources/classes.json").read_text("utf-8")
    return vocab_from_dict(json.loads(text))
