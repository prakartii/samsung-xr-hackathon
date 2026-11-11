"""The bundled class list must be the kit's, so inference can run without kit/."""

from rewind.vocab import load_vocab
from tests.conftest import KIT_DIR, REPO_ROOT


def test_bundled_classes_are_identical_to_kit_file():
    bundled = REPO_ROOT / "code" / "rewind" / "resources" / "classes.json"

    assert bundled.read_bytes() == (KIT_DIR / "supported_classes.json").read_bytes()


def test_load_vocab_has_kit_counts():
    # Kit README §1: 16 labels, 98 verbs, 301 nouns.
    v = load_vocab()

    assert (len(v.labels), len(v.verbs), len(v.nouns)) == (16, 98, 301)


def test_load_vocab_label_names_use_kit_spelling():
    # The kit's example file says "skiing", but the vocabulary has "ski" (PROGRESS Q10).
    v = load_vocab()

    assert "ski" in v.labels
    assert "skiing" not in v.labels
    assert "cooking" in v.labels
