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
    # The kit spells it "ski" (its first example file said "skiing"; PROGRESS Q10).
    v = load_vocab()

    assert "ski" in v.labels
    assert "skiing" not in v.labels
    assert "cooking" in v.labels


def test_load_vocab_keys_keep_kit_order_so_index_i_is_class_i():
    # ActionFormer's class index i is kit key i (EPIC class id order, `other` last;
    # checked against EPIC_100_{verb,noun}_classes.csv by tools/build_vn_prior.py).
    v = load_vocab()

    assert v.verb_keys[:4] == ("take", "put", "wash", "open")
    assert v.verb_keys[-1] == "other" and v.noun_keys[-1] == "other"
    assert set(v.verb_keys) == v.verbs and set(v.noun_keys) == v.nouns
    assert (len(v.verb_keys), len(v.noun_keys)) == (98, 301)
