"""SlowFast cache entries (features/slowfast.py)."""

import numpy as np
import pytest

from rewind.features.slowfast import find_slowfast, slowfast_cache, slowfast_entry


def test_entry_attaches_center_times_and_float32():
    # 3 features -> centers (16, 32, 48) / 30 s = 0.5333, 1.0667, 1.6
    e = slowfast_entry(np.ones((3, 2304), dtype=np.float64))
    np.testing.assert_allclose(e["t"], [16 / 30, 32 / 30, 48 / 30], atol=1e-12)
    assert e["feats"].dtype == np.float32 and e["feats"].shape == (3, 2304)


@pytest.mark.parametrize(
    "bad",
    [np.ones((3, 2048)), np.ones((0, 2304)), np.ones(2304), np.full((2, 2304), np.nan)],
)
def test_entry_rejects_wrong_shape_or_nonfinite(bad):
    with pytest.raises(ValueError):
        slowfast_entry(bad)


def test_entry_round_trips_through_cache(tmp_path):
    cache = slowfast_cache(tmp_path)
    cache.put("fp", slowfast_entry(np.arange(2 * 2304, dtype=np.float32).reshape(2, 2304)))
    got = cache.get("fp")
    assert got["feats"].shape == (2, 2304)
    assert got["feats"][1, 0] == pytest.approx(2304.0)


# --- lookup across cache roots (local cache, then features shipped in weights/) -------


def _entry(value):
    return slowfast_entry(np.full((2, 2304), value, dtype=np.float32))


def test_find_slowfast_falls_back_to_the_shipped_root(tmp_path):
    # The organizers' clean machine has no local cache; features ship in weights/.
    local, shipped = tmp_path / "cache", tmp_path / "weights" / "released_features"
    slowfast_cache(shipped).put("fp", _entry(1.0))

    got = find_slowfast("fp", [local, shipped])

    assert got is not None and got["feats"][0, 0] == pytest.approx(1.0)


def test_find_slowfast_prefers_the_first_root(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    slowfast_cache(a).put("fp", _entry(1.0))
    slowfast_cache(b).put("fp", _entry(2.0))

    assert find_slowfast("fp", [a, b])["feats"][0, 0] == pytest.approx(1.0)


def test_find_slowfast_missing_everywhere_is_none(tmp_path):
    assert find_slowfast("fp", [tmp_path / "a", None, tmp_path / "b"]) is None
