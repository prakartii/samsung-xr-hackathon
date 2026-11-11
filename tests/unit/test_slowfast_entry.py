"""SlowFast cache entries (features/slowfast.py)."""

import numpy as np
import pytest

from rewind.features.slowfast import slowfast_cache, slowfast_entry


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
