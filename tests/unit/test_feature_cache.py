"""Unit tests for the versioned feature cache (PROGRESS.md P3, ARCHITECTURE.md §8)."""

import hashlib

import numpy as np
import pytest

from rewind.features.cache import FINGERPRINT_CHUNK, FeatureCache, content_fingerprint


def _write(path, data: bytes):
    path.write_bytes(data)
    return path


# --- content_fingerprint ------------------------------------------------------------


def test_fingerprint_small_file_matches_spec(tmp_path):
    # Spec: sha1(size as 8-byte LE + first 1 MiB + last 1 MiB). For a 5-byte file
    # both chunks are the whole file: sha1(b"\x05\0\0\0\0\0\0\0" + b"hello" + b"hello").
    f = _write(tmp_path / "a.mp4", b"hello")
    expected = hashlib.sha1((5).to_bytes(8, "little") + b"hello" + b"hello").hexdigest()
    assert content_fingerprint(f) == expected


def test_fingerprint_renamed_file_is_identical(tmp_path):
    data = bytes(range(256)) * 100
    a = _write(tmp_path / "P01_11.mp4", data)
    b = _write(tmp_path / "Ski_-Z5a4UQmZLs.mp4", data)
    assert content_fingerprint(a) == content_fingerprint(b)


def test_fingerprint_changed_last_byte_differs(tmp_path):
    # 3 MiB file: the last byte is inside the hashed tail chunk.
    data = bytearray(3 * FINGERPRINT_CHUNK)
    a = _write(tmp_path / "a.mp4", bytes(data))
    data[-1] = 1
    b = _write(tmp_path / "b.mp4", bytes(data))
    assert content_fingerprint(a) != content_fingerprint(b)


def test_fingerprint_same_ends_different_size_differs(tmp_path):
    # Same head and tail bytes (all zero), sizes 3 MiB vs 3 MiB + 1: only the size
    # term separates them.
    a = _write(tmp_path / "a.mp4", bytes(3 * FINGERPRINT_CHUNK))
    b = _write(tmp_path / "b.mp4", bytes(3 * FINGERPRINT_CHUNK + 1))
    assert content_fingerprint(a) != content_fingerprint(b)


# --- FeatureCache -------------------------------------------------------------------


def test_cache_put_then_get_returns_same_arrays(tmp_path):
    cache = FeatureCache(tmp_path, "flow", "farneback-v1")
    flow = np.array([0.5, -1.25, 3.0])
    t = np.array([0.0, 0.25, 0.5])
    cache.put("abc", {"flow": flow, "t": t})

    got = cache.get("abc")
    assert set(got) == {"flow", "t"}
    np.testing.assert_array_equal(got["flow"], flow)
    np.testing.assert_array_equal(got["t"], t)


def test_cache_missing_entry_is_miss(tmp_path):
    assert FeatureCache(tmp_path, "flow", "v1").get("nope") is None


def test_cache_version_mismatch_is_miss(tmp_path):
    FeatureCache(tmp_path, "flow", "farneback-v1").put("abc", {"x": np.zeros(3)})
    assert FeatureCache(tmp_path, "flow", "farneback-v2").get("abc") is None


def test_cache_version_mismatch_recomputes_and_overwrites(tmp_path):
    FeatureCache(tmp_path, "flow", "v1").put("abc", {"x": np.zeros(2)})
    new = FeatureCache(tmp_path, "flow", "v2")

    got = new.get_or_compute("abc", lambda: {"x": np.ones(2)})

    np.testing.assert_array_equal(got["x"], np.ones(2))
    np.testing.assert_array_equal(new.get("abc")["x"], np.ones(2))


def test_cache_hit_does_not_call_compute(tmp_path):
    cache = FeatureCache(tmp_path, "flow", "v1")
    cache.put("abc", {"x": np.arange(3.0)})

    def boom():
        raise AssertionError("compute must not run on a cache hit")

    np.testing.assert_array_equal(cache.get_or_compute("abc", boom)["x"], np.arange(3.0))


def test_cache_features_are_separate_namespaces(tmp_path):
    FeatureCache(tmp_path, "flow", "v1").put("abc", {"x": np.zeros(1)})
    assert FeatureCache(tmp_path, "hands", "v1").get("abc") is None


def test_cache_renamed_video_hits_cache(tmp_path):
    # The cache key is the content fingerprint, so renaming the MP4 is a cache hit.
    data = b"fake mp4 bytes" * 1000
    original = _write(tmp_path / "P01_11.mp4", data)
    renamed = _write(tmp_path / "renamed.mp4", data)
    cache = FeatureCache(tmp_path / "cache", "clip", "v1")
    cache.put(content_fingerprint(original), {"emb": np.eye(2)})

    hit = cache.get(content_fingerprint(renamed))

    assert hit is not None
    np.testing.assert_array_equal(hit["emb"], np.eye(2))


class _Unserializable:
    def __array__(self, dtype=None, copy=None):
        raise RuntimeError("simulated failure while writing")


def test_cache_failed_write_leaves_no_partial_file(tmp_path):
    cache = FeatureCache(tmp_path, "flow", "v1")
    with pytest.raises(RuntimeError):
        cache.put("abc", {"ok": np.zeros(3), "bad": _Unserializable()})

    assert cache.get("abc") is None
    leftovers = list(cache.dir.glob("*")) if cache.dir.exists() else []
    assert leftovers == []


def test_cache_failed_overwrite_keeps_previous_entry(tmp_path):
    cache = FeatureCache(tmp_path, "flow", "v1")
    cache.put("abc", {"x": np.arange(3.0)})
    with pytest.raises(RuntimeError):
        cache.put("abc", {"x": _Unserializable()})

    np.testing.assert_array_equal(cache.get("abc")["x"], np.arange(3.0))


def test_cache_put_succeeds_when_meta_index_cannot_be_updated(tmp_path):
    # Parallel cache workers can collide on meta.json (on Windows, os.replace fails
    # while another process has it open). meta.json is only an index: the entry must
    # still be stored and readable. Simulate an unwritable index with a directory.
    cache = FeatureCache(tmp_path, "flow", "v1")
    (cache.dir / "meta.json").mkdir(parents=True)

    with pytest.warns(RuntimeWarning):
        cache.put("abc", {"x": np.arange(3.0)})

    np.testing.assert_array_equal(cache.get("abc")["x"], np.arange(3.0))


def test_cache_reserved_key_rejected(tmp_path):
    with pytest.raises(ValueError):
        FeatureCache(tmp_path, "flow", "v1").put("abc", {"__cache_version__": np.zeros(1)})
