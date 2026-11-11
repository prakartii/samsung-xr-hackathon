"""Versioned on-disk feature cache (ARCHITECTURE.md §8).

Layout::

    cache/<feature>/<fingerprint>.npz   # the arrays, plus the version they were made with
    cache/<feature>/meta.json           # {"version": ..., "videos": {fingerprint: {...}}}

Entries are keyed by a **content fingerprint** of the MP4, never by `video_id`, so a
renamed file still hits the cache and inference never depends on the filename
(CLAUDE.md §6.1). Each entry stores the extractor version it was built with; a
reader asking for a different version gets a miss and recomputes (CLAUDE.md §6.8).
All writes are atomic: a temp file in the same directory, then `os.replace`.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

# Bytes hashed from each end of the file. 1 MiB covers the MP4 header (moov atom is
# usually at one end) plus enough payload to separate different encodes; together
# with the exact size, collisions between different videos are not a practical risk.
FINGERPRINT_CHUNK = 1 << 20

# Reserved key inside each .npz that records the extractor version.
_VERSION_KEY = "__cache_version__"
_META_NAME = "meta.json"


def content_fingerprint(path: Path) -> str:
    """sha1 over (file size as 8-byte little-endian, first 1 MiB, last 1 MiB), hex.

    Fast on multi-GB videos (reads at most 2 MiB) and independent of the filename.
    """
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(size.to_bytes(8, "little"))
    with path.open("rb") as f:
        h.update(f.read(FINGERPRINT_CHUNK))
        f.seek(max(0, size - FINGERPRINT_CHUNK))
        h.update(f.read(FINGERPRINT_CHUNK))
    return h.hexdigest()


def _atomic_write_bytes(target: Path, write: Callable[[Path], None]) -> None:
    """Call `write(tmp_path)`, then rename tmp onto `target`. On error, remove tmp."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=target.suffix)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write(tmp)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class FeatureCache:
    """Cache for one feature type (e.g. "flow") at one extractor version."""

    def __init__(self, root: Path, feature: str, version: str) -> None:
        if not version:
            raise ValueError("cache version must be a non-empty string")
        self.dir = Path(root) / feature
        self.feature = feature
        self.version = version

    def _entry(self, fingerprint: str) -> Path:
        return self.dir / f"{fingerprint}.npz"

    def get(self, fingerprint: str) -> dict[str, np.ndarray] | None:
        """Arrays for `fingerprint`, or None if absent or built by another version."""
        path = self._entry(fingerprint)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as npz:
            if _VERSION_KEY not in npz.files or str(npz[_VERSION_KEY]) != self.version:
                return None
            return {k: npz[k] for k in npz.files if k != _VERSION_KEY}

    def put(self, fingerprint: str, arrays: Mapping[str, np.ndarray]) -> None:
        """Store `arrays` for `fingerprint` atomically, tagged with this version."""
        if _VERSION_KEY in arrays:
            raise ValueError(f"{_VERSION_KEY!r} is reserved")

        def write(tmp: Path) -> None:
            # Runs after the temp file exists, so any failure here (bad array, full
            # disk) is cleaned up by _atomic_write_bytes and never leaves an entry.
            payload = {k: np.asarray(v) for k, v in arrays.items()}
            payload[_VERSION_KEY] = np.asarray(self.version)
            with tmp.open("wb") as f:
                np.savez(f, **payload)

        _atomic_write_bytes(self._entry(fingerprint), write)
        self._record(fingerprint, sorted(arrays))

    def get_or_compute(
        self, fingerprint: str, compute: Callable[[], Mapping[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """Cached arrays if valid, else `compute()`, store it, and return it."""
        hit = self.get(fingerprint)
        if hit is not None:
            return hit
        arrays = compute()
        self.put(fingerprint, arrays)
        return {k: np.asarray(v) for k, v in arrays.items()}

    def _record(self, fingerprint: str, keys: list[str]) -> None:
        """Update meta.json (informational index). A version change resets the index.

        Best effort: `get` never reads the index, so when parallel workers collide on
        it (on Windows `os.replace` fails while another process has the file open),
        the entry is already safely stored and we only warn.
        """
        meta_path = self.dir / _META_NAME
        try:
            meta: dict = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("version") != self.version:
                meta = {"version": self.version, "videos": {}}
            meta["videos"][fingerprint] = {"keys": keys}
            text = json.dumps(meta, indent=1, sort_keys=True)
            _atomic_write_bytes(meta_path, lambda tmp: tmp.write_text(text, encoding="utf-8"))
        except (OSError, ValueError) as e:
            warnings.warn(f"cache index {meta_path} not updated: {e}", RuntimeWarning, stacklevel=3)
