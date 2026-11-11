"""Download ActionFormer's released EPIC-100 features (epic_kitchens.tar.gz, 6.53 GB).

    python tools/fetch_released_features.py --out <dir>/epic_kitchens.tar.gz

Source: the Box mirror listed in ActionFormer's README (the Google Drive copy throttles
to ~25 KiB/s). Box serves HTTP range requests, so the file is fetched as parallel
byte ranges (one connection gets ~0.5 MiB/s) and any existing prefix of --out (e.g.
from an interrupted run) is kept. Finishes with the README's MD5; a mismatch deletes
nothing but exits non-zero, so the file can be inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = (
    "https://uwmadison.box.com/index.php?rm=box_download_shared_file"
    "&shared_name=vdha47qnce6jhqktz9g4mq1gc40w82yj&file_id=f_888429773664"
)
SIZE = 6_527_971_289  # bytes, from the Box share page
MD5 = "add9803756afd9a023bc9a9c547e0229"  # ActionFormer README
CHUNK = 64 << 20  # 64 MiB per range request


def fetch_range(out: Path, start: int, end: int, retries: int = 8) -> None:
    """Write bytes [start, end] (inclusive) of URL at offset `start` of `out`."""
    pos = start
    for _ in range(retries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as resp, out.open("r+b") as f:
                if resp.status != 206:
                    raise OSError(f"expected 206 Partial Content, got {resp.status}")
                f.seek(pos)
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    pos += len(chunk)
            if pos > end:
                return
        except OSError as e:
            print(f"  range {start}-{end}: {e}; retrying from {pos}", flush=True)
    raise OSError(f"range {start}-{end} failed after {retries} attempts")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    # Progress lives in a sidecar: line 1 = size of the prefix kept from an earlier,
    # non-ranged download; then one line per finished range start.
    state = args.out.with_name(args.out.name + ".done")
    if state.exists():
        lines = state.read_text(encoding="utf-8").split()
        have, finished = int(lines[0]), {int(x) for x in lines[1:]}
    else:
        have = args.out.stat().st_size if args.out.exists() else 0
        finished = set()
        state.write_text(f"{have}\n", encoding="utf-8")
    if have > SIZE:
        sys.exit(f"{args.out} is larger than the archive ({have} > {SIZE})")
    with args.out.open("ab") as f:  # create if missing, then extend to full size for seeks
        if f.tell() < SIZE:
            f.truncate(SIZE)
    ranges = [(s, min(s + CHUNK, SIZE) - 1) for s in range(have, SIZE, CHUNK)]
    todo = [(s, e) for s, e in ranges if s not in finished]
    print(
        f"prefix {have / 2**30:.2f} GiB; {len(todo)} of {len(ranges)} ranges to fetch", flush=True
    )

    def job(span: tuple[int, int]) -> None:
        fetch_range(args.out, *span)
        with state.open("a", encoding="utf-8") as f:
            f.write(f"{span[0]}\n")

    with ThreadPoolExecutor(args.workers) as pool:
        for i, fut in enumerate([pool.submit(job, r) for r in todo], 1):
            fut.result()
            if i % 8 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ranges", flush=True)
    got = md5_file(args.out)
    if got != MD5:
        sys.exit(f"MD5 MISMATCH: {got} != {MD5} ({args.out})")
    print(f"md5 ok: {args.out}", flush=True)
    os.utime(args.out)


if __name__ == "__main__":
    main()
