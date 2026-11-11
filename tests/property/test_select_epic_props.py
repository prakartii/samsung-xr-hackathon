"""select_epic output always satisfies the highlight rules (review of #16)."""

import dataclasses
from itertools import pairwise
from types import MappingProxyType

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from rewind.config import load_config
from rewind.select.epic import select_epic
from rewind.types import Cluster
from tests.conftest import DEFAULT_CONFIG

CFG = load_config(DEFAULT_CONFIG)
PAIRS = [(v, n) for v in "abcd" for n in "xyz"]


@st.composite
def case(draw):
    T = draw(st.floats(5.0, 600.0))
    clusters = []
    for i in range(draw(st.integers(0, 25))):
        s = draw(st.floats(0.0, T - 0.2))
        e = draw(st.floats(s + 0.1, min(T, s + 20.0)))
        pair = draw(st.sampled_from(PAIRS))
        conf = draw(st.floats(0.1, 1.0))
        clusters.append(Cluster(i, (s, e), (i,), MappingProxyType({pair: 1.0}), conf))
    share = draw(st.sampled_from([0.0, 0.125, 0.3]))
    mode = draw(st.sampled_from(["expected", "dedup"]))
    profile = None
    if draw(st.booleans()):
        t = np.linspace(0.0, T, 20)
        profile = (t, np.array(draw(st.lists(st.floats(0.0, 1.0), min_size=20, max_size=20))))
    return T, clusters, share, mode, profile


@settings(max_examples=300, deadline=None)
@given(case())
def test_select_epic_output_obeys_the_window_rules(c):
    T, clusters, share, mode, profile = c
    cfg = dataclasses.replace(CFG, hedge=dataclasses.replace(CFG.hedge, share=share))

    segs, values = select_epic(clusters, T, cfg, mode=mode, profile=profile)

    assert len(values) == len(segs)
    assert all(0.0 <= s < e <= T for s, e in segs)
    assert all(e - s >= min(4.05, T) - 1e-9 for s, e in segs)
    assert all(s1 - e0 >= cfg.post.merge_gap - 1e-9 for (_, e0), (s1, _) in pairwise(segs))
    total = sum(e - s for s, e in segs)
    assert total <= max(cfg.select.budget_frac * T, min(4.05, T)) + 1e-9
