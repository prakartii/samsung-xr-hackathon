"""select/evidence.py: per-step classifier evidence on cluster pairs (C1)."""

from types import MappingProxyType

import numpy as np
import pytest

from rewind.select.evidence import span_steps, with_step_evidence
from rewind.types import Cluster

VERBS, NOUNS = ("take", "open"), ("cup", "door")
T = np.array([0.5, 1.5, 2.5, 3.5])
# step probabilities: verb take high at steps 1-2, noun cup high at steps 1-2
V = np.array([[0.1, 0.9], [0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
N = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])


def cl(span, dist):
    return Cluster(0, span, (0,), MappingProxyType(dist), 1.0)


def test_span_steps_inside_and_nearest_fallback():
    assert span_steps(T, (1.0, 3.0)).tolist() == [1, 2]
    assert span_steps(T, (1.9, 2.1)).tolist() == [1]  # none inside -> nearest to 2.0 (1.5)


def test_power_zero_leaves_clusters_unchanged():
    c = cl((1.0, 3.0), {("take", "cup"): 1.0})
    assert with_step_evidence([c], T, V, N, VERBS, NOUNS, 0.0) == [c]


def test_pair_scaled_by_geometric_mean_evidence():
    # steps 1, 2: mean v_prob take = (0.8 + 0.6) / 2 = 0.7; mean n_prob cup = (0.9 + 0.5) / 2
    # = 0.7 -> e = sqrt(0.7 * 0.7) = 0.7; power 1: 0.6 * 0.7 = 0.42.
    # open door: v .3, n .3 -> e .3 -> 0.4 * 0.3 = 0.12.
    c = cl((1.0, 3.0), {("take", "cup"): 0.6, ("open", "door"): 0.4})
    (out,) = with_step_evidence([c], T, V, N, VERBS, NOUNS, 1.0)
    assert out.pair_dist[("take", "cup")] == pytest.approx(0.42)
    assert out.pair_dist[("open", "door")] == pytest.approx(0.12)


def test_unknown_class_gets_no_support():
    c = cl((1.0, 3.0), {("other", "cup"): 1.0})
    (out,) = with_step_evidence([c], T, V, N, VERBS, NOUNS, 1.0)
    assert out.pair_dist[("other", "cup")] == 0.0
