"""Video label accuracy (ARCHITECTURE.md §9.4). Pure."""

from collections.abc import Mapping


def label_accuracy(pred: Mapping[str, str], gt: Mapping[str, str]) -> float:
    """Share of GT videos whose predicted label matches exactly; missing = wrong."""
    if not gt:
        return 0.0
    return sum(pred.get(vid) == name for vid, name in gt.items()) / len(gt)
