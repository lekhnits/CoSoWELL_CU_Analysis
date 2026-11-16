"""Vector and sequence helpers used by several creativity families (embeddings are compared by cosine)."""

from __future__ import annotations

import numpy as np


def unit_norm(vectors):
    """Each row scaled to length 1, so that a dot product of two rows is their cosine similarity."""
    vectors = np.asarray(vectors, dtype=float)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def entropy(counts):
    """Shannon entropy in bits of a set of counts (zeros ignored)."""
    counts = np.asarray(list(counts), dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def slope(values):
    """Linear trend of a sequence, with its positions spread over [0, 1] (NaN for fewer than 3 values)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.nan
    position = np.linspace(0, 1, len(values))
    return float(np.polyfit(position, values, 1)[0])


def mean_pairwise_distance(vectors):
    """Mean cosine distance between all pairs of rows."""
    if len(vectors) < 2:
        return np.nan
    similarity = unit_norm(vectors) @ unit_norm(vectors).T
    upper_triangle = np.triu_indices(len(vectors), 1)          # every pair once, without the diagonal
    return float(1 - similarity[upper_triangle].mean())
