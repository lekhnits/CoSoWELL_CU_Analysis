"""
The markers of one research question at once: principal component analysis (PCA) of their session values.

Used for the figures of research questions with more than three markers, where one box plot per marker would
be too many to compare. Each marker is rank-transformed to normal scores first (word rates are skewed and often
zero), so that every marker weighs the same. Sessions missing a marker's text type are left out.
"""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA

from ..standardisation import rank_int


def principal_components_of_markers(sessions: pd.DataFrame, marker_columns: list[str], group: str,
                                    n_components: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Component scores of every session with complete marker values (with its group), the loadings of each
    marker and the share of variance each component explains.
    """
    complete = sessions.dropna(subset=marker_columns + [group])
    values = complete[marker_columns].apply(rank_int)
    pca = PCA(n_components=n_components)
    names = [f"PC{i + 1}" for i in range(n_components)]
    scores = pd.DataFrame(pca.fit_transform(values), index=complete.index, columns=names)
    scores[group] = complete[group]
    loadings = pd.DataFrame(pca.components_.T, index=marker_columns, columns=names)
    explained = pd.Series(pca.explained_variance_ratio_, index=names)
    return scores, loadings, explained
