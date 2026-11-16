"""Summary tables: hit counts per finding, and sample sizes."""

from __future__ import annotations

import pandas as pd

from .hit_replication import ALPHA

def count_tests_and_hits(all_tests: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """Number of tests, nominal hits, FDR hits and robust hits per finding and term."""
    counts = all_tests.groupby(["finding", "term"], sort=False).agg(
        tests=("p", "size"), nominal_p05=("p", lambda p: (p < ALPHA).sum()),
        fdr_hits=("q", lambda q: (q < ALPHA).sum()), holm_hits=("p_holm", lambda p: (p < ALPHA).sum()),
        fdr_hits_all_findings=("q_all_findings", lambda q: (q < ALPHA).sum()),
        max_abs_beta=("beta", lambda b: b.abs().max()))
    robust = hits[hits["verdict"].eq("robust")]
    robust_all_findings = robust[robust["q_all_findings"] < ALPHA]
    counts["robust_hits"] = hits_per_finding_and_term(robust, counts.index)
    counts["robust_hits_all_findings"] = hits_per_finding_and_term(robust_all_findings, counts.index)
    return counts.reset_index()


def hits_per_finding_and_term(hits: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Number of hits for each (finding, term) of `index`; 0 where there is none."""
    if len(hits) == 0:
        return pd.Series(0, index=index, dtype=int)
    per_term = hits.groupby(["finding", "term"]).size()
    return per_term.reindex(index).fillna(0).astype(int)


def sample_summary(session_scores: pd.DataFrame, scored_texts: pd.DataFrame) -> pd.Series:
    """Sessions, participants, narratives and a few descriptive numbers of the analysed sample."""
    repeat = session_scores.groupby("id").size().gt(1)
    return pd.Series({
        "sessions": len(session_scores), "participants": session_scores["id"].nunique(),
        "participants_two_sessions": int(repeat.sum()),
        "narratives": len(scored_texts),
        "spearman_loneliness_isolation": session_scores[["loneliness", "isolation_index"]].corr("spearman").iloc[0, 1],
        **{f"sessions_with_{m}": int(session_scores[m].notna().sum()) for m in
           ["loneliness", "isolation_index", "health"]},
    })
