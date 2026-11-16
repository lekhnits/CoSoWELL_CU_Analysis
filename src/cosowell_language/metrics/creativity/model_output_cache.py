"""
Caches for the slow steps of the creativity metrics (data/cache/creativity/).

Transformer outputs are stored per text, so a model only runs on texts it has not seen.
Without --run-models, texts missing from a cache get None (NaN metrics) instead of
loading the model, so the metrics can be recomputed in minutes from the caches.
"""

from __future__ import annotations

import joblib

from .creativity_settings import CACHE


def cached(name: str, compute):
    """A whole object (e.g. the fitted LDA model), computed once and then read from disk."""
    path = CACHE / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    result = compute()
    joblib.dump(result, path)
    return result


def cached_per_text(name: str, keys: list, compute, may_compute: bool) -> list:
    """
    One cached value per key (a text or a sentence). `compute(new_keys) -> values` runs only for
    keys not cached yet, and only when may_compute is true; otherwise missing keys give None.
    """
    path = CACHE / f"map_{name}.joblib"
    store = joblib.load(path) if path.exists() else {}
    missing = list(dict.fromkeys(key for key in keys if key not in store))
    if missing and not may_compute:
        print(f"[cache] {name}: {len(missing)} item(s) not cached -> NaN (use --run-models to compute)")
        return [store.get(key) for key in keys]
    if missing:
        print(f"[cache] {name}: computing {len(missing)} new item(s)")
        store.update(zip(missing, compute(missing)))
        joblib.dump(store, path)
    return [store[key] for key in keys]
