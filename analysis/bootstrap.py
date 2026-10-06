"""Two-stage cluster bootstrap (§12.3): resample formulas (clusters), then rationales within formulas.

Clustered resampling respects the nesting claims-in-rationales-in-formulas (Miller 2024); for paired
Arm-B contrasts the cluster is the base formula so a K variant and its SA/SP/NL variants move together.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def two_stage_indices(df: pd.DataFrame, cluster: str, sub: str | None, rng: np.random.Generator) -> np.ndarray:
    groups = df.groupby(cluster, sort=False).indices
    keys = list(groups)
    pick = rng.integers(0, len(keys), len(keys))
    out = []
    for k in pick:
        idx = groups[keys[k]]
        if sub is None:
            out.append(idx)
            continue
        subs = df.iloc[idx].groupby(sub, sort=False).indices
        skeys = list(subs)
        spick = rng.integers(0, len(skeys), len(skeys))
        out.extend(idx[subs[skeys[j]]] for j in spick)
    return np.concatenate(out) if out else np.array([], dtype=int)


def cluster_bootstrap(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], cluster: str = "formula_id",
                      sub: str | None = "rationale_id", n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> dict:
    df = df.reset_index(drop=True)
    est = float(stat(df))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = two_stage_indices(df, cluster, sub, rng)
        if len(idx) == 0:
            continue
        v = stat(df.iloc[idx])
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    a = (1 - level) / 2
    if len(vals) < 10:
        return {"estimate": est, "ci": [float("nan"), float("nan")], "n_boot": len(vals)}
    return {"estimate": est, "ci": [float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))], "n_boot": len(vals),
            "se": float(vals.std(ddof=1)), "boot": vals}


def bootstrap_difference(df: pd.DataFrame, group_col: str, a: str, b: str, stat: Callable[[pd.DataFrame], float],
                         cluster: str = "formula_id", sub: str | None = "rationale_id", n_boot: int = 1000,
                         seed: int = 0, paired: bool = False) -> dict:
    """stat(group a) - stat(group b) with clusters resampled jointly (paired) or per group."""
    da, db = df[df[group_col] == a], df[df[group_col] == b]
    est = float(stat(da) - stat(db))
    rng = np.random.default_rng(seed)
    vals = []
    if paired:
        both = pd.concat([da, db]).reset_index(drop=True)
        groups = both.groupby(cluster).indices
        keys = list(groups)
        for _ in range(n_boot):
            pick = rng.integers(0, len(keys), len(keys))
            s = both.iloc[np.concatenate([groups[keys[k]] for k in pick])]
            v = stat(s[s[group_col] == a]) - stat(s[s[group_col] == b])
            if np.isfinite(v):
                vals.append(v)
    else:
        da, db = da.reset_index(drop=True), db.reset_index(drop=True)
        for _ in range(n_boot):
            ia = two_stage_indices(da, cluster, sub, rng)
            ib = two_stage_indices(db, cluster, sub, rng)
            v = stat(da.iloc[ia]) - stat(db.iloc[ib])
            if np.isfinite(v):
                vals.append(v)
    vals = np.array(vals)
    if len(vals) < 10:
        return {"estimate": est, "ci": [float("nan")] * 2, "p_greater": float("nan"), "p_less": float("nan")}
    return {"estimate": est, "ci": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
            "p_greater": float((vals <= 0).mean()), "p_less": float((vals >= 0).mean()),
            "p_two_sided": float(min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))), "n_boot": len(vals)}


def micro_precision(df: pd.DataFrame) -> float:
    d = df[df["decidable"]]
    return float(d["supported"].mean()) if len(d) else float("nan")


def macro_precision(df: pd.DataFrame) -> float:
    d = df[df["decidable"]]
    if not len(d):
        return float("nan")
    return float(d.groupby("rationale_id")["supported"].mean().mean())
