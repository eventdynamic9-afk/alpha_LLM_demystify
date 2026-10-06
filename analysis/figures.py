"""Figures (§12.8): forest plots of claim precision by model and condition, verdict composition, and
the calibration confusion matrix.  Colors follow a neutral, colour-blind-safe palette."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2f6db5", "#d1495b", "#3d9970", "#edae49", "#7a5195", "#00798c", "#8c564b", "#6c757d"]


def forest_plot(summary: pd.DataFrame, path: str | Path, title: str = "Claim precision by model and condition") -> Path:
    """``summary``: rows with model, condition, cp_micro, cp_micro_ci (from metrics.precision_summary)."""
    s = summary.dropna(subset=["cp_micro"]).sort_values(["condition", "model"])
    models = list(dict.fromkeys(s["model"]))
    conds = list(dict.fromkeys(s["condition"]))
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(s) + 1.5))
    y = np.arange(len(s))[::-1]
    for i, (_, r) in enumerate(s.iterrows()):
        lo, hi = r["cp_micro_ci"] if isinstance(r["cp_micro_ci"], (list, tuple)) else (np.nan, np.nan)
        c = PALETTE[models.index(r["model"]) % len(PALETTE)]
        ax.plot([lo, hi], [y[i], y[i]], color=c, lw=1.6)
        ax.plot(r["cp_micro"], y[i], "o", color=c, ms=5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['condition']} · {r['model']}" for _, r in s.iterrows()], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("claim precision (decidable claims), 95% cluster-bootstrap CI")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    del conds
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def verdict_composition(C: pd.DataFrame, path: str | Path) -> Path:
    order = ["SUPPORTED", "REFUTED", "UNRESOLVED", "UNVERIFIABLE", "AMBIGUOUS"]
    tab = pd.crosstab(C["condition"], C["verdict"], normalize="index").reindex(columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(tab) + 1.5))
    left = np.zeros(len(tab))
    for i, v in enumerate(order):
        ax.barh(tab.index, tab[v], left=left, color=PALETTE[i], label=v.lower())
        left += tab[v].values
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of claims")
    ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    p = Path(path)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def confusion_heatmap(confusion: dict, path: str | Path) -> Path:
    labels = ["SUPPORTED", "REFUTED", "UNRESOLVED"]
    m = np.array([[confusion.get(pred, {}).get(true, 0) for pred in labels] for true in labels])
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.imshow(m, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", color="#111111" if m[i, j] < m.max() / 2 else "white")
    ax.set_xticks(range(3))
    ax.set_xticklabels([l.lower() for l in labels], fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels([l.lower() for l in labels], fontsize=8)
    ax.set_xlabel("verifier verdict")
    ax.set_ylabel("planted truth")
    fig.tight_layout()
    p = Path(path)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p
