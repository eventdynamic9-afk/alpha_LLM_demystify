"""Planted-claims calibration set (§10.7, Appendix F): 300 synthetic (formula/signal, claim, truth)
triples whose truth is known by construction.

Acceptance: >= 99% accuracy on decidable static (C1/C6) items, >= 95% on statistical (C2) items with
zero sign errors (SUPPORTED <-> REFUTED confusions).  The confusion matrix is published.

    python -m verify calibrate --out runs/calibration
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from data.synthetic import synthetic_panel
from dsl import parse, to_alpha101, to_program, to_qlib, walk
from pools.library import base_set

from .context import VerificationContext
from .dispatcher import verify_claim
from .verdicts import REFUTED, SUPPORTED, UNRESOLVED

STATIC, STATISTICAL = "static", "statistical"


@dataclass
class Planted:
    item_id: str
    recipe: str
    category: str
    claim: dict
    expected: str
    formula: str | None = None           # Qlib DSL (or alpha101 when dialect says so)
    dialect: str = "qlib"
    signal_key: str | None = None        # key of a planted signal array (statistical items)
    meta: dict = field(default_factory=dict)


def calibration_context(n_stocks: int = 120, n_days: int = 900, seed: int = 11, fast: bool = False) -> VerificationContext:
    p = synthetic_panel(n_stocks, n_days, seed=seed)
    split = str(p.dates[int(n_days * 0.6)])
    end = str(p.dates[-1])
    return VerificationContext(p, windows={"train": (str(p.dates[0]), split), "test": (split, end)}, fast=fast)


# ------------------------------------------------------------------------------------- signals
def _normal_scores(x: np.ndarray, member: np.ndarray) -> np.ndarray:
    pct = pd.DataFrame(np.where(member & np.isfinite(x), x, np.nan)).rank(axis=1, pct=True).to_numpy()
    n = np.isfinite(pct).sum(axis=1, keepdims=True)
    adj = (pct * n - 0.5) / n                       # (rank - 0.5)/n
    return sps.norm.ppf(adj)


def _pearson_for_spearman(rho_s: float) -> float:
    return 2.0 * np.sin(np.pi * rho_s / 6.0)


def planted_signals(ctx: VerificationContext, refs=("STREV_5d", "VOL_20d", "ABNVOL_60d", "CLV", "HIGH52"),
                    seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    m = ctx.panel.member
    out = {}
    for ref in refs:
        z = _normal_scores(ctx.references.signal(ref), m)
        for target in (0.6, 0.2, 0.0):
            r = _pearson_for_spearman(target)
            noise = rng.standard_normal(z.shape)
            out[f"mix_{ref}_{target}"] = np.where(np.isfinite(z), r * z + np.sqrt(1 - r * r) * noise, np.nan)
        noise = rng.standard_normal(z.shape)
        # residualize noise on z cross-sectionally (exact orthogonality each day)
        res = np.full_like(z, np.nan)
        for t in range(z.shape[0]):
            ok = np.isfinite(z[t])
            if ok.sum() > 5:
                zz, nn = z[t, ok], noise[t, ok]
                b = (zz - zz.mean()) @ (nn - nn.mean()) / ((zz - zz.mean()) @ (zz - zz.mean()))
                res[t, ok] = nn - b * zz
        out[f"orth_{ref}"] = res
        out[f"near_{ref}"] = np.where(np.isfinite(z), z + 0.1 * rng.standard_normal(z.shape), np.nan)
    return out


# ------------------------------------------------------------------------------------- recipes
def build_planted_set(ctx: VerificationContext, n_target: int = 300, seed: int = 3) -> tuple[list[Planted], dict]:
    rng = random.Random(seed)
    items: list[Planted] = []

    def add(recipe, cat, claim, expected, formula=None, dialect="qlib", signal_key=None, **meta):
        items.append(Planted(f"P{len(items) + 1:03d}", recipe, cat, claim, expected, formula, dialect, signal_key, meta))

    # 1. Direction (Appendix F: f = CSRank(-(close/Ref(close,5)-1)) and variants)
    for d in (1, 2, 3, 5, 10, 20):
        for sgn, base in ((-1, f"-1*($close/Ref($close, {d})-1)"), (1, f"$close/Ref($close, {d})-1")):
            for wrap in ("{x}", "CSRank({x})", "CSZScore({x})", "2*({x})"):
                f = wrap.format(x=base)
                true_dir = "+" if sgn > 0 else "-"
                false_dir = "-" if sgn > 0 else "+"
                add("direction", STATIC, {"predicate": "SIGN", "args": {"input": f"ret_{d}d", "direction": true_dir}}, SUPPORTED, f)
                add("direction", STATIC, {"predicate": "SIGN", "args": {"input": f"ret_{d}d", "direction": false_dir}}, REFUTED, f)
    for f, inp, dirn in (("CSRank($close/$open)", "close", "+"), ("CSRank($close/$open)", "open", "-"),
                         ("Log($volume+1)", "volume", "+"), ("Mean($volume, 20)/($volume+1e-12)", "volume", "-"),
                         ("($high-$low)/$close", "high", "+"), ("($high-$low)/$close", "low", "-"),
                         ("-1*$amount/Mean($amount, 10)", "amount", "-"), ("TsRank($vwap, 10)", "vwap", "+"),
                         # fields read only through lagged leaves: Ref preserves (§10.2)
                         ("Ref($close, 5)", "close", "+"), ("-1*Ref($volume, 3)", "volume", "-"),
                         ("CSRank(Ref($close, 2)/Ref($open, 2))", "open", "-")):
        add("direction_field", STATIC, {"predicate": "SIGN", "args": {"input": inp, "direction": dirn}}, SUPPORTED, f)
        add("direction_field", STATIC, {"predicate": "SIGN", "args": {"input": inp, "direction": "-" if dirn == "+" else "+"}}, REFUTED, f)

    # 2. Field dependence (formulas with and without each field; out-of-panel inputs)
    # only non-degenerate formulas (e.g. Alpha#7 compares dollar volume to share volume and is constant
    # on any panel with prices > 1, so it depends on nothing — the verifier correctly refutes it)
    base = [lf for lf in base_set() if _non_degenerate(ctx.signal(lf.node), ctx.panel.member)]
    rng.shuffle(base)
    for lf in base[:24]:
        deps = {n.name for n in walk(lf.node) if n.is_field}
        for fld in ("volume", "close"):
            add("dependence", STATIC, {"predicate": "DEPENDS_ON", "args": {"input": fld}},
                SUPPORTED if fld in deps else REFUTED, lf.source_text, lf.dialect)
    for f in ("Mean($close, 10)/$close", "Corr($close, $volume, 10)"):
        add("dependence", STATIC, {"predicate": "DEPENDS_ON", "args": {"input": "analyst revisions"}}, REFUTED, f)
        add("dependence", STATIC, {"predicate": "DEPENDS_ON", "args": {"input": "earnings"}}, REFUTED, f)

    # 3. Lookback (Mean(x,20) vs Mean(x,60); nested Ref/Mean path sums). The claimed n is compared with the
    #    effective lookback L only (n = L or L + 1); inner window parameters do not count (§10.2).
    for n, other in ((20, 60), (5, 10), (60, 20), (10, 30)):
        f = f"Mean($close, {n})/$close"
        add("lookback", STATIC, {"predicate": "LOOKBACK", "args": {"window": n}}, SUPPORTED, f)
        add("lookback", STATIC, {"predicate": "LOOKBACK", "args": {"window": other}}, REFUTED, f)
    for d, n in ((5, 20), (10, 10), (3, 60)):
        f = f"Mean(Ref($close, {d}), {n})/$close"
        L = d + n - 1
        add("lookback_nested", STATIC, {"predicate": "LOOKBACK", "args": {"window": L + 1}}, SUPPORTED, f)
        add("lookback_nested", STATIC, {"predicate": "LOOKBACK", "args": {"window": L + 7}}, REFUTED, f)
        add("horizon", STATIC, {"predicate": "HORIZON", "args": {"bin": "long"}}, REFUTED, f)
    f = "Mean(Ref($close, 60), 5)/$close"                 # L = 64: neither inner parameter is the lookback
    for n, exp in ((5, REFUTED), (60, REFUTED), (64, SUPPORTED), (65, SUPPORTED)):
        add("lookback_nested", STATIC, {"predicate": "LOOKBACK", "args": {"window": n}}, exp, f)
    add("horizon", STATIC, {"predicate": "HORIZON", "args": {"bin": "short"}}, SUPPORTED, "Mean($close, 5)/$close")
    add("horizon", STATIC, {"predicate": "HORIZON", "args": {"bin": "medium"}}, SUPPORTED, "Ref($close, 21)/Ref($close, 120)")
    # horizon bins apply to the effective lookback L (<= 5 / <= 21 / 22-126 / > 126), incl. the bin edges
    for d, good, bad in ((5, "very_short", "medium"), (21, "short", "medium"), (22, "medium", "short"),
                         (126, "medium", "long"), (127, "long", "medium")):
        f = f"-1*($close/Ref($close, {d})-1)"
        add("horizon_edge", STATIC, {"predicate": "HORIZON", "args": {"bin": good}}, SUPPORTED, f)
        add("horizon_edge", STATIC, {"predicate": "HORIZON", "args": {"bin": bad}}, REFUTED, f)

    # 4. Cross-sectional (with / without CSRank at the root)
    for inner in ("Mean($close, 5)/$close", "Corr($close, $volume, 10)", "Std($close, 20)/$close",
                  "($close-$open)/$open", "TsRank($volume, 10)"):
        add("xsec", STATIC, {"predicate": "XSEC", "args": {"value": True}}, SUPPORTED, f"CSRank({inner})")
        add("xsec", STATIC, {"predicate": "XSEC", "args": {"value": True}}, REFUTED, inner)
        add("xsec", STATIC, {"predicate": "XSEC", "args": {"value": False}}, SUPPORTED, inner)

    # 5. Identity: exact library formulas, renamed copies (true), one-character edits (false)
    for lf in base[24:44]:
        add("identity_exact", STATIC, {"predicate": "IDENTITY", "args": {"library_id": lf.lib_id}}, SUPPORTED,
            lf.source_text, lf.dialect)
        renamed = to_program(lf.node, min_size=2) if lf.library != "alpha101" else to_alpha101(lf.node)
        add("identity_renamed", STATIC, {"predicate": "IDENTITY", "args": {"library_id": lf.lib_id}}, SUPPORTED,
            renamed, "qlib" if lf.library != "alpha101" else "alpha101")
        edited = _one_char_edit(to_qlib(lf.node))
        if edited:
            add("identity_edit", STATIC, {"predicate": "IDENTITY", "args": {"library_id": lf.lib_id}}, REFUTED, edited)

    # 6. Range
    for f, lo, hi, exp in (("CSRank($close)", 0, 1, SUPPORTED), ("Corr($close, $volume, 10)", -1, 1, SUPPORTED),
                           ("Log($volume+1)", 0, 1, REFUTED), ("TsRank($close, 10)", 0, 1, SUPPORTED),
                           ("Sign($close-$open)", -1, 1, SUPPORTED), ("$volume/Mean($volume, 5)", -1, 0, REFUTED)):
        add("range", STATIC, {"predicate": "RANGE", "args": {"low": lo, "high": hi}}, exp, f)

    # 6b. Invariance: one constant c multiplies all prices (volumes) (§10.2 metamorphic relations)
    for f, inp, exp in (("CSRank($close)", "price", SUPPORTED), ("$close/Mean($close, 10)", "price", SUPPORTED),
                        ("$close-Mean($close, 5)", "price", REFUTED), ("Log($close)", "price", REFUTED),
                        ("CSRank($volume)", "volume", SUPPORTED), ("Log($volume+1)", "volume", REFUTED)):
        add("invariance", STATIC, {"predicate": "INVARIANT", "args": {"transform": "scale", "input": inp}}, exp, f)

    # 7-8. Resemblance / independence on planted signals
    sigs = planted_signals(ctx)
    for key in sigs:
        if key.startswith("mix_"):
            ref = key[4:].rsplit("_", 1)[0]
            target = float(key.rsplit("_", 1)[1])
            exp = {0.6: SUPPORTED, 0.2: UNRESOLVED, 0.0: REFUTED}[target]
            add("resemblance", STATISTICAL, {"predicate": "RESEMBLES", "args": {"ref": ref, "sign": "+"}}, exp, signal_key=key)
            if target == 0.6:
                add("resemblance_sign", STATISTICAL, {"predicate": "RESEMBLES", "args": {"ref": ref, "sign": "-"}}, REFUTED, signal_key=key)
                add("independence", STATISTICAL, {"predicate": "INDEPENDENT", "args": {"ref": ref}}, REFUTED, signal_key=key)
        elif key.startswith("orth_"):
            ref = key[5:]
            add("independence", STATISTICAL, {"predicate": "INDEPENDENT", "args": {"ref": ref}}, SUPPORTED, signal_key=key)
            add("resemblance", STATISTICAL, {"predicate": "RESEMBLES", "args": {"ref": ref, "sign": "+"}}, REFUTED, signal_key=key)
        elif key.startswith("near_"):
            ref = key[5:]
            add("independence", STATISTICAL, {"predicate": "INDEPENDENT", "args": {"ref": ref}}, REFUTED, signal_key=key)
            add("resemblance", STATISTICAL, {"predicate": "RESEMBLES", "args": {"ref": ref, "sign": "+"}}, SUPPORTED, signal_key=key)

    # 9. Turnover: slow (Mean(x, 60)) vs fast (daily differences)
    for fld in ("close", "volume", "high", "vwap"):
        slow = f"Mean(${fld}, 60)/Mean(${fld}, 120)"
        fast = f"Delta(${fld}, 1)/Ref(${fld}, 1)"
        add("turnover", STATISTICAL, {"predicate": "TURNOVER", "args": {"level": "low"}}, SUPPORTED, slow)
        add("turnover", STATISTICAL, {"predicate": "TURNOVER", "args": {"level": "high"}}, REFUTED, slow)
        add("turnover", STATISTICAL, {"predicate": "TURNOVER", "args": {"level": "high"}}, SUPPORTED, fast)
        add("turnover", STATISTICAL, {"predicate": "TURNOVER", "args": {"level": "low"}}, REFUTED, fast)

    if len(items) > n_target:
        keep_stat = [i for i in items if i.category == STATISTICAL]
        keep_static = [i for i in items if i.category == STATIC]
        rng.shuffle(keep_static)
        items = keep_stat + keep_static[: n_target - len(keep_stat)]
        for k, it in enumerate(items):
            it.item_id = f"P{k + 1:03d}"
    return items, sigs


def _non_degenerate(sig: np.ndarray, member: np.ndarray, share: float = 0.95) -> bool:
    v = np.where(member, sig, np.nan)
    rows = np.isfinite(v).sum(axis=1) >= 5
    if rows.sum() == 0:
        return False
    with np.errstate(all="ignore"):
        sd = np.nanstd(v[rows], axis=1)
    return float((sd > 0).mean()) >= share


def _one_char_edit(src: str) -> str | None:
    """Change one window digit or flip one sign -> a different formula (not equivalent)."""
    import re

    m = re.search(r", (\d+)\)", src)
    if m:
        n = int(m.group(1))
        return src[: m.start(1)] + str(n + 1 if n < 9 else n - 1) + src[m.end(1):]
    if "-" in src:
        i = src.index("-")
        return src[:i] + "+" + src[i + 1:]
    return None


# ------------------------------------------------------------------------------------- runner
def run_calibration(ctx: VerificationContext | None = None, out_dir: str | Path | None = None,
                    n_target: int = 300, categories: tuple[str, ...] | None = None) -> dict:
    ctx = ctx or calibration_context()
    items, sigs = build_planted_set(ctx, n_target)
    if categories:
        items = [i for i in items if i.category in categories]
    rows = []
    for it in items:
        node = parse(it.formula, it.dialect) if it.formula else None
        sig = sigs[it.signal_key] if it.signal_key else None
        v = verify_claim(it.claim, node, ctx, signal=sig)
        rows.append({**asdict(it), "verdict": v.verdict, "method": v.method,
                     "correct": v.verdict == it.expected,
                     "sign_error": {v.verdict, it.expected} == {SUPPORTED, REFUTED}})
    df = pd.DataFrame(rows)
    cfg = ctx.thr["calibration"]
    summary = {"n_items": int(len(df))}
    for cat in (STATIC, STATISTICAL):
        d = df[df.category == cat]
        if not len(d):
            continue
        dec = d[d.expected.isin([SUPPORTED, REFUTED])]
        summary[cat] = {"n": int(len(d)), "accuracy": float(d.correct.mean()),
                        "accuracy_decidable": float(dec.correct.mean()) if len(dec) else float("nan"),
                        "sign_errors": int(d.sign_error.sum())}
    acc_static = summary.get(STATIC, {}).get("accuracy_decidable", 1.0)
    acc_stat = summary.get(STATISTICAL, {}).get("accuracy", 1.0)
    sign_err = int(df.sign_error[df.category == STATISTICAL].sum()) if len(df) else 0
    summary["passed"] = bool(acc_static >= cfg["static_accuracy"] and acc_stat >= cfg["statistical_accuracy"]
                             and sign_err <= cfg["max_sign_errors"])
    confusion = pd.crosstab(df.expected, df.verdict).to_dict() if len(df) else {}
    summary["confusion_matrix"] = {str(k): {str(i): int(j) for i, j in v.items()} for k, v in confusion.items()}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_json(out / "planted_claims_results.jsonl", orient="records", lines=True)
        (out / "calibration_summary.json").write_text(json.dumps(summary, indent=1))
    summary["failures"] = df[~df.correct][["item_id", "recipe", "claim", "expected", "verdict", "formula"]].to_dict(orient="records")
    return summary
