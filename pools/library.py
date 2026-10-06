"""Public formula libraries (§5.5, §7.2): Alpha101 (Kakushadze 2016), GTJA-191 (Guotai Junan 2017)
and Alpha158 (Qlib, ``qlib/contrib/data/handler.py``).

Each entry keeps the formula *verbatim in its source notation* (``source_text``) — that string is
what narrators see in pool K — plus the dialect used to parse it into the typed tree.  Only formulas
expressible over OHLCV(+VWAP, amount) are included: Alpha101 entries using ``IndNeutralize``,
``cap`` or ``product`` and GTJA entries using ``SMA``/``WMA``/``REGBETA``/benchmark series are out
of scope by construction.

Base set for Arm B (§7.2): 20 + 20 + 20 formulas, stratified by complexity tercile in
``pools.build``; the remaining entries serve the same-panel reference library and identity checks.

Transcription note: formulas are implemented from the original texts; ``verify_transcriptions()``
re-parses every entry and the test-suite checks published structural facts (fields, windows).
Re-check against the source PDFs before the main run (Appendix E.7 deviation log otherwise).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

from dsl import Node, parse


@dataclass(frozen=True)
class LibraryFormula:
    lib_id: str               # e.g. alpha101_012
    short_id: str             # e.g. A101-012 (pool ids, Appendix D)
    library: str              # alpha101 | gtja191 | alpha158
    source_text: str          # verbatim, source notation
    dialect: str              # parser dialect for source_text
    base: bool = False        # member of the 60-formula Arm B base set
    display_name: str = ""    # e.g. "WorldQuant Alpha#12" (K-named condition)
    notes: str = field(default="")

    @property
    def node(self) -> Node:
        return _parse_cached(self.source_text, self.dialect)


@functools.lru_cache(maxsize=None)
def _parse_cached(src: str, dialect: str) -> Node:
    return parse(src, dialect)


# ------------------------------------------------------------------------------------- Alpha101
_A101 = {
    1: "(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)",
    2: "(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))",
    3: "(-1 * correlation(rank(open), rank(volume), 10))",
    4: "(-1 * Ts_Rank(rank(low), 9))",
    5: "(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))",
    6: "(-1 * correlation(open, volume, 10))",
    7: "((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))",
    8: "(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))",
    9: "((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))",
    10: "rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))",
    11: "((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))",
    12: "(sign(delta(volume, 1)) * (-1 * delta(close, 1)))",
    13: "(-1 * rank(covariance(rank(close), rank(volume), 5)))",
    14: "((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))",
    15: "(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))",
    16: "(-1 * rank(covariance(rank(high), rank(volume), 5)))",
    17: "(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))",
    18: "(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))",
    19: "((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))",
    20: "(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))",
    22: "(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))",
    23: "(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)",
    26: "(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))",
    28: "scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))",
    30: "(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))",
    33: "rank((-1 * ((1 - (open / close))^1)))",
    34: "rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))",
    35: "((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))",
    38: "((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))",
    40: "((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))",
    41: "(((high * low)^0.5) - vwap)",
    42: "(rank((vwap - close)) / rank((vwap + close)))",
    44: "(-1 * correlation(high, rank(volume), 5))",
    53: "(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))",
    54: "((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))",
    55: "(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))",
    101: "((close - open) / ((high - low) + .001))",
}
_A101_BASE = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 20, 33, 41, 101}
# Alpha101 entries outside the OHLCV(+VWAP) panel (IndNeutralize / cap / product) — not expressible.
A101_NOT_EXPRESSIBLE = {48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100}

# ------------------------------------------------------------------------------------- GTJA-191
_GTJA = {
    1: "(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))",
    2: "(-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))",
    5: "(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))",
    11: "SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)",
    12: "(RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))",
    13: "(((HIGH * LOW)^0.5) - VWAP)",
    14: "CLOSE-DELAY(CLOSE,5)",
    15: "OPEN/DELAY(CLOSE,1)-1",
    18: "CLOSE/DELAY(CLOSE,5)",
    20: "(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100",
    29: "(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME",
    31: "(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100",
    32: "(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))",
    34: "MEAN(CLOSE,12)/CLOSE",
    42: "((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))",
    46: "(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)",
    53: "COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100",
    65: "MEAN(CLOSE,6)/CLOSE",
    71: "(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100",
    76: "STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)",
    83: "(-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))",
    88: "(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100",
    97: "STD(VOLUME,10)",
    104: "(-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))",
    141: "(RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)",
    161: "MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)",
    167: "SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)",
    168: "(-1*VOLUME/MEAN(VOLUME,20))",
    178: "(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME",
    185: "RANK((-1 * ((1 - (OPEN / CLOSE))^2)))",
    189: "MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)",
    191: "((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)",
}
# GTJA 32 and 83 duplicate Alpha101 #15 and #16 and are kept out of the base set (§7.1 dedup rule).
_GTJA_BASE = {1, 2, 5, 11, 12, 14, 15, 18, 20, 31, 42, 46, 53, 65, 76, 104, 141, 161, 167, 191}

# ------------------------------------------------------------------------------------- Alpha158
_A158_KBAR = {
    "KMID": "($close-$open)/$open",
    "KLEN": "($high-$low)/$open",
    "KMID2": "($close-$open)/($high-$low+1e-12)",
    "KUP": "($high-Greater($open, $close))/$open",
    "KUP2": "($high-Greater($open, $close))/($high-$low+1e-12)",
    "KLOW": "(Less($open, $close)-$low)/$open",
    "KLOW2": "(Less($open, $close)-$low)/($high-$low+1e-12)",
    "KSFT": "(2*$close-$high-$low)/$open",
    "KSFT2": "(2*$close-$high-$low)/($high-$low+1e-12)",
}
_A158_PRICE = {"OPEN0": "$open/$close", "HIGH0": "$high/$close", "LOW0": "$low/$close", "VWAP0": "$vwap/$close"}
_A158_ROLLING = {
    "ROC": "Ref($close, {d})/$close",
    "MA": "Mean($close, {d})/$close",
    "STD": "Std($close, {d})/$close",
    "BETA": "Slope($close, {d})/$close",
    "RSQR": "Rsquare($close, {d})",
    "RESI": "Resi($close, {d})/$close",
    "MAX": "Max($high, {d})/$close",
    "MIN": "Min($low, {d})/$close",
    "QTLU": "Quantile($close, {d}, 0.8)/$close",
    "QTLD": "Quantile($close, {d}, 0.2)/$close",
    "RANK": "Rank($close, {d})",
    "RSV": "($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)",
    "IMAX": "IdxMax($high, {d})/{d}",
    "IMIN": "IdxMin($low, {d})/{d}",
    "IMXD": "(IdxMax($high, {d})-IdxMin($low, {d}))/{d}",
    "CORR": "Corr($close, Log($volume+1), {d})",
    "CORD": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})",
    "CNTP": "Mean($close>Ref($close, 1), {d})",
    "CNTN": "Mean($close<Ref($close, 1), {d})",
    "CNTD": "Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})",
    "SUMP": "Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)",
    "SUMN": "Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)",
    "SUMD": "(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)",
    "VMA": "Mean($volume, {d})/($volume+1e-12)",
    "VSTD": "Std($volume, {d})/($volume+1e-12)",
    "WVMA": "Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)",
    "VSUMP": "Sum(Greater($volume-Ref($volume, 1), 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)",
    "VSUMN": "Sum(Greater(Ref($volume, 1)-$volume, 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)",
    "VSUMD": "(Sum(Greater($volume-Ref($volume, 1), 0), {d})-Sum(Greater(Ref($volume, 1)-$volume, 0), {d}))/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)",
}
A158_WINDOWS = (5, 10, 20, 30, 60)
_A158_BASE = {"KMID", "KLEN", "KUP2", "KSFT2", "VWAP0", "ROC20", "MA10", "STD20", "BETA10", "RSQR20", "RESI10",
              "MAX30", "QTLU20", "RANK10", "RSV20", "IMXD30", "CORR20", "SUMP20", "VMA5", "WVMA20"}


def alpha158_definitions() -> dict[str, str]:
    """All 158 Alpha158 features (9 K-bar + 4 price + 29 rolling families x 5 windows)."""
    out = dict(_A158_KBAR)
    out.update(_A158_PRICE)
    for fam, tmpl in _A158_ROLLING.items():
        for d in A158_WINDOWS:
            out[f"{fam}{d}"] = tmpl.format(d=d)
    return out


@functools.lru_cache(maxsize=None)
def library() -> dict[str, LibraryFormula]:
    lib: dict[str, LibraryFormula] = {}
    for k, src in _A101.items():
        lid = f"alpha101_{k:03d}"
        lib[lid] = LibraryFormula(lid, f"A101-{k:03d}", "alpha101", src, "alpha101", k in _A101_BASE,
                                  f"WorldQuant Alpha#{k}")
    for k, src in _GTJA.items():
        lid = f"gtja191_{k:03d}"
        lib[lid] = LibraryFormula(lid, f"GTJA-{k:03d}", "gtja191", src, "gtja", k in _GTJA_BASE,
                                  f"GTJA-191 Alpha{k}")
    for name, src in alpha158_definitions().items():
        lid = f"alpha158_{name}"
        lib[lid] = LibraryFormula(lid, f"A158-{name}", "alpha158", src, "qlib", name in _A158_BASE,
                                  f"Qlib Alpha158 {name}")
    return lib


def base_set() -> list[LibraryFormula]:
    return [f for f in library().values() if f.base]


def get(lib_id: str) -> LibraryFormula | None:
    return library().get(lib_id)


def by_short_id(short_id: str) -> LibraryFormula | None:
    for f in library().values():
        if f.short_id == short_id:
            return f
    return None


def library_id_status(lib_id: str) -> str:
    """'available' | 'not_expressible' (exists but outside the OHLCV panel) | 'unknown' | 'nonexistent'."""
    if lib_id in library():
        return "available"
    lib, _, num = lib_id.partition("_")
    if lib == "alpha101" and num.isdigit():
        k = int(num)
        if 1 <= k <= 101:
            return "not_expressible" if k in A101_NOT_EXPRESSIBLE else "unknown"
        return "nonexistent"
    if lib == "gtja191" and num.isdigit():
        return "unknown" if 1 <= int(num) <= 191 else "nonexistent"
    if lib == "alpha158":
        return "nonexistent"
    return "unknown"


def verify_transcriptions() -> list[str]:
    """Parse every entry; returns ids that fail (should be empty)."""
    bad = []
    for lid, f in library().items():
        try:
            f.node
        except Exception as exc:  # pragma: no cover - reported to the caller
            bad.append(f"{lid}: {exc}")
    return bad
