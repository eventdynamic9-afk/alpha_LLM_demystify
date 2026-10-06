"""Public formula libraries (§5.5, §7.2): Alpha101 (Kakushadze 2016), GTJA-191 (Guotai Junan 2017)
and Alpha158 (Qlib, ``qlib/contrib/data/handler.py``).

Each entry keeps the formula *verbatim in its source notation* (``source_text``) — that string is
what narrators see in pool K — plus the dialect used to parse it into the typed tree.  Every formula
the DSL can express over OHLCV(+VWAP, amount) is included.  Each Alpha101 / GTJA-191 id is in
exactly one of: the formula dictionary (``available``), ``*_NOT_EXPRESSIBLE_REASONS`` (Alpha101:
``IndNeutralize``, ``cap``, ``product``, non-constant exponents; GTJA: ``SMA``/``WMA``/``SMEAN``,
``REGBETA``/``SEQUENCE``, benchmark series, ``SUMIF``/``SUMAC``/``PROD``, ``SELF``, non-constant
exponents) or ``*_PENDING`` (``unknown``: transcription pending source access).

Base set for Arm B (§7.2): 20 + 20 + 20 formulas, stratified by complexity tercile in
``pools.build``; the remaining entries serve the same-panel reference library and identity checks.

Transcription note: formulas are implemented from the original texts; ``verify_transcriptions()``
re-parses every entry, checks the id partition, and the test-suite checks published structural facts
(fields, windows).  Entries in ``_A101_LATE`` / ``_GTJA_LATE`` were transcribed without access to the
source PDFs (their ``notes`` say so).  Re-check against the source PDFs before the main run (Appendix
E.7 deviation log otherwise).
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
# Every Alpha101 id is in exactly one of _A101 (available), A101_NOT_EXPRESSIBLE (with the reason) and
# A101_PENDING (transcription pending source access); verify_transcriptions() checks the partition.
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
    21: "((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))",
    22: "(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))",
    23: "(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)",
    24: "((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))",
    25: "rank(((((-1 * returns) * adv20) * vwap) * (high - close)))",
    26: "(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))",
    27: "((0.5 < rank((sum(rank(correlation(rank(volume), rank(vwap), 6)), 2) / 2.0))) ? (-1 * 1) : 1)",
    28: "scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))",
    30: "(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))",
    31: "((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))",
    32: "(scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))",
    33: "rank((-1 * ((1 - (open / close))^1)))",
    34: "rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))",
    35: "((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))",
    36: "(((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))",
    37: "(rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))",
    38: "((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))",
    39: "((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))",
    40: "((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))",
    41: "(((high * low)^0.5) - vwap)",
    42: "(rank((vwap - close)) / rank((vwap + close)))",
    43: "(ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))",
    44: "(-1 * correlation(high, rank(volume), 5))",
    45: "(-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))",
    46: "((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))",
    47: "((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))",
    49: "(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))",
    50: "(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))",
    51: "(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))",
    52: "((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))",
    53: "(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))",
    54: "((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))",
    55: "(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))",
    57: "(0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))",
    60: "(0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))",
    61: "(rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))",
    62: "((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)",
    64: "((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)",
    65: "((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)",
    66: "((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)",
    68: "((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)",
    71: "max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))",
    72: "(rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))",
    73: "(max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)",
    74: "((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)",
    75: "(rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))",
    77: "min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))",
    83: "((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))",
    86: "((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)",
    88: "min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))",
    92: "min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))",
    95: "(rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))",
    96: "(max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)",
    98: "(rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))",
    99: "((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)",
    101: "((close - open) / ((high - low) + .001))",
}
# Entries added when the library was completed (§5.5 audit): transcribed without access to the source PDF.
_A101_LATE = frozenset({21, 24, 25, 27, 31, 32, 36, 37, 39, 43, 45, 46, 47, 49, 50, 51, 52, 57, 60, 61, 62, 64, 65,
                        66, 68, 71, 72, 73, 74, 75, 77, 83, 86, 88, 92, 95, 96, 98, 99})
# Base-set members are frozen from select_base_set() (seed 20261006); see that function.
_A101_BASE = {3, 4, 5, 7, 11, 14, 15, 16, 17, 20, 22, 26, 28, 35, 40, 42, 53, 54, 55, 101}

_IND = "IndNeutralize: industry classification is outside the OHLCV(+VWAP) panel"
_POW = "power with a non-constant exponent (x^y, SignedPower(x, y)) is not a DSL operator"
_PROD = "product (rolling product) is not a DSL operator"
# Alpha101 entries that exist but cannot be expressed (library_id_status -> "not_expressible").
A101_NOT_EXPRESSIBLE_REASONS: dict[int, str] = {
    29: _PROD, 48: _IND, 56: "cap: market capitalisation is outside the OHLCV(+VWAP) panel", 58: _IND, 59: _IND,
    63: _IND, 67: f"{_IND}; {_POW}", 69: f"{_IND}; {_POW}", 70: f"{_IND}; {_POW}", 76: _IND, 78: _POW, 79: _IND,
    80: f"{_IND}; {_POW}", 81: _PROD, 82: _IND, 84: _POW, 85: _POW, 87: _IND, 89: _IND, 90: f"{_IND}; {_POW}",
    91: _IND, 93: _IND, 94: _POW, 97: _IND, 100: _IND,
}
A101_NOT_EXPRESSIBLE = frozenset(A101_NOT_EXPRESSIBLE_REASONS)
# Alpha101 entries whose transcription is pending source access (library_id_status -> "unknown"): none.
A101_PENDING: dict[int, str] = {}

# ------------------------------------------------------------------------------------- GTJA-191
# Same partition as Alpha101: _GTJA / GTJA_NOT_EXPRESSIBLE / GTJA_PENDING.  Repeated sub-expressions of
# #49-#51 (DMI-style up/down moves) and #55/#137 (accumulation swing index) are spliced in verbatim.
_G_DN = "((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1))))"
_G_UP = "((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1))))"
_G_ASI = ("16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>"
          "ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+"
          "ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:"
          "(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & "
          "ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+"
          "ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))"
          "*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))")
_GTJA = {
    1: "(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))",
    2: "(-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))",
    3: "SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)",
    4: "((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) < ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME / MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))",
    5: "(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))",
    6: "(RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)",
    8: "RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)",
    11: "SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)",
    12: "(RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))",
    13: "(((HIGH * LOW)^0.5) - VWAP)",
    14: "CLOSE-DELAY(CLOSE,5)",
    15: "OPEN/DELAY(CLOSE,1)-1",
    16: "(-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))",
    18: "CLOSE/DELAY(CLOSE,5)",
    19: "(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))",
    20: "(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100",
    25: "((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))",
    26: "((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))",
    29: "(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME",
    31: "(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100",
    32: "(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))",
    33: "((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) * TSRANK(VOLUME, 5))",
    34: "MEAN(CLOSE,12)/CLOSE",
    35: "(MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) + (OPEN *0.35)), 17),7))) * -1)",
    37: "(-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))",
    38: "(((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)",
    39: "((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)",
    40: "SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100",
    42: "((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))",
    43: "SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)",
    44: "(TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))",
    45: "(RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))",
    46: "(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)",
    48: "(-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))",
    49: f"SUM({_G_DN},12)/(SUM({_G_DN},12)+SUM({_G_UP},12))",
    50: f"SUM({_G_UP},12)/(SUM({_G_UP},12)+SUM({_G_DN},12))-SUM({_G_DN},12)/(SUM({_G_DN},12)+SUM({_G_UP},12))",
    51: f"SUM({_G_UP},12)/(SUM({_G_UP},12)+SUM({_G_DN},12))",
    52: "SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)*100",
    53: "COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100",
    55: f"SUM({_G_ASI},20)",
    56: "(RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19), SUM(MEAN(VOLUME,40), 19), 13))^5)))",
    58: "COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100",
    59: "SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)",
    60: "SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,20)",
    61: "(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)), RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)",
    62: "(-1 * CORR(HIGH, RANK(VOLUME), 5))",
    65: "MEAN(CLOSE,6)/CLOSE",
    66: "(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100",
    70: "STD(AMOUNT,6)",
    71: "(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100",
    73: "((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) - RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)",
    74: "(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))",
    76: "STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)",
    77: "MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))",
    80: "(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100",
    83: "(-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))",
    84: "SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)",
    85: "(TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))",
    86: "((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) : (((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ? 1 : ((-1 * 1) * (CLOSE - DELAY(CLOSE, 1)))))",
    87: "((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)",
    88: "(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100",
    90: "(RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)",
    92: "(MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)), TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)",
    93: "SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)",
    94: "SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)",
    95: "STD(AMOUNT,20)",
    97: "STD(VOLUME,10)",
    98: "((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))",
    99: "(-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))",
    100: "STD(VOLUME,20)",
    101: "((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))), RANK(VOLUME), 11))) * -1)",
    103: "((20-LOWDAY(LOW,20))/20)*100",
    104: "(-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))",
    105: "(-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))",
    106: "CLOSE-DELAY(CLOSE,20)",
    107: "(((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))",
    110: "SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100",
    112: "(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100",
    113: "(-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))))",
    114: "((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) / (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))",
    117: "((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))",
    118: "SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100",
    120: "(RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))",
    123: "((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)",
    124: "(CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)",
    125: "(RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))",
    126: "(CLOSE+HIGH+LOW)/3",
    128: "100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0), 14)))",
    129: "SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)",
    130: "(RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))",
    132: "MEAN(AMOUNT,20)",
    133: "((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100",
    134: "(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME",
    136: "((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))",
    137: _G_ASI,
    138: "((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)",
    139: "(-1 * CORR(OPEN, VOLUME, 10))",
    140: "MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))",
    141: "(RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)",
    142: "(((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME,20)), 5)))",
    145: "(MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100",
    148: "((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)",
    150: "(CLOSE+HIGH+LOW)/3*VOLUME",
    153: "(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4",
    156: "(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)), 2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)",
    161: "MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)",
    163: "RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))",
    167: "SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)",
    168: "(-1*VOLUME/MEAN(VOLUME,20))",
    170: "((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))) - RANK((VWAP - DELAY(VWAP, 5))))",
    171: "((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))",
    175: "MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)",
    176: "CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)",
    177: "((20-HIGHDAY(HIGH,20))/20)*100",
    178: "(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME",
    179: "(RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))",
    184: "(RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))",
    185: "RANK((-1 * ((1 - (OPEN / CLOSE))^2)))",
    187: "SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)",
    189: "MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)",
    191: "((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)",
}
# Entries added when the library was completed (§5.5 audit): transcribed without access to the source PDF.
_GTJA_LATE = frozenset({3, 4, 6, 8, 16, 19, 25, 26, 33, 35, 37, 38, 39, 40, 43, 44, 45, 48, 49, 50, 51, 52, 55, 56,
                        58, 59, 60, 61, 62, 66, 70, 73, 74, 77, 80, 84, 85, 86, 87, 90, 92, 93, 94, 95, 98, 99, 100,
                        101, 103, 105, 106, 107, 110, 112, 113, 114, 117, 118, 120, 123, 124, 125, 126, 128, 129, 130,
                        132, 133, 134, 136, 137, 138, 139, 140, 142, 145, 148, 150, 153, 156, 163, 170, 171, 175, 176,
                        177, 179, 184, 187})
# GTJA 5, 13, 32, 42, 83 and 104 are canonical duplicates of Alpha101 #26, #41, #15, #40, #16 and #22
# and are not base-set candidates (§7.1 dedup rule).  Among the late entries, GTJA 16, 33, 37, 38, 62, 86,
# 98, 99, 105, 107, 113, 114, 117, 120, 136, 139 and 184 duplicate Alpha101 #50, #52, #8, #23, #44, #46,
# #24, #13, #3, #20, #45, #83, #35, #42, #14, #6 and #37.
_GTJA_BASE = {1, 11, 14, 15, 18, 20, 31, 34, 46, 53, 65, 71, 76, 88, 97, 161, 167, 168, 185, 189}

_SMA = "SMA(A, n, m) (recursive exponential moving average) is not a DSL operator"
_BENCH = "benchmark index series (BANCHMARKINDEX*) are outside the per-stock OHLCV panel"
_REG = "REGBETA/SEQUENCE (rolling regression on a time index) is not in the gtja dialect"
_SUMAC = "SUMAC (cumulative sum over the whole history) is not a DSL operator"
_SUMIF = "SUMIF is not in the gtja dialect"
# GTJA-191 entries that exist but cannot be expressed (library_id_status -> "not_expressible").
GTJA_NOT_EXPRESSIBLE_REASONS: dict[int, str] = {
    **{k: _SMA for k in (9, 23, 24, 28, 47, 57, 63, 67, 68, 72, 79, 81, 82, 89, 96, 102, 109, 111, 122, 135, 146, 151,
                         152, 155, 158, 160, 162, 164, 169, 173, 174, 188)},
    **{k: _POW for k in (17, 108, 115, 121, 131)},
    **{k: _REG for k in (21, 116, 147)},
    **{k: _BENCH for k in (75, 181, 182)},
    **{k: _SUMAC for k in (165, 183)},
    **{k: _SUMIF for k in (144, 190)},
    22: "SMEAN (recursive moving average) is not a DSL operator",
    27: "WMA(A, n) (0.9^i-weighted moving average) is not a DSL operator",
    30: "REGRESI on the MKT/SMB/HML factor series (outside the OHLCV panel) and WMA are not DSL operators",
    143: "SELF (recursive definition) is not expressible in the DSL",
    149: f"REGBETA/FILTER on {_BENCH}",
    157: "PROD (rolling product) is not a DSL operator",
}
GTJA_NOT_EXPRESSIBLE = frozenset(GTJA_NOT_EXPRESSIBLE_REASONS)
_TS_OR_ELEM = ("MAX/MIN(x, n) with a constant n: the gtja dialect reads it as elementwise, the intent is probably "
               "TSMAX/TSMIN; unknown (transcription pending source access)")
_MACRO = ("uses the report's macros ({}), which the gtja dialect does not define; "
          "unknown (transcription pending source access)")
_TYPO = "{} in the circulated text; unknown (transcription pending source access)"
# GTJA-191 entries whose transcription is pending source access (library_id_status -> "unknown").
GTJA_PENDING: dict[int, str] = {
    **{k: _TS_OR_ELEM for k in (7, 41, 64, 91, 119, 154)},
    10: _TYPO.format("unbalanced arguments (RANK(MAX(...^2),5))"),
    36: _TYPO.format("CORR without a window and a two-argument RANK"),
    54: _TYPO.format("STD without a window"),
    69: _MACRO.format("DTM/DBM"),
    78: _TYPO.format("undefined function MA"),
    127: _TYPO.format("unbalanced parentheses and MAX(CLOSE,12)"),
    159: _TYPO.format("undefined field HGIH"),
    166: _TYPO.format("malformed expression ((20-2)(SUM(...)))"),
    172: _MACRO.format("HD/LD/TR"),
    180: _TYPO.format("ternary ':' inside the parentheses of its true branch"),
    186: _MACRO.format("HD/LD/TR"),
}

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
_A158_BASE = {"BETA10", "CNTD20", "CNTP10", "CORD10", "CORR5", "IMXD5", "KLOW2", "KSFT2", "KUP2", "LOW0", "OPEN0",
              "QTLU30", "RESI10", "STD5", "SUMD20", "SUMP60", "VSUMN10", "VSUMP10", "VWAP0", "WVMA5"}


def alpha158_definitions() -> dict[str, str]:
    """All 158 Alpha158 features (9 K-bar + 4 price + 29 rolling families x 5 windows)."""
    out = dict(_A158_KBAR)
    out.update(_A158_PRICE)
    for fam, tmpl in _A158_ROLLING.items():
        for d in A158_WINDOWS:
            out[f"{fam}{d}"] = tmpl.format(d=d)
    return out


LATE_NOTE = "transcribed without access to the source PDF; re-check before the main run"


@functools.lru_cache(maxsize=None)
def library() -> dict[str, LibraryFormula]:
    lib: dict[str, LibraryFormula] = {}
    for k, src in _A101.items():
        lid = f"alpha101_{k:03d}"
        lib[lid] = LibraryFormula(lid, f"A101-{k:03d}", "alpha101", src, "alpha101", k in _A101_BASE,
                                  f"WorldQuant Alpha#{k}", LATE_NOTE if k in _A101_LATE else "")
    for k, src in _GTJA.items():
        lid = f"gtja191_{k:03d}"
        lib[lid] = LibraryFormula(lid, f"GTJA-{k:03d}", "gtja191", src, "gtja", k in _GTJA_BASE,
                                  f"GTJA-191 Alpha{k}", LATE_NOTE if k in _GTJA_LATE else "")
    for name, src in alpha158_definitions().items():
        lid = f"alpha158_{name}"
        lib[lid] = LibraryFormula(lid, f"A158-{name}", "alpha158", src, "qlib", name in _A158_BASE,
                                  f"Qlib Alpha158 {name}")
    return lib


def base_set() -> list[LibraryFormula]:
    return [f for f in library().values() if f.base]


def complexity_terciles(formulas) -> tuple[dict[str, int], list[float]]:
    """Pooled node-count terciles (0/1/2) over the given formulas and the two cut points."""
    import numpy as np

    from dsl import descriptors

    sizes = {f.lib_id: descriptors(f.node)["nodes"] for f in formulas}
    q = np.quantile(list(sizes.values()), [1 / 3, 2 / 3])
    return {k: int(s > q[0]) + int(s > q[1]) for k, s in sizes.items()}, [float(q[0]), float(q[1])]


def base_candidates(include_late: bool = False) -> list[LibraryFormula]:
    """Every library formula with a distinct canonical form (the first occurrence wins).  By default the
    entries added after the base set was drawn (``_A101_LATE`` / ``_GTJA_LATE``) are left out, so that
    ``select_base_set()`` keeps reproducing the frozen draw and ``pools.build`` keeps its strata (§7.2)."""
    from dsl import canonical_hash

    late = {f"alpha101_{k:03d}" for k in _A101_LATE} | {f"gtja191_{k:03d}" for k in _GTJA_LATE}
    seen, out = set(), []
    for f in library().values():
        if f.lib_id in late and not include_late:
            continue
        h = canonical_hash(f.node)
        if h not in seen:
            seen.add(h)
            out.append(f)
    return out


def select_base_set(seed: int = 20261006, per_library: int = 20) -> dict[str, list[str]]:
    """§7.2 base set: per library, ``per_library`` formulas drawn at random within pooled complexity
    terciles, as evenly as availability allows (7/7/6 for 20; a short tercile passes its quota to the
    tercile with most remaining candidates). Alpha158 contributes at most one window per feature family.
    The result is frozen in ``_A101_BASE`` / ``_GTJA_BASE`` / ``_A158_BASE``."""
    import random
    import re

    cands = base_candidates()
    terc, _ = complexity_terciles(cands)

    def family(f):
        return re.sub(r"\d+$", "", f.lib_id) if f.library == "alpha158" else f.lib_id

    rng = random.Random(seed)
    out = {}
    for lib in ("alpha101", "gtja191", "alpha158"):
        cells = []
        for t in range(3):
            fams: dict[str, list[str]] = {}
            for f in sorted((f for f in cands if f.library == lib), key=lambda f: f.lib_id):
                if terc[f.lib_id] == t:
                    fams.setdefault(family(f), []).append(f.lib_id)
            cells.append(fams)
        avail = [len(c) for c in cells]
        take = [min(per_library // 3 + (1 if i < per_library % 3 else 0), a) for i, a in enumerate(avail)]
        while sum(take) < per_library:
            i = max(range(3), key=lambda j: (avail[j] - take[j], -j))
            if avail[i] - take[i] <= 0:
                break
            take[i] += 1
        sel, used = [], set()
        for t in range(3):
            keys = sorted(cells[t])
            rng.shuffle(keys)
            n = 0
            for k in keys:
                if n >= take[t]:
                    break
                if k in used:
                    continue
                ids = cells[t][k]
                sel.append(ids[rng.randrange(len(ids))])
                used.add(k)
                n += 1
        out[lib] = sorted(sel)
    return out


# numbered libraries: (size, available formulas, not-expressible reasons, pending reasons)
_NUMBERED = {"alpha101": (101, _A101, A101_NOT_EXPRESSIBLE_REASONS, A101_PENDING),
             "gtja191": (191, _GTJA, GTJA_NOT_EXPRESSIBLE_REASONS, GTJA_PENDING)}


def _numbered(lib_id: str) -> tuple[str, int] | None:
    """('alpha101' | 'gtja191', number) for a numbered library id (zero padding optional), else None."""
    lib, _, num = str(lib_id).partition("_")
    return (lib, int(num)) if lib in _NUMBERED and num.isdigit() else None


def get(lib_id: str) -> LibraryFormula | None:
    f = library().get(lib_id)
    if f is None and (n := _numbered(lib_id)):
        f = library().get(f"{n[0]}_{n[1]:03d}")
    return f


def by_short_id(short_id: str) -> LibraryFormula | None:
    for f in library().values():
        if f.short_id == short_id:
            return f
    return None


def library_id_status(lib_id: str) -> str:
    """'available' | 'not_expressible' (exists, but uses an operator or series the DSL cannot express;
    listed with the reason in ``*_NOT_EXPRESSIBLE_REASONS``) | 'unknown' (transcription pending source
    access, ``*_PENDING``; also any id of a library outside the reference set) | 'nonexistent'."""
    if lib_id in library():
        return "available"
    n = _numbered(lib_id)
    if n:
        size, avail, nx, pending = _NUMBERED[n[0]]
        k = n[1]
        if not 1 <= k <= size:
            return "nonexistent"
        if k in avail:
            return "available"
        if k in nx:
            return "not_expressible"
        if k in pending:
            return "unknown"
        raise LookupError(f"{lib_id} is in none of the {n[0]} status sets (see verify_transcriptions)")
    if str(lib_id).partition("_")[0] == "alpha158":
        return "nonexistent"
    return "unknown"


def library_id_reason(lib_id: str) -> str:
    """Why an id is not available ('' when it is): the not-expressible or pending reason."""
    st = library_id_status(lib_id)
    if st == "available":
        return ""
    n = _numbered(lib_id)
    if st == "nonexistent":
        return f"{lib_id} is not an id of the named library"
    if n is None:
        return f"{lib_id} is not in the reference libraries (Alpha101, GTJA-191, Alpha158)"
    _, _, nx, pending = _NUMBERED[n[0]]
    return {**nx, **pending}[n[1]]


def verify_transcriptions() -> list[str]:
    """Parse every entry and check that each Alpha101 / GTJA-191 id is in exactly one of available /
    not-expressible / pending, and that every not-expressible or pending entry has a reason; returns the
    problems (should be empty)."""
    bad = []
    for lid, f in library().items():
        try:
            f.node
        except Exception as exc:  # pragma: no cover - reported to the caller
            bad.append(f"{lid}: {exc}")
    for lib, (size, avail, nx, pending) in _NUMBERED.items():
        sets = {"available": set(avail), "not_expressible": set(nx), "pending": set(pending)}
        for k in range(1, size + 1):
            hits = [name for name, ids in sets.items() if k in ids]
            if len(hits) != 1:
                bad.append(f"{lib}_{k:03d}: in {hits or 'no'} status set(s)")
        for name, ids in sets.items():
            if any(not 1 <= k <= size for k in ids):
                bad.append(f"{lib}: {name} ids outside 1..{size}")
        for k, why in {**nx, **pending}.items():
            if not why:
                bad.append(f"{lib}_{k:03d}: no reason given")
    return bad
