# rationale-fidelity — Experimental Pipeline v2

**Execution-verified rationale fidelity for LLM alpha discovery.**
Measures, with execution-based ground truth, whether the natural-language rationales that LLMs write
for formulaic alpha factors are *true of the formulas they describe*, and how much of that text is
recalled from memory rather than read from the formula.

This repository implements the full pipeline of the protocol (version 2.0, pre-registration draft,
6 Oct 2026): typed formula DSL with two independent executors, formula pools for the authorship and
counter-recall arms, rationale elicitation at three access levels, claim extraction, an
execution-based verifier with a planted-claims calibration set, the judge baselines, and the
pre-registered statistical analysis. `TRACEABILITY.md` maps every section of the protocol to code and
tests.

> **What is and is not here.** The code, configs, prompts, tests and an offline end-to-end run are
> complete. No study results are included: the main run needs run-date actions that cannot be done in
> code — pinning the model roster against model cards (`configs/models.yaml`), downloading the market
> data (licensing: `data/LICENSES.md`), and human annotation. The offline run uses deterministic *mock*
> LLMs (`configs/models.mock.yaml`) that exist only to exercise every stage.

## Repository layout (§16)

```
configs/      thresholds.yaml (pre-registered, §10.6), study.yaml, codebook.yaml (Appendix C),
              models.yaml (roster template, §4), models.mock.yaml, prompts/ (Appendix A/B verbatim)
data/         panel, synthetic fixtures, labels (isolated from the engine), Qlib-bin reader, PIT
              membership, cross-source validation, QA checklist, MANIFEST.csv, rebuild/ scripts
dsl/          grammar, typed operator table, parsers (Qlib / Alpha101 / GTJA-191), serializers,
              canonicalizer, look-ahead validator, units, ranges, monotonicity, complexity
executors/    e1_qlib/ (Qlib semantics), e2_numpy/ (independent NumPy), causality (truncation test),
              compare (agreement), tests/ (property-based), fixtures/
pools/        public libraries (Alpha101, GTJA-191, all 158 Alpha158), SP/SA/NL/N generators, GP,
              random grammar, LLM authoring P1/P2, P4 importers, validity, novelty, canary
narrate/      clients (OpenAI-compatible + mock), call logger, cache, prompts, A1 diagnostics,
              A2 sandbox, roster validation, run plan (3,555 cells/model), runner, cutoff probe
parse/        schema, rule layer (E1), frozen LLM parser (E2), ensemble (E3), normalization, validation
verify/       static, nudge, metamorphic, drivers, identity, behavioral, originality, performance,
              SMT, dispatcher, planted-claims calibration
judges/       B1 holistic, B2 claim-level, B3 claim-level + tools, B4 reconstruction, B5 NLI, B6 human
analysis/     dataset, metrics (§11), cluster bootstrap, PPI, GLMM (+R script), power, multiplicity,
              confirmatory CF1–CF6, robustness battery, figures
annotation/   guidelines, sampling, sheets, agreement (κ, α, AC1), adjudication, verdict spot-check
paper/        tables generated from analysis outputs only, pre-registration template, search log
```

## Quickstart

```bash
make install                 # venv + requirements.lock + editable install
make test                    # fast suite: DSL, executors, verifier, pools, narration, parser, judges, analysis
make test-all                # + 500-formula E1/E2 agreement, full calibration set, end-to-end pipeline
make calibrate               # planted-claims calibration report (§10.7) -> runs/calibration/
make smoke                   # offline end-to-end run with mock LLMs -> runs/smoke/analysis/tables.md
```

Docker: `docker build -t rationale-fidelity . && docker run --rm -v "$PWD/runs:/app/runs" rationale-fidelity`.

Each stage is also a CLI (`python -m <package> --help`): `data`, `pools`, `narrate`, `parse`, `verify`,
`judges`, `analysis`, `annotation`, `paper`.

## First (free) run — §17

| Phase | Commands |
|---|---|
| 0 Setup | `make data-cn` (chenditc Qlib bins + BaoStock cross-check), `make data-us` (yfinance + historical S&P 500 membership + free-tier delisted fills; set `TIINGO_API_KEY` / `EODHD_API_KEY` / `FMP_API_KEY` if available), `make refs` (Ken French daily factors); `python -m data qa --panel data/processed/cn_csi500.npz --second data/processed/cn_csi500_baostock.npz`; `make test-all`; `make calibrate` — gate: executors agree, calibration ≥ targets, QA complete |
| 1 Pilot | pin 3 models in `configs/models.yaml`; `python -m narrate check-roster --pilot`; `make pools narrate parse verify RUN=runs/pilot PANEL=… MODELS=models.yaml SCALE=0.2` ; double-annotate 60 rationales (`python -m annotation sample --n 60 …`); `python -m parse validate …` — gate: parser go/no-go |
| 2 Freeze | final N via `analysis.power.required_formulas(sd_formula=…, sd_rationale=…)` from the pilot GLMM; freeze `configs/thresholds.yaml` (hash via `configs.thresholds_hash()`), prompts; register `paper/preregistration.md` |
| 3 Main run | `python -m narrate check-roster` (must pass); `make pools narrate parse verify drivers judges RUN=runs/main PANEL=… MODELS=models.yaml` |
| 4 Analysis | `python -m annotation sample --n 200 …`, gold adjudication, `make analysis tables RUN=runs/main` |

Local inference: serve an open-weight model with any OpenAI-compatible server (vLLM, llama.cpp,
Ollama on Kaggle / Colab GPUs) and point `base_url` at it. Free API tiers that expose the same protocol
work through the same client. The Novel pool is only ever sent to local models or endpoints whose
terms exclude training on inputs (`narrate/roster.py::novel_pool_allowed`).

## Implementation decisions to note in the paper

These are places where the protocol left a detail open and the code fixes one. Each is documented in
the module and frozen with the thresholds:

* **E1 executor.** `pyqlib` has no wheels for Python ≥ 3.13, so E1 is a pandas engine that follows Qlib's
  operator definitions (`percentileofscore` ranks, `argmax()+1`, per-instrument rolling windows, a
  cross-sectional layer). `executors/e1_qlib/qlib_engine.py` runs the real Qlib engine on every
  cross-section-free subtree when `pyqlib` is installed (Python ≤ 3.12); it is untested here.
  `E1Executor(qlib_native=True)` reproduces Qlib's `min_periods=1` and `WMA` normalization.
* **Canonical NaN, tie and summation rules** (both executors, `executors/semantics.py`): a window is NaN
  unless all values are present; a window is constant iff max = min (Corr/Rsquare → NaN, Std/Var → 0);
  tie-sensitive operators compare inputs after rounding to 12 significant digits; cross-sectional
  normalizers (`CSScale`, `CSZScore`) use correctly rounded sums. These rules make the two independent
  implementations agree on ties that differ only by floating-point summation order. With them, all 500
  random composite formulas agree on each of three fixture panels (≥ 491 within 1e-8, the rest by the
  rank-correlation rule).
* `Rsquare` / `Resi` need windows ≥ 3, because a 2-point fit is identically 1 / 0.
* **Direction claims about a raw field** are defined as raising that field's most recent value; claims
  about derived quantities (`ret_5d`, volatility, range, abnormal volume) use the pre-registered path
  perturbations in `verify/nudge.py`.
* **LOOKBACK** is SUPPORTED when the stated window equals a window/lag parameter of the formula or its
  effective span (L or L + 1 days).
* **Codebook terms with several operationalizations** (e.g. short-term reversal = 5d and 21d) are
  SUPPORTED if any operationalization is SUPPORTED and REFUTED only if all are; independence claims use
  the reverse rule (refuted if dependence on any operationalization is shown).
* **AlphaLogics' "> 90% agreement"** is operationalized as mean daily pairwise concordance
  (1 + Kendall τ) / 2 > 0.90; the strict rule is ρ ≥ 0.999 on ≥ 99% of dates.
* **FGX double selection** uses decile portfolios of the reference characteristics as test assets; the
  NOVEL alpha is the HAC t of the formula's long–short return on the union of selected controls.
* **B5** uses an off-the-shelf NLI cross-encoder when `transformers` is installed; otherwise a
  deterministic surface-matching entailment over the template description of the AST.
* **Degenerate published formulas.** The validity filter removes formulas that are constant on the
  panel — e.g. Alpha101 #7 compares dollar volume (`adv20`) with share volume, so it is identically −1
  whenever prices exceed 1. Base-set formulas that fail are replaced by the nearest-complexity reserve
  formula from the same library, and the swap is logged in `pools_report.json`.
* **GTJA-191 formulas** that need `SMA`, GTJA's exponential `WMA`, `REGBETA` or benchmark series fall
  outside the operator set and are not in the library. The MACD variant template uses an SMA
  approximation because the DSL has no EMA.
* **Formula transcriptions** of Alpha101 and GTJA-191 come from the original texts. Re-check them against
  the source PDFs before the main run; any correction made after registration is a deviation.

## Data tiers

Mechanistic (C1) and identity (C6) verdicts only need a clean OHLCV panel. Behavioral claims (C2/C3) use
the same-panel reference library. Performance claims (C4) are survivorship-sensitive: they are flagged
on free data until replicated on paid data. Licensing and the "paid-grade but legitimately free"
categories are in `data/LICENSES.md`. Raw licensed data is never committed.
