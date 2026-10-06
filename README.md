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

### Running without a reachable model endpoint: the file relay

`provider: relay` (`narrate/relay.py`, roster example `configs/models.pilot.yaml`) turns every LLM call into a
request file. When `<RELAY_DIR>/responses/<key>.txt` is missing, the stage writes
`<RELAY_DIR>/requests/<key>.txt`, skips that item and reports how many are pending. An external agent
answers each request in a fresh context: it reads only that file and writes only the response. Re-running
the stage then completes it. The key hashes (model, messages, tools, temperature, max_tokens, seed), so the
k samples of a cell are separate requests and re-runs are deterministic. A2 and B3 tool use goes through a
text protocol: `TOOL_CALL: {...}` lines are executed by the sandbox, and the next turn is a new request
carrying the whole conversation, as with a stateless chat API. Temperature and seed are recorded but cannot
be enforced on the agent.

```bash
export RELAY_DIR=runs/pilot_us/relay
python -m pools build --panel … --out runs/pilot_us --scale 0.4 --authors configs/models.pilot.yaml --protocols P1_raw --n-arm-a 12
python -m narrate relay pending --dir $RELAY_DIR --out pending.json      # work list for the answering agents
# … answer, then re-run the same command until nothing is pending; same for narrate / parse / judges
python -m narrate run --pilot --formulas runs/pilot_us/formulas.jsonl --panel … --run-dir runs/pilot_us --models configs/models.pilot.yaml
```

### US fallback panel (2013–2018)

`python -m data rebuild us-plotly` builds `data/processed/us_sp500_plotly.npz` when the §5.4 sources
cannot be reached. Prices come from the CC0 Kaggle "S&P 500 stock data" file (plotly/datasets mirror),
and point-in-time membership from fja05680/sp500. Cleaning is deterministic and every change is logged in
`data/coverage/`:
* no-trade rows, one-day reversed spikes and inconsistent OHLC prints are set to missing;
* the 13 corporate actions with an overnight gap of 30 % or more are adjusted (see
  `data/rebuild/us_plotly.py::CORPORATE_ACTIONS`).

The panel is a documented deviation (Appendix E.7):
* it is survivorship-biased, holding only the February-2018 constituents (77 % of point-in-time members
  in 2013, 99 % in 2018);
* it carries price returns only, with no dividend adjustment;
* it has no post-cutoff window.

The panel declares its windows (train 2014–2015, valid 2016H1, test 2016-07 to 2018-02), and every CLI
uses them.

## Implementation decisions to note in the paper

These are places where the protocol left a detail open and the code fixes one. Each is documented in
the module and frozen with the thresholds:

* **E1 executor.** `pyqlib` has no wheels for Python ≥ 3.13, so E1 is a pandas engine that follows Qlib's
  operator definitions (`percentileofscore` ranks, `argmax()+1`, per-instrument rolling windows, a
  cross-sectional layer). `executors/e1_qlib/qlib_engine.py` runs the real Qlib engine on every
  cross-section-free subtree when `pyqlib` is installed (Python ≤ 3.12); it is untested here.
  `E1Executor(qlib_native=True)` reproduces Qlib's `min_periods=1` and `WMA` normalization.
* **Canonical NaN, tie and summation rules** (both executors, `executors/semantics.py`):
  * A window is NaN unless all its values are present.
  * A window is constant iff max = min; then Corr/Rsquare are NaN and Std/Var are 0.
  * Tie-sensitive operators zero same-date summation residues of an exact zero (below 1e-12 of that date's
    cross-sectional max, so the rule is causal) and compare inputs rounded to 12 significant digits.
  * `CSScale` and `CSZScore` use correctly rounded sums: E1 uses `math.fsum`; E2 has its own exact-integer
    implementation, so the code is independent but the results are bit-identical.
* **Agreement rule (§6.4).** The criterion is max |E1 − E2| ≤ 1e-8. The rank-correlation rule (min daily
  ρ ≥ 0.9999) applies only "where float order matters", which the code defines as either:
  * the tree contains an order-dependent operator (ranks, argmax, comparisons, selection, `If`, `Abs`); or
  * the 99th-percentile relative difference is at floating-point noise (≤ 1e-9), as with sums of share
    volumes or an ill-conditioned division.

  A wrong constant, a ddof slip or a weighting bug fails both conditions.

  Results:
  * All 44 operators agree on an edge-case panel with suspensions, cross-stock ties and constant windows.
  * All 500 random formulas agree, now drawn with every operator.
  * All 355 library formulas agree.
  * All 227 formulas of the previous library agree on the real US slice: 222 within 1e-8, 5 by the rank
    rule.
* **Numerical equivalence (§6.5).** The denominator is every date on which either signal is evaluable. A
  date where one side is undefined or constant counts as a failure, and the two signals' coverage must
  agree to within 1%.
* `Rsquare` / `Resi` need windows ≥ 3, because a 2-point fit is identically 1 / 0.
* **Direction claims about a raw field.**
  * Raising the most recent value of the field that the formula reads: today's value, or the latest
    lagged leaf when the field enters only through `Ref`.
  * A claim is REFUTED as "does not depend" only when the field is absent from the dependency set.
  * Derived quantities (`ret_5d`, volatility, range, abnormal volume) use the pre-registered path
    perturbations in `verify/nudge.py`.
* **LOOKBACK / HORIZON** are decided on the effective lookback L (§10.2). LOOKBACK(n) is SUPPORTED iff
  n ∈ {L, L+1} (`thresholds.lookback.convention`); window parameters are reported as evidence only.
  HORIZON bins L itself.
* **INVARIANT(scale)** uses one global constant c = 1.7 (`thresholds.metamorphic.scale_c`). Per-stock
  scalings are reported, not decided.
* **Codebook terms with several operationalizations** (e.g. short-term reversal = 5d and 21d) follow the
  pre-registered any/all rule (Appendix C): SUPPORTED if any operationalization is SUPPORTED, REFUTED only
  if all are. This applies to INDEPENDENT as well.
* **Performance claims (§10.4).**
  * SUPPORTED iff OOS t > 3.0. REFUTED iff the 95% interval t ± 1.96 lies at or below 3.0. This rule is
    new: the former refute_t = 2.0 is removed, and the change must be logged as a pre-registration
    amendment.
  * DSR is deflated by the recorded trial count and by the variance of the logged candidates' Sharpe
    ratios (`trials_gp.jsonl`, P1 refinement logs).
  * `verify perf-family` reports PBO/CSCV (S = 16), RC and SPA. "Best of" claims use Romano–Wolf.
  * Test-window and H_post verdicts are separate; an H_post shorter than 6 months is exploratory.
* **AlphaLogics' "> 90% agreement"** is operationalized as mean daily pairwise concordance
  (1 + Kendall τ) / 2 > 0.90; the strict rule is ρ ≥ 0.999 on ≥ 99% of dates.
* **FGX double selection** uses decile portfolios of the reference characteristics as test assets; the
  NOVEL alpha is the HAC t of the formula's long–short return on the union of selected controls.
* **B5** uses an off-the-shelf NLI cross-encoder when `transformers` is installed; otherwise a
  deterministic surface-matching entailment over the template description of the AST.
* **Base set (§7.2)** is frozen from `pools.library.select_base_set()`. Per library, 20 formulas are drawn
  at random within pooled node-count terciles, 7/7/6 as far as availability allows: Alpha101 has no
  formula in the lowest tercile. Alpha158 contributes at most one window per feature family.
* **Degenerate published formulas.** The validity filter removes formulas that are constant on the
  panel. For example, Alpha101 #7 compares dollar volume (`adv20`) with share volume, so it is identically
  −1 whenever prices exceed 1. Base-set formulas that fail are replaced by the nearest-complexity reserve
  formula from the same library, and the swap is logged in `pools_report.json`.
* **SA / SP / N validation (§7.2).**
  * An SA variant is kept only when the verifier confirms the targeted change. Otherwise it is dropped and
    counted as shortfall, with one variant per base and type.
    * Sign: the decided PRED_SIGN verdicts flip, or every statically decided input direction flips.
    * Window: the parameter is gone and the effective lookback changes.
    * Field: the new dependence holds, and either the old one is gone or the behaviour changed.
  * SP variants are validated on the exact presented text: math notation (`dsl/math_parser.py`),
    anonymized text and program form are each parsed back.
  * N is built last and checked against every other pool. Its novelty "passes" only when every search
    completed with zero hits; otherwise the status is `pending_search`, and such formulas are kept only with
    `--allow-pending-search` (pilot runs without search access).
* **Library coverage (§5.5).**
  * 355 formulas: 76 Alpha101, 121 GTJA-191 and 158 Alpha158.
  * Every other Alpha101 / GTJA id is explicitly `not_expressible` with a reason (IndNeutralize, cap,
    product; SMA, EMA-type WMA, REGBETA, benchmark series) or `pending` its source check (17 GTJA ids).
  * The 128 later transcriptions were made without access to the source PDFs and carry a note saying so.
    Re-check them against the sources before the main run, or log a deviation.
* The MACD variant template uses an SMA approximation because the DSL has no EMA.
* **Narration defaults.**
  * The T = 0 reference sample is planned by default (`--no-reference-t0` opts out).
  * Each P2 formula gets exactly one cross-narrator from another family (a global assignment).
  * After an exhausted A2 tool budget, the model gets one final turn with tools disabled.
  * §14 ablation cells (structured elicitation, 150-word cap, reasoning effort) use separate templates;
    the primary templates are byte-identical to Appendix A.

## Data tiers

Mechanistic (C1) and identity (C6) verdicts only need a clean OHLCV panel. Behavioral claims (C2/C3) use
the same-panel reference library. Performance claims (C4) are survivorship-sensitive: they are flagged
on free data until replicated on paid data. Licensing and the "paid-grade but legitimately free"
categories are in `data/LICENSES.md`. Raw licensed data is never committed.
