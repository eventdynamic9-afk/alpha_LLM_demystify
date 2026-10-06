# Traceability: protocol section → implementation

Every requirement of *Experimental Pipeline v2* (6 Oct 2026) mapped to the code or file that implements
it, and to the test that checks it. "Run-date action" marks items that cannot be completed in code
(pinning models, downloading licensed data, human annotation).

| § | Requirement | Implementation | Tests / evidence |
|---|---|---|---|
| 0.1 | Decision matrix, free-run options | `configs/study.yaml`, `data/rebuild/`, `configs/models.yaml` | — |
| 1.2 | RQ1–RQ6, directional hypotheses | `analysis/confirmatory.py` (CF1–CF6), `paper/preregistration.md` | `analysis/tests/test_end_to_end.py` |
| 3.1–3.3 | Units, arms A/B/C, independent variables | `narrate/plan.py`, `pools/build.py`, `judges/` | `narrate/tests` (`test_full_plan_is_3555_per_model`) |
| 3.5 | Not a full factorial: A2 subset, minimal prompt on 25% | `narrate/plan.py` | same |
| 4.1 | Roster criteria (families, ladder, reasoning, tool use, pinning, cutoff, data-use, role separation) | `narrate/roster.py`, `configs/models.yaml` | `narrate/tests::test_roster_validation` · **run-date action: pin models** |
| 4.2 | Role assignment (authors, narrators, cross-narrators, parser, judges) | `configs/models.yaml`, `narrate/plan.py` (cross-narration), `parse/__main__.py`, `judges/__main__.py` | — |
| 4.3 | Determinism (k = 3), JSONL call log, prompt-template SHA-256, response cache | `narrate/logger.py`, `narrate/runner.py` | `narrate/tests::test_cache_and_classify`, `test_mock_narration_and_runner` |
| 4.4 | Free LLM routes (local open-weight, free API tiers) | `narrate/clients.py` (OpenAI-compatible: vLLM, llama.cpp, Ollama, OpenRouter, Groq, Mistral, Cerebras, Gemini compat.) | `narrate/tests::test_get_client_requires_endpoint` |
| 4.1 | Empirical cutoff probe | `narrate/cutoff_probe.py` | **run-date action: fill question bank** |
| 5.2 | Splits (comparability, long-history CN), post-cutoff window, label convention, horizons | `configs/study.yaml`, `narrate/roster.py::post_cutoff_window`, `data/labels.py` | `data/tests::test_forward_returns_convention` |
| 5.3 | CN free stack: chenditc Qlib bins, BaoStock cross-check, 5 bp mismatch, PIT membership, price limits | `data/rebuild/cn.py`, `data/qlib_bin.py`, `data/validation.py`, `data/labels.py::limit_locked` | `data/tests` · **run-date action: download** |
| 5.4 | US free stack: yfinance, fja05680 S&P 500 membership, Tiingo/EODHD/FMP delisted fills, VWAP proxy, coverage, Shumway bound | `data/rebuild/us.py`, `data/membership.py`, `data/panel.py`, `data/validation.py` | `data/tests::test_membership_helpers`, `test_cross_source_and_coverage` |
| 5.5 | Same-panel reference library (Alpha158 / Alpha101 / GTJA-191 + Appendix C characteristics); French, OSAP, JKP, q-factors, CH-3/4 | `verify/references.py`, `pools/library.py`, `data/rebuild/references.py` | `data/tests::test_french_parser`, `pools/tests::test_library_shape` |
| 5.6 | JPX / crypto robustness universes | `data/rebuild/references.py::jpx_panel`, `binance_daily` | — |
| 5.7–5.8 | Paid sources; legitimate "paid-grade free" categories 1–7 | `data/LICENSES.md` | — |
| 5.9 | Data QA checklist, MANIFEST with SHA-256 | `data/validation.py::qa_checklist`, `data/manifest.py`, `data/MANIFEST.csv` | `data/tests::test_qa_checklist_runs` |
| 5.10 | Licensing table | `data/LICENSES.md` | — |
| 6.1 | Typed expression tree; Qlib DSL canonical with `CSRank`/`CSZScore`; Alpha101 and GTJA parsers; serializers to Qlib, Alpha101, math | `dsl/ast.py`, `dsl/parser.py`, `dsl/serialize.py`, `dsl/grammar.ebnf` | `dsl/tests` (round trips, Hypothesis) |
| 6.2 | Operator table: arity, kinds, monotonicity per argument, range, warm-up, unit rule | `dsl/operators.py`, `dsl/ranges.py`, `dsl/units.py` | `dsl/tests::test_units`, `test_ranges` |
| 6.3 | Look-ahead: grammar, AST validator, dynamic truncation test (≥ 5 prefixes), label hygiene | `dsl/grammar.ebnf`, `dsl/validate.py`, `executors/causality.py`, `data/labels.py` | `executors/tests::test_truncation_test` (catches a leaky executor), `test_label_hygiene.py` |
| 6.4 | Two independent executors; ≤ 1e-8 or rank corr ≥ 0.9999; 500 random formulas; property tests; fixtures | `executors/e1_qlib/`, `executors/e2_numpy/`, `executors/compare.py`, `executors/fixtures/` | `executors/tests` (`test_e1_e2_agree_on_500_random_formulas` [slow], Hypothesis laws, hand-computed values) |
| 6.5 | Canonical form, numerical equivalence (ρ ≥ 0.999 on ≥ 99% dates), dedup | `dsl/canonical.py`, `verify/identity.py::numerically_equivalent`, `pools/validity.py` | `dsl/tests::test_canonical_identities` |
| 6.6 | Complexity descriptors | `dsl/complexity.py` | `dsl/tests::test_descriptors` |
| 7.1 | Validity filter; P1 (raw/mined, ≤ 5 rounds, trial log), P2, P3a GP, P3b random grammar (complexity matched), P4 importers | `pools/validity.py`, `pools/llm_authors.py`, `pools/gp.py`, `pools/random_grammar.py`, `pools/in_the_wild.py` | `pools/tests::test_build_small_pools`, e2e test |
| 7.2 | Base set 20/20/20 by complexity tercile; K, K-named, SP (4 types), SA (sign/window/field), NL, N; novelty checks 1–5; canary | `pools/library.py`, `pools/perturb.py`, `pools/build.py`, `pools/novelty.py`, `pools/canary.py` | `pools/tests` |
| 7.3 | Run counts: 3,555 narrations per model | `narrate/plan.py` | `narrate/tests::test_full_plan_is_3555_per_model` |
| 8.1 | Guided / minimal prompts verbatim; ≤ 250 words | `configs/prompts/narrator_*.txt`, `narrate/prompts.py` | `narrate/tests::test_prompt_is_verbatim_template` |
| 8.2 | A0 / A1 diagnostics table / A2 sandbox tool (≤ 10 calls, training window) | `narrate/diagnostics.py`, `narrate/sandbox.py`, `narrate/runner.py` | `narrate/tests::test_sandbox_tools` |
| 8.3–8.5 | k = 3 at T = 0.7 + T = 0 reference; randomization; fresh context; refusals re-sampled once | `narrate/plan.py`, `narrate/runner.py`, `narrate/prompts.py` | `narrate/tests::test_field_order_randomized_but_deterministic` |
| 9.1 | Claim taxonomy C1–C6 + slots | `configs/codebook.yaml`, `parse/schema.py` | `parse/tests::test_rule_extraction_covers_taxonomy` |
| 9.2 | E1 rules, E2 frozen LLM parser, E3 ensemble | `parse/rules.py`, `parse/llm_parser.py`, `parse/ensemble.py` | `parse/tests` |
| 9.3 | Parser validation: P/R/F1 by type, slots, κ / α / AC1, go/no-go | `parse/validate.py`, `annotation/agreement.py`, `annotation/tools.py` | `parse/tests::test_validation_metrics_and_gate`, `analysis/tests::test_agreement_statistics` · **run-date action: annotate** |
| 9.4 | PPI / PPI++ bias correction | `analysis/ppi.py` | `analysis/tests::test_ppi_unbiased_with_biased_parser` |
| 9.5 | Normalization (bins, C5, out-of-panel inputs) | `parse/normalize.py`, `verify/inputs.py` | `parse/tests::test_slots_and_normalization` |
| 10.1 | Verdicts SUPPORTED / REFUTED / UNRESOLVED / UNVERIFIABLE / AMBIGUOUS | `verify/verdicts.py` | `verify/tests::test_verdict_helpers` |
| 10.2 | Static analysis, monotonicity abstract interpretation, nudge tests (Clopper–Pearson), metamorphic relations, optional SMT, driver attribution (S_T), identity / variant claims | `verify/static.py`, `dsl/monotonicity.py`, `verify/nudge.py`, `verify/metamorphic.py`, `verify/smt.py`, `verify/drivers.py`, `verify/identity.py` | `verify/tests` |
| 10.3 | Behavioral: ρ̄ with stationary bootstrap (auto block) + Newey–West; RESEMBLES, INDEPENDENT (TOST), EXPOSED (HAC t ≥ 3, floor), TURNOVER (reference percentiles), REGIME, PRED_SIGN; NOVEL (ceiling + FGX double selection); POPPER alternative noted | `verify/behavioral.py`, `verify/stats.py`, `verify/originality.py`, `verify/factors.py` | `verify/tests` |
| 10.4 | Performance: OOS only, HLZ t > 3, DSR with trial counts, PBO/CSCV (S = 16), Reality Check, SPA, Romano–Wolf, costs, limit locks | `verify/performance.py` | `verify/tests::test_pbo_dsr_and_snooping_tests`, `test_backtest_costs_reduce_returns` |
| 10.5 | Theory claims never scored | `verify/dispatcher.py` | `verify/tests` (THEORY → UNVERIFIABLE) |
| 10.6 | Pre-registered thresholds + sensitivity ranges | `configs/thresholds.yaml`, `analysis/robustness.py::threshold_variants` | — |
| 10.7 | Planted-claims calibration (300 items; ≥ 99% static, ≥ 95% statistical, zero sign errors), dual executors, human spot-check, unit tests in CI | `verify/calibration.py`, `annotation/tools.py::export_verdict_spotcheck`, `.github/workflows/ci.yml` | `verify/tests::test_calibration_subset_passes`, `test_full_calibration_set` [slow] |
| 11 | All metrics | `analysis/metrics.py`, `judges/validity.py` | `analysis/tests`, e2e test |
| 12.2 | Logistic GLMM with crossed random effects; marginal effects; R equivalent | `analysis/glmm.py` | `analysis/tests::test_glmm_runs_and_recovers_direction` |
| 12.3 | Two-stage cluster bootstrap; small-cell exact intervals; paired Arm-B contrasts | `analysis/bootstrap.py`, `verify/stats.py` (Clopper–Pearson, Wilson) | `analysis/tests::test_cluster_bootstrap_and_differences` |
| 12.4 | Power: worked example, design effects, GLMM simulation | `analysis/power.py` | `analysis/tests::test_worked_example_matches_section_12_4` |
| 12.5 | Holm confirmatory family; BH/BY exploratory | `analysis/multiplicity.py`, `analysis/confirmatory.py` | `analysis/tests::test_multiplicity` |
| 12.6–12.8 | Performance statistics; robustness battery; reporting standards; forest plots | `verify/performance.py`, `analysis/robustness.py`, `analysis/figures.py`, `paper/make_tables.py` | e2e test |
| 13 | Baselines B1–B6 | `judges/llm_judges.py`, `judges/b5_nli.py`, `judges/b6_human.py`, `judges/validity.py` | `judges/tests` |
| 14 | Ablations | supported by plan/prompt options (`prompt_variant`, notation SP types, anonymization, cross-narration, dimensional flag); ChronoGPT-Instruct = an extra roster entry | — |
| 15 | Threats and mitigations | parser validation + PPI, verifier calibration, pre-registered thresholds, codebook bins, separate fidelity/performance axes, complexity matching, minimal prompt, version pinning + cache, PIT membership, post-cutoff window, Novel-pool guard (`narrate/roster.py::novel_pool_allowed`), trial logging | — |
| 16 | Reproducibility package, repository layout | this repository (`configs/ data/ dsl/ executors/ pools/ narrate/ parse/ verify/ judges/ analysis/ annotation/ paper/`), `requirements.lock`, `Dockerfile` | CI |
| 17 | Phased execution plan; budget arithmetic; first-week commands | `Makefile`, `README.md`, `configs/study.yaml` (pilot) | — |
| 18 | Explicit non-claims | `paper/README.md` | — |
| App. A | Narrator prompt templates (verbatim) | `configs/prompts/narrator_*.txt` | `narrate/tests` |
| App. B | Parser prompt and output schema | `configs/prompts/parser_instruction.txt`, `parse/schema.py` | `parse/tests` |
| App. C | Codebook operational definitions | `configs/codebook.yaml` (`terms`, `reference_signals`, `horizon_bins`) | — |
| App. D | Record schemas | `pools/records.py`, `narrate/runner.py`, `parse/llm_parser.py`, `verify/__main__.py` | — |
| App. E | Pre-registration template | `paper/preregistration.md` | — |
| App. F | Planted-claims recipes | `verify/calibration.py` | `verify/tests` |
| App. G | Search log | `paper/search_log.md` | — |
