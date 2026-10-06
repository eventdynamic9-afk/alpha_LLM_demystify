# Experimental Pipeline v2: Execution-Verified Rationale Fidelity for LLM Alpha Discovery

> Requirement document this repository implements (verbatim copy of the specification received on
> 6 October 2026). `TRACEABILITY.md` maps each section to code.

**Version:** 2.0 (supersedes v1) · **Prepared:** 6 October 2026 · **Status:** pre-registration draft
**One-line goal:** measure, with execution-based ground truth, whether the natural-language rationales that LLMs write for formulaic alpha factors are *true of the formulas they describe* — and how much of that text is recalled from memory rather than read from the formula.

**How to read this document**

- `[FREE]` = usable in the first, zero-cost run. `[LOW-COST]` = trials, free tiers with limits, or cheap subscriptions. `[PAID]` = institutional or commercial data / frontier closed models.
- ✅ marks the recommended option for the first (free) run. Every major stage lists alternatives (Option A/B/C) so the design survives a change of budget or a reviewer request.
- Nothing here is a new statistical method. Every component is standard either in the LLM-alpha-mining literature, in LLM-evaluation methodology, in software testing, or in empirical asset pricing. The novelty is the *measurement target* (claim-level truth of factor rationales) and the *counter-recall control*, not the yardstick.
- Facts about third-party data sources, free tiers and model availability change frequently. Every such item is marked "verify on run date" where it matters, and the run log must record what was true when you ran.

---

## 0. What changed from v1 (and why)

| # | Area | v1 | v2 | Why |
|---|---|---|---|---|
| 1 | Positioning | Six named alpha-mining papers, gap stated generally | Literature map of ~40 works (2023–Oct 2026) with the exact "consistency check" each one uses (§2); gap stated against the closest five | Reviewers in this space now know AlphaAgent, Alpha Jungle, AlphaLogics, XALPHA, AlphaQT-Bench, AlphaBench; the novelty claim has to be precise |
| 2 | Research questions | Implicit (RQ1 only) | Six explicit RQs with pre-registered directional hypotheses (§1) | Pre-registration needs named hypotheses |
| 3 | "Field-standard" split | One split asserted as standard | Splits differ across papers; adopt the AlphaAgent/AlphaLogics split for comparability, add a **post-cutoff holdout** and rolling robustness (§5.2) | Alpha Jungle uses 2011–2020/2021–2024; AlphaAgent/AlphaLogics use 2015–2019/2020/2021–2024 |
| 4 | Lookahead | Grammar ban only | Grammar ban + AST validator + **dynamic truncation test** + dual independent executors (§6.3–6.4) | AlphaQT-Bench shows executable code can still leak the future; a verifier must itself be verified |
| 5 | Model-knowledge lookahead | Not addressed for performance claims | Performance claims split into pre-cutoff and post-cutoff windows; "profit mirage" control (§5.2, §10.4) | Memorization and post-cutoff collapse are now documented in finance |
| 6 | Counter-recall pools | Known / Perturbed / Novel; renaming counted as a "perturbation" | Known, Known-with-name, **Semantics-Preserving (SP)**, **Semantics-Altering (SA)**, **Misleading-label (NL)**, Novel (§7.2) | Renaming is semantics-preserving: rationales *should not* change. Mixing SP and SA hides the effect |
| 7 | "Provably absent" novel formulas | Claimed | Replaced by "absent from searchable public sources at generation time" + search log + numerical dedup (§7.2) | Absence from a closed model's training set cannot be proven |
| 8 | Claim taxonomy | 5 types | 6 types incl. **identity/provenance claims** ("this is Alpha#12") and hedging/scope slots (§9.1) | Identity claims are the most direct trace of recall |
| 9 | Verdicts | true / false / unresolved | SUPPORTED / REFUTED / UNRESOLVED / UNVERIFIABLE / AMBIGUOUS, with **equivalence tests (TOST)** for "independent of X" claims (§10) | Failing to reject a correlation does not prove independence |
| 10 | Verifier validation | Implicit ("exact — no dispute possible") | **Planted-claims calibration set** + two independent executors + unit/property tests (§10.7) | "Who verifies the verifier" is the first reviewer question |
| 11 | Parser validation | κ ≥ 0.6 on 200 rationales | κ / Krippendorff α + Gwet AC1, **plus prediction-powered inference (PPI)** to bias-correct aggregate metrics (§9.3–9.4) | Turns a validated-but-imperfect parser into unbiased estimates with honest CIs |
| 12 | Power analysis | Two-proportion test | Two-proportion start + **design effects for clustering** + GLMM simulation (§12.4); worked numbers included | Claims are nested in rationales nested in formulas; ignoring this overstates power |
| 13 | Multiplicity | Bonferroni/BH across everything | Small **confirmatory family (Holm)** + labelled exploratory family (BH/BY) (§12.5) | Correcting across every cell destroys power and is not what reviewers ask for |
| 14 | Baselines | LLM judge + reconstruction | Six baselines incl. **claim-level LLM judge**, **execution-augmented LLM judge**, NLI, human experts (§13) | Separates "claim decomposition" from "execution access" as the source of any advantage |
| 15 | Data | Qlib/Yahoo vs CRSP | Tiered free / low-cost / paid stack per market, cross-source validation, PIT index membership, and an explicit answer on **paid-grade data that is legitimately free** (§5.3–5.8) | First run is free-only; the paper still needs a credible data story |
| 16 | Behavioral ground truth | Chen–Zimmermann (319) + FF5 | **Same-panel price-volume reference library** as primary, plus OSAP, JKP, Ken French, q-factors, CH-3/CH-4 (§5.5) | Mined formulas here are OHLCV-only; most CRSP/Compustat characteristics (value, profitability) cannot be expressed by them, and identifier crosswalks need paid data |
| 17 | Baseline attribution | "LLM-judged consistency as used in AlphaAgent and Alpha-R1" | AlphaAgent (LLM-scored description–expression consistency), XALPHA (ex-ante tri-alignment), QuantaAlpha (semantic consistency), Alpha Jungle (LLM-ranked interpretability); Alpha-R1 reclassified as a *consumer* of LLM-written factor descriptions (§2.1) | Accurate attribution |
| 18 | Free run feasibility | Not addressed | Free LLM access routes, run-size arithmetic, a pilot→main schedule (§4.4, §17) | Requested |

---

## 0.1 Decision matrix (one table to plan the whole study)

| Stage | Option A | Option B | Option C | First (free) run |
|---|---|---|---|---|
| Universe | CSI500 + S&P 500 (matches AlphaAgent, AlphaLogics) | CSI300 + CSI1000 (matches Alpha Jungle) | + JPX (Kaggle/J-Quants) or crypto as a data-quality-independent replication | ✅ A, with CSI300 as secondary |
| CN prices | Community Qlib bins (chenditc/investment_data) | BaoStock direct | AkShare / Tushare Pro | ✅ A, cross-checked against B |
| US prices | yfinance + historical S&P 500 membership | + Tiingo/EODHD/FMP free tiers for delisted names | CRSP / Norgate `[PAID]` | ✅ A + partial B, survivorship disclosed |
| Formula language | Qlib expression DSL | Alpha101/WorldQuant-style | Python code (CogAlpha/AlphaQT style) | ✅ A as canonical, translators from B |
| Formula sources | LLM hypothesis-first / formula-first | GP + random grammar (complexity-matched) | In-the-wild rationales from published frameworks | ✅ A + B; C if time allows |
| Narrator models | Local open-weight (Kaggle/Colab GPUs) | Free API tiers | Frontier closed models `[PAID]` | ✅ A + B |
| Claim extraction | Rule-based | Frozen LLM parser (different family) | Ensemble + human adjudication | ✅ B + rules for slots + human sample |
| Mechanistic truth | Static AST analysis | Perturbation ("nudge") + metamorphic tests | SMT solver on fragments | ✅ A + B |
| Behavioral truth | Same-panel reference-signal correlations | Return-level factor regressions | Spanning tests / double-selection LASSO | ✅ A + B; C for originality claims |
| Inference | Cluster bootstrap | GLMM (logit) with crossed random effects | Bayesian hierarchical | ✅ A + B |
| Aggregate correction | None (parser assumed correct) | PPI with human-labelled subset | — | ✅ B |

---

## 1. Research questions, hypotheses, contribution

### 1.1 Problem statement
LLM alpha-mining systems now routinely attach a natural-language rationale ("captures short-term reversal in illiquid names", "a 20-day volume-confirmed momentum signal", "novel, not explained by existing factors") to every formula they produce, and some systems feed those descriptions back into generation, filtering or screening. Whether those sentences are *true of the formula* is almost never checked by execution. Existing checks are holistic LLM judgments of consistency, LLM rankings of interpretability, or reconstruction of the formula from the explanation (§2.1). None decomposes a rationale into atomic claims and executes the formula to decide each one.

### 1.2 Research questions and pre-registered directional hypotheses

| RQ | Question | Primary contrast | Pre-registered expectation (to be tested, not assumed) |
|---|---|---|---|
| RQ1 | Does *how* a formula was produced change rationale fidelity? | Hypothesis-first (P1) vs formula-first LLM (P2) vs post-hoc narration of non-LLM formulas (P3), complexity-matched | H1: claim precision is lower for P3 than for P1 |
| RQ2 | Does giving the narrator evidence or execution access improve fidelity? | Access A0 (formula only) vs A1 (diagnostics table) vs A2 (sandboxed execution tool) | H2: A1/A2 raise precision for behavioral/performance claims more than for mechanistic claims |
| RQ3 | How much rationale content is recalled rather than read? | Known (K) vs semantics-altering perturbations (SA); K vs misleading-label (NL); K vs semantics-preserving (SP) | H3a: precision drops from K to SA and the drop is concentrated in claims about the perturbed property ("anchoring"); H3b: NL labels pull rationales toward the label; H3c: SP ≈ K |
| RQ4 | Do current LLM-judge consistency checks detect false rationales? | Judge scores vs execution-verified truth | H4: holistic judge scores discriminate rationales containing refuted claims poorly (AUROC well below 0.9) |
| RQ5 | Is fidelity on Novel formulas different from Known formulas? | N vs K (complexity-matched) | Two-sided (no direction pre-specified) |
| RQ6 | Is rationale fidelity related to model scale/family, and to out-of-sample performance? | Size ladder within one family; formula-level precision vs OOS IC | Exploratory only |

### 1.3 Contribution and its boundaries
1. A claim-level, execution-verified audit protocol for factor rationales, with a validated verifier and a validated parser.
2. A counter-recall design that separates reading from recalling at the level of *rationale claims* (prior leakage probes in alpha mining test whether models recall *good formulas*, not whether their explanations are recalled).
3. A head-to-head validity test of the consistency checks the field currently uses.

To our knowledge (search log in Appendix G, run on 6 Oct 2026), no published or preprint work does (1) and (2). The statement must be re-checked at submission time.

---

## 2. Literature map (2023 → October 2026)

### 2.1 LLM alpha-mining systems and where "rationale checking" appears

| Work (venue/ID) | Data, universe, split | LLMs | How rationale/consistency is handled | Relevance to this study |
|---|---|---|---|---|
| AlphaGen (KDD 2023) | Qlib CN | none (RL) | n/a | Non-LLM formula source / baseline |
| FAMA (ACL Findings 2024) | CN | LLM agent | In-context examples, "chain-of-experience" | LLM baseline in Alpha Jungle |
| **AlphaAgent** (KDD 2025; arXiv 2502.16789) | CSI500 via BaoStock, S&P 500 via Yahoo; OHLCV only; train 2015-01–2019-12, valid 2020, test 2021-01–2025-01; Qlib backtests | GPT-3.5-turbo base; also Qwen-Plus, DeepSeek-R1 | **LLM-scored consistency** C = 0.5·c1(hypothesis, description) + 0.5·c2(description, expression), each in [0,1]; AST-subtree originality vs Alpha101 | Baseline B1 replicates c2; the closest prior "rationale check" |
| **RD-Agent(Q)** (NeurIPS 2025; arXiv 2505.15155) | Qlib | GPT-4-class | Research/development/feedback loop; hypotheses drive code | Source of in-the-wild hypotheses (P4) |
| **Alpha Jungle** (AAAI 2026; arXiv 2505.11122) | Qlib CN: CSI300, CSI1000, train 2011–2020, test 2021–2024-11; Qlib US S&P 500, train 2007–2015, test 2016–2020-10 | GPT-4.1 + four others | Two-step "alpha portrait" (name + description + pseudo-code) then formula; **LLM-judged overfitting risk**; interpretability = **rankings by three LLMs**, no human study; leakage probe = asking 3 LLMs for high-performing CSI300 formulas (10 each) and comparing with random | Its leakage probe tests recall of *good formulas*; our RQ3 tests recall inside *rationales* |
| CogAlpha (ACL 2026; arXiv 2511.18850) | 5 datasets, 3 markets | multiple | Code-level alphas, LLM-driven evolution, multi-agent quality checking | Code-based formulas; option C of the formula language |
| **AlphaLogics** (arXiv 2603.20247, 2026) | CSI500, S&P 500; train 2015-01–2019-12, valid 2020, test 2021-01–2024-12 | Gemini-2.5-Flash for reconstruction | **Reconstruction test**: rebuild formula from explanation, equivalent if rankings/trends match in >90% of cases; reports >95% (math) and 92–99% (financial explanations) consistency | Baseline B4. Reconstruction tests sufficiency of the explanation, not truth of each claim |
| FactorMiner (arXiv 2602.14670, 2026) | multiple assets/markets | agent with skills + experience memory | Memory of successful patterns/failure constraints | Context |
| QuantaAlpha (arXiv 2602.07085, 2026) | — | — | Enforces semantic consistency across hypothesis, factor expression and code | Another consistency check to compare against if code is released |
| FactorEngine (arXiv 2603.16365, 2026) | — | LLM + Bayesian search | Program-level, knowledge-infused | Context |
| **XALPHA** (arXiv 2607.08332, Jul 2026) | CSI300 | multi-brain agent | "Micro Brain" verifies **ex-ante tri-alignment** among hypothesis idea, code logic and financial plausibility | Closest new consistency mechanism; LLM-verified, not execution-verified |
| Alpha-R1 (arXiv 2512.23515) | multiple pools; Qlib backtest | Qwen3-8B + GRPO | Screens Alpha101 factors by reading **LLM-written semantic factor profiles** | A downstream *consumer* of rationales: if profiles are false, screening reasons over false premises. Its profiles are an in-the-wild audit target (P4) |
| Chain-of-Alpha (arXiv 2508.06312) | — | — | Dual-chain generation/selection | Context |

### 2.2 Benchmarks and meta-evaluation of LLM quant work

| Work | Finding relevant here |
|---|---|
| **AlphaBench** (ICLR 2026) | Benchmarks LLMs on formulaic alpha mining; reports LLMs are unreliable as evaluators of alpha quality; uses an LLM-as-judge for instruction faithfulness |
| **AlphaQT-Bench** (Findings of ACL 2026) | 270 instruction-grounded tasks, 12 LLMs; multi-layer protocol: executability → causality (dynamic truncation test, 5 truncations) → functional accuracy vs expert "golden code" (corr ≥ 0.999 or NRMSE ≤ 0.001) → structural compliance; large gap between "runs" and "correct"; documents look-ahead leakage and models over-relying on remembered templates. Checks *code vs instruction*; we check *text vs code* |
| AlphaEval (arXiv 2508.13174) | Evaluation framework for formulaic alpha mining |
| FINSABER (arXiv 2505.07078) | 20 years, 100+ symbols, delisted names included: earlier LLM-trading advantages shrink under longer, broader, bias-mitigated tests |
| Kong et al. 2026, *Evaluating LLMs in Finance Requires Explicit Bias Consideration* (arXiv 2602.14233) | Five biases: look-ahead, survivorship, **narrative**, objective, cost; review of 164 papers (2023–2025) finds no bias discussed in more than 28% of them; proposes a structural-validity checklist |
| Survey: Zhang et al. 2025, *A survey on LLM-based alpha mining* (FITEE 26(10)) | Taxonomy for related-work section |

### 2.3 Memorization and look-ahead in financial LLMs

| Work | Finding |
|---|---|
| Lopez-Lira, Tang & Zhu, *The Memorization Problem* (SSRN 5217505 / arXiv 2504.14765; revised Apr 2026) | LLMs recall exact pre-cutoff economic/market values; instructions to respect dates and entity masking do not stop it; no recall after the cutoff; formal non-identification argument |
| Li et al., *Profit Mirage* (arXiv 2510.07920) | Backtested gains of LLM trading agents largely disappear past the model's knowledge cutoff; FinLake-Bench; counterfactual perturbations as mitigation |
| He, Lv, Manela & Wu, ChronoBERT/ChronoGPT (arXiv 2502.21206) and ChronoGPT-Instruct (arXiv 2510.11677) | Chronologically consistent "vintage" models trained only on text available up to each cutoff; instruction-tuned versions released on Hugging Face (manelalab) — usable as a no-look-ahead control narrator if they can follow the task |
| Glasserman & Lin (2024, *J. Financial Data Science*); Sarkar & Vafa (2024, SSRN) | Earlier evidence and tests for look-ahead bias in GPT-based return prediction |

### 2.4 Faithfulness of explanations, claim verification, counterfactual task variants

| Work | Use in this pipeline |
|---|---|
| Turpin et al. (NeurIPS 2023); Lanham et al. (2023); Chen et al. (2025, reasoning models) | Motivation: self-explanations can be unfaithful |
| **Matton et al., *Walk the Talk?*** (ICLR 2025) | Defines faithfulness as mismatch between concepts an explanation *implies* are influential and those that truly are; counterfactual concept edits + hierarchical Bayes. Our "driver coverage" metric is the formula-level analogue, where true influence is computed exactly |
| Atanasova et al. (ACL 2023) | Counterfactual faithfulness tests for NL explanations |
| *From Plausible to Actionable* (arXiv 2607.15957, 2026) | Faithfulness evaluation should account for LLM non-determinism, not single runs → our k-sample design |
| FActScore (EMNLP 2023); SAFE (NeurIPS 2024); **Claimify** (ACL 2025) | Atomic-claim decomposition; Claimify extracts claims only when the interpretation is unambiguous → our AMBIGUOUS bucket |
| **CodeCrash** (NeurIPS 2025) | 1,279 code-reasoning items, 17 LLMs: misleading natural-language cues degrade output prediction by ~23% on average → our NL (misleading-label) pool |
| Wu et al., *Reasoning or Reciting?* (NAACL 2024); GSM-Symbolic (ICLR 2025) | Counterfactual/perturbed task variants as a memorization control → design template for SA/SP pools |
| CRUXEval (ICML 2024) | Code-execution reasoning benchmark; context for mechanistic-claim difficulty |
| POPPER (Huang et al., ICML 2025) | Sequential falsification with e-values for hypothesis validation — optional alternative for behavioral claims (§10.3) |

### 2.5 Statistics for LLM evaluation and for finance
- LLM evals: Miller (2024, arXiv 2411.00640) — clustered standard errors (can exceed naive SEs by >3×), paired designs, power; Bowyer et al. (ICML 2025 position) — avoid CLT intervals with few hundred items; Card et al. (EMNLP 2020) — underpowered NLP comparisons; Angelopoulos et al. (*Science* 2023) PPI and Boyeau et al. (ICML 2025) "AutoEval done right" — bias-corrected estimates from many automatic labels plus few human labels.
- Finance: Harvey, Liu & Zhu (RFS 2016) t > 3.0; Harvey & Liu (JPM 2015) haircut Sharpe; Bailey & López de Prado (JPM 2014) Deflated Sharpe Ratio; Bailey, Borwein, López de Prado & Zhu (J. Comput. Finance 2017) PBO/CSCV; White (2000) Reality Check; Hansen (2005) SPA; Romano & Wolf (2005) stepdown; Chordia, Goyal & Saretto (RFS 2020) mass-generated signals as a data-mining benchmark; Feng, Giglio & Xiu (JF 2020) double-selection for "is this factor new"; Jensen, Kelly & Pedersen (JF 2023) global replication; Lakens (2017) equivalence testing.

---

## 3. Study design overview

### 3.1 Units of analysis (nested, partly crossed)
`model` × `formula` × `condition (pool/protocol × access level × prompt variant)` → `rationale sample (k per cell)` → `atomic claim` → `verdict`.
Formulas are crossed with models in Arm B (every model narrates the same formulas); in Arm A (P1, P2) each model authors its own formulas, so formula is nested in model. The analysis model (§12) reflects this.

### 3.2 Three experimental arms

| Arm | Purpose | Formula source | Narration | Answers |
|---|---|---|---|---|
| **A — Authorship** | Does provenance and evidence access change fidelity? | P1 hypothesis-first (LLM), P2 formula-first (LLM), P3 GP + random grammar (non-LLM, complexity-matched), P4 in-the-wild (optional) | Access A0, A1, A2 | RQ1, RQ2, RQ6 |
| **B — Counter-recall** | Reading vs recalling | Public libraries (Alpha101, GTJA-191, Alpha158) and controlled variants; Novel pool | Access A0 (primary), A2 on a subset | RQ3, RQ5 |
| **C — Judge validity** | Do current checks catch false rationales? | Re-uses all rationales from A and B | Baseline judges B1–B6 | RQ4 |

### 3.3 Independent variables

| Variable | Levels | Notes |
|---|---|---|
| Narrator model | ≥ 5 models from ≥ 4 families (§4) | One family with ≥ 3 sizes |
| Protocol (Arm A) | P1, P2, P3a (GP), P3b (random grammar), P4 (optional) | P1 split into *raw* (all valid formulas) and *mined* (kept after a short search loop) strata |
| Pool (Arm B) | K, K-named, SP, SA-sign, SA-window, SA-field, NL, N | §7.2 |
| Access | A0 formula + glossary; A1 + diagnostics table; A2 + sandboxed execution tool | §8.2 |
| Prompt variant | Guided (asks for mechanism, drivers, horizon, resemblance, novelty, conditions) vs Minimal ("explain the rationale") | Guided is primary; Minimal on a 25% subsample |
| Decoding | T = 0.7 with k = 3 samples (primary); T = 0, k = 1 (reference) | Reasoning models at their default reasoning setting, recorded |
| Market | CN (CSI500 primary, CSI300 secondary), US (S&P 500) | Formula semantics are market-independent; behavioral truth is computed per market |

### 3.4 Pipeline

```
  ┌───────────────┐   ┌───────────────────────────┐   ┌──────────────────────────┐
  │ Data layer     │──►│ Typed DSL + 2 independent │──►│ Verifier engine           │
  │ (§5)           │   │ executors (§6)             │   │ static / nudge / stats    │
  └───────────────┘   └─────────────┬─────────────┘   └────────────▲─────────────┘
                                     │                               │
  ┌───────────────┐   ┌─────────────▼─────────────┐   ┌────────────┴─────────────┐
  │ Formula pools  │──►│ Narrators (LLMs) ×         │──►│ Claim parser + codebook   │
  │ A: P1–P4       │   │ access A0/A1/A2 × k (§8)  │   │ (§9) → predicates         │
  │ B: K…N (§7)    │   └─────────────┬─────────────┘   └────────────┬─────────────┘
  └───────────────┘                 │                               │
                                     ▼                               ▼
                       ┌───────────────────────────┐   ┌──────────────────────────┐
                       │ Baseline judges B1–B6     │──►│ Metrics + GLMM + PPI      │
                       │ (§13)                      │   │ (§11–12)                  │
                       └───────────────────────────┘   └──────────────────────────┘
```

### 3.5 Not a full factorial (by design)
A full factorial (models × 8 pools × 3 access × 2 prompts × 2 markets) is unaffordable and unnecessary. The run plan is:
- Arm B at A0, guided prompt, both markets' formula semantics are identical so narration happens once per formula; behavioral verdicts are computed per market. A2 on a 30-formula subset of K and SA only.
- Arm A at A0/A1/A2, guided prompt; minimal prompt on 25% of formulas at A0.
- Arm C re-scores existing rationales (no new narration except B2–B4 judge calls).

---

## 4. LLM selection

### 4.1 Criteria (v1 criteria kept, five added)

| Axis | Requirement | Rationale |
|---|---|---|
| Provider diversity | ≥ 2 closed-weight families, ≥ 2 open-weight families, ≥ 3 organizations | Rules out one lab's training mix as the explanation |
| Size ladder | ≥ 3 sizes in one open-weight family | Separates capability from family style (RQ6) |
| Reasoning vs non-reasoning | ≥ 1 reasoning-mode model and ≥ 1 non-reasoning model | Reasoning traces change explanation behavior; CodeCrash reports pathological self-reflection under misleading cues |
| Tool use | Models used at A2 must support function/tool calling | Otherwise A2 is not a fair condition |
| Version pinning | Exact model string, provider, access date, parameters, any system fingerprint; for local models the weights file hash, quantization, inference engine and version | Silent model updates and quantization change outputs |
| Training cutoff | Documented per model (model card), plus an empirical probe (ask dated facts) | Needed for the post-cutoff window (§5.2) and Novel-pool timing (§7.2) |
| Data-use terms | Record whether the endpoint may use prompts for training/product improvement | The Novel pool must never be sent to an endpoint that may train on it before the study ends (§7.2) |
| Role separation | Author, narrator, parser and judge roles filled by *different* families where they interact | Avoids self-preference in parsing and judging |

**Do not fix brand names in the plan.** Pin the roster against model cards on the run date. Families that exist as of late 2026 and are commonly used: open-weight — Qwen, Llama, Gemma, Mistral, DeepSeek, GLM, gpt-oss, Kimi; closed — Gemini, GPT, Claude, Grok. Treat this list as examples only.

### 4.2 Role assignment (example template)

| Role | Requirement | Example policy |
|---|---|---|
| Authors (P1/P2) | Same set as narrators | Each narrator authors its own P1/P2 formulas |
| Narrators | Full roster | — |
| Cross-narrators | Narrate formulas authored by another family | 1 cross-narration per P2 formula (self vs other narrator) |
| Parser | A family *not* among narrators, or an ensemble of two families | Temperature 0, JSON schema, frozen version |
| Judges (B1–B3) | A family not among narrators for the main analysis; same-family judges for a self-preference check | — |

### 4.3 Determinism, logging, provenance
- Non-determinism persists even at temperature 0 on many serving stacks; the design therefore uses k = 3 samples and reports between-sample variance and a sample-level agreement statistic.
- Log every call as JSONL: request (system + user messages, tool schema), response (text, tool calls, reasoning tokens if exposed), parameters, timestamps, latency, token counts, model fingerprint, and a SHA-256 of the prompt template.
- Cache responses keyed by (model, prompt hash, seed/sample index); never regenerate silently.

### 4.4 First run with free LLM access `[FREE]` (verify on run date)
Free tiers change without notice; record what you used.

| Route | What it gives you | Caveats |
|---|---|---|
| Local open-weight inference on free GPUs ✅ | Kaggle notebooks expose 2× T4 (16 GB each); Kaggle documents 12-hour GPU sessions and shows a weekly GPU quota in account settings (commonly reported ≈ 30 h/week). Google Colab free T4; Lightning AI free monthly GPU hours | Quantized 7–32B models fit; throughput limits main-run size — measure tokens/s in the pilot. Best route for the Novel pool (data never leaves your session) |
| Google AI Studio (Gemini API free tier) | Free access to Flash-class Gemini models | Google moved Pro models to paid in April 2026 and now shows free-tier limits only in the console; check data-use terms of the free tier |
| Groq, Mistral (free mode), Cohere (trial key) | Free tiers without billing | Per-minute/per-day caps; model lists change |
| OpenRouter `:free` model variants | One key, many open-weight models | Daily request cap tied to purchase history; provider behind a model can change |
| Cerebras | Sign-up credit | Model list pruned frequently |

**Practical policy for the free run.**
1. Run ≥ 3 open-weight families locally (including the size ladder) — these give exact version control.
2. Add 1–2 closed families through free tiers if available in your region; if closed-family coverage is incomplete, label the free run as the **pilot/first study** and complete closed-family coverage in the paid run.
3. Keep a provider-independent cache so a removed free model does not invalidate collected data.

### 4.5 Paid upgrade `[PAID]`
Frontier closed models via official APIs with batch endpoints (cheaper, asynchronous), plus prompt caching for the fixed glossary/system text.

---

## 5. Data

### 5.1 Data integrity needed depends on the claim type

| Claim type | What data is used for | Sensitivity to survivorship / corporate-action errors | Minimum acceptable source |
|---|---|---|---|
| C1 Mechanistic (inputs, sign, lookback, monotonicity) | Realistic contexts for perturbation tests | Very low — truth is a property of the formula | Any clean OHLCV panel `[FREE]` |
| C6 Identity ("this is Alpha#12") | Numerical equivalence to library formulas | Very low | Any clean panel `[FREE]` |
| C2/C3 Behavioral, originality | Correlations with reference signals and factor returns | Moderate — biases both sides similarly if computed on the same panel | Same-panel references `[FREE]`; OSAP/JKP/French for US return-level checks |
| C4 Performance | Out-of-sample IC, Sharpe, DSR, PBO | High | Survivorship-aware, PIT universe; post-cutoff window `[PAID]` preferred; `[FREE]` with disclosed bias |

### 5.2 Universes, windows and splits
- **Primary universes:** CSI500 and S&P 500 (same as AlphaAgent and AlphaLogics, so baselines are comparable). **Secondary:** CSI300 (Alpha Jungle, XALPHA).
- **Split (comparability):** train 2015-01-01 → 2019-12-31; validation 2020; test 2021-01-01 → 2024-12-31. **Alternative long-history split (CN):** train 2011–2020, test 2021–2024-11 (Alpha Jungle).
- **Which window decides behavioral truth?** Primary = training window (it describes what the formula *does* on data the narrator could plausibly know about); replication = test window. A claim whose verdict flips between windows is tagged *regime-dependent* and reported separately.
- **Post-cutoff holdout (new).** H_post = [latest training cutoff among narrators + 1 month, data end]. Used for (i) performance claims, (ii) checking whether recalled formulas' "known" performance survives. If H_post is shorter than 6 months it is reported as exploratory.
- **Rolling robustness:** annual re-estimation of behavioral statistics (2015…2024) to show claim verdict stability.
- **Labels:** signal at day t uses data up to and including t's close; forward return from t+1 open (or t close) to t+h close; h ∈ {1, 5, 10, 20}. State the convention once; AlphaAgent predicts next-day returns, Alpha Jungle uses 10- and 30-day horizons.

### 5.3 China A-shares — free stack `[FREE]`

| Option | Source | Strengths | Caveats |
|---|---|---|---|
| **CN-A ✅** | Community Qlib bins from `chenditc/investment_data` (GitHub releases; raw tables on DoltHub). Qlib's own README currently points users here because the official Qlib dataset is temporarily disabled | Qlib-ready; merges several sources (Wind and Caihui static snapshots through 2019, Tushare, AkShare, Yahoo via Qlib's collector, BaoStock) with cross-validation; explicitly aims to fill delisted-company data; each release ships a manifest and a validation script | Provenance/licensing of the vendor-derived tables (Wind, Caihui) is not documented in the repo — disclose, do not redistribute raw tables, release code that rebuilds from sources |
| CN-B | BaoStock (free, no registration or API key) | Daily K-lines with adjustment flags; `query_all_stock(day=…)` lists securities trading on a given date (point-in-time universe, including names that later delist); CSI300/CSI500 constituent queries (check how far back the `date` argument returns historical membership); quarterly financial statements (Qlib's point-in-time collector builds on it). AlphaAgent's CSI500 data came from BaoStock | Throughput limits; constituents refreshed weekly |
| CN-C | AkShare (open-source scraper library); Tushare Pro (free registration, point-based access) | Independent second/third source for cross-validation | Scraped endpoints change; some Tushare endpoints (e.g., index weights) need higher point levels — verify |

**Cleaning and validation (CN).**
1. Use CN-A as the panel; rebuild daily returns independently from CN-B for all CSI500/CSI300 constituents.
2. Flag stock-days with |r_A − r_B| > 5 bp; report the mismatch rate by year; exclude flagged days from behavioral statistics (keep for mechanistic contexts).
3. Point-in-time membership from dated constituent queries; never use today's constituent list for history.
4. Market rules that matter for performance claims only: daily price limits (main board ±10%, STAR/ChiNext ±20%, ST ±5%), suspensions, ST/*ST flags, T+1. Treat limit-locked days as untradable in backtests.
5. VWAP: available in Qlib CN bins; if a source lacks it, use amount/volume and document.

### 5.4 United States — free stack `[FREE]`

| Component | Option (free) | Notes |
|---|---|---|
| Prices for current constituents | yfinance; Qlib's Yahoo collector | Survivorship-biased (delisted names largely missing). AlphaAgent's S&P 500 data came from Yahoo; Alpha Jungle's US experiment on Qlib US data stops in Oct 2020, so 2021+ windows need your own crawl |
| Historical S&P 500 membership ✅ | `fja05680/sp500` (historical constituents since 1996, maintained on GitHub) | Turns "today's S&P 500" into a point-in-time universe |
| Prices for removed/delisted names | Tiingo free tier; EODHD free key (delisted symbol lists via `delisted=1`; beware reused tickers, which EODHD suffixes `_old`); Financial Modeling Prep free plan (rate-limited; includes a delisted-companies list); Stooq bulk files | Coverage is partial — measure it |
| Fundamentals / shares outstanding | SEC EDGAR Financial Statement Data Sets and XBRL frames | Filing-date timestamps make them point-in-time friendly; not needed for OHLCV formulas, needed for size/turnover references |
| VWAP | Not in Yahoo daily data | Use (H+L+C)/3 proxy and say so, or drop VWAP from the US operator set |

**Survivorship handling (US, free run).**
- Report **coverage** = share of point-in-time member-days with valid prices, by year.
- Run performance analyses on the covered subset and add a bound: re-run with Shumway (1997)-style delisting-return imputation (e.g., −30% for performance-related delistings) as a pessimistic sensitivity.
- Keep fidelity results on the main (covered) panel — their validity does not hinge on survivorship (§5.1).

### 5.5 Reference libraries for behavioral ground truth

**Primary (both markets) ✅ — same-panel price-volume reference library.** Computed on exactly the same stocks/dates/adjustments as the audited formulas:
- Alpha158 (Qlib; definitions in `qlib/contrib/data/handler.py`), Alpha101 (Kakushadze 2016), GTJA-191 (Guotai Junan 2017 report). Many community implementations disagree on details (e.g., `decay_linear`, ties in `ts_rank`) — implement from the original texts and unit-test against published examples.
- Canonical characteristics with fixed operational definitions (Appendix C): short-term reversal (1d, 5d, 21d), momentum (12-1 and 6-1 months), 52-week-high proximity, total and idiosyncratic volatility (20d/60d), MAX (max daily return, 21d), Amihud illiquidity (21d), abnormal volume (volume / 60d average), price–volume correlation, market beta (252d), return skewness, close-location value, overnight vs intraday return.

**United States — public factor/portfolio libraries `[FREE]`.**
- Ken French Data Library: FF5 (daily/monthly), momentum, short- and long-term reversal.
- Open Source Asset Pricing (Chen & Zimmermann, CFR 2022): the paper reproduces 319 characteristics; current releases provide predictor portfolio returns (≈ 212 predictors) free and ≈ 209 firm-level signals free by PERMNO, while three CRSP-based signals (price, size, short-term reversal) require WRDS. A PERMNO→ticker crosswalk needs CRSP, so in the free run use **portfolio returns** (return-level regressions), not firm-level merges. Python: `pip install openassetpricing`; R: `OpenSourceAP.DownloadR`.
- JKP Global Factor Data (Jensen, Kelly & Pedersen, JF 2023): 153 factors in 13 themes across 93 countries, free under a non-commercial license (R: `tidyfinance::download_data_jkp`).
- Hou–Xue–Zhang q-factors (global-q.org); AQR data sets.

**China `[FREE]`.**
- CH-3 factors (Liu, Stambaugh & Yuan, JFE 2019) from Stambaugh's website: monthly CH-3 updated; daily CH-3 and CH-4 (monthly/daily) through 2021; built from Wind data items (mapping documented in their online appendix).
- JKP country factors: check whether China is in the availability manifest for your factor set.
- Self-computed Alpha158/GTJA-191 on the CN panel (primary).

### 5.6 Optional robustness universes `[FREE]`
- **Japan (TSE):** Kaggle "JPX Tokyo Stock Exchange Prediction" competition data supplied by JPX/J-Quants (≈ 2,000 issues; prices, financial statements, trading by investor type, index options). Check the competition's rules for post-competition research use. J-Quants API offers a free plan with delayed, limited history — verify.
- **Crypto:** exchange public data archives (e.g., Binance's public data site) give daily/minute bars including pairs that were later delisted if you enumerate them. Useful as a "data-quality-independent" replication of mechanistic fidelity; different microstructure, so not for behavioral claims about equity anomalies.

### 5.7 Paid / institutional sources `[PAID]` — what each one fixes

| Source | Market | Fixes |
|---|---|---|
| CRSP (via WRDS) | US | Survivorship, delisting returns, PERMNO identity, shares outstanding |
| Compustat (via WRDS) | US/global | Point-in-time fundamentals |
| Norgate Data | US (also AU/CA) | Survivorship-free EOD incl. delisted, point-in-time S&P 500 / Russell 3000 membership flags; free trial |
| Sharadar (Nasdaq Data Link) | US | Survivorship-free prices + fundamentals |
| EODHD / Tiingo paid plans | US/global | Delisted coverage, historical constituents |
| CSMAR (direct or WRDS), Wind, RESSET, iFinD, Choice | CN | Survivorship-free A-share data, PIT financials, index weights |
| Tushare Pro higher tiers; JoinQuant / RiceQuant data | CN | Index weights, adjusted factors, PIT fundamentals |
| Datastream, Compustat Global, JKP stock-level (WRDS) | Global | Cross-market replication |

### 5.8 "Paid-grade data that is free": what is legitimate and what is not

You asked whether paid data is available for free on GitHub, Kaggle or elsewhere. The answer splits into seven categories:

| # | Category | Examples | Use? |
|---|---|---|---|
| 1 | **Free academic releases derived from licensed data** | OSAP (CRSP/Compustat-derived signals and portfolios), JKP factors, Ken French portfolios (CRSP-based), q-factors, AQR data, Stambaugh CH-3/CH-4 (Wind-derived), Pástor–Stambaugh liquidity | ✅ Yes — this is the main legitimate route to "paid-grade" information at zero cost; cite the releases |
| 2 | **Community-merged datasets containing vendor snapshots** | `chenditc/investment_data` (Wind and Caihui tables through 2019 + free sources) | ✅ With disclosure; don't redistribute raw vendor tables; publish rebuild scripts |
| 3 | **Use-in-place platforms** (vendor data usable free inside the platform, not downloadable) | QuantConnect: free tier, survivorship-bias-free US equities since 1998 including delisted names, usable in the cloud research environment (external download/local use of its data needs paid access). WorldQuant BRAIN: free sign-up, a very large catalogue of data fields (WorldQuant's postings cite 150,000–180,000+), browser-based alpha simulation | ✅ For **performance re-checks** of formulas in-platform (compute statistics there, export only results). Not suitable as the primary panel because perturbation tests need local execution |
| 4 | **Competition data released by data owners** | Kaggle: JPX (J-Quants), Jane Street, Optiver, Ubiquant (anonymized features), G-Research (crypto) | ✅ Within each competition's data rules; anonymized-feature sets are of little use for OHLCV formula audits |
| 5 | **Free trials / free tiers of paid vendors** | Norgate trial, EODHD free key, Tiingo free tier, FMP free plan | ✅ For coverage audits (e.g., how many delisted names your free panel misses); read license terms before publishing derived numbers |
| 6 | **Institutional access you may already have** | University library: WRDS (CRSP, Compustat, CSMAR, Datastream), Bloomberg/Wind terminals | ✅ Ask your library first — often the cheapest upgrade |
| 7 | **Unauthorized re-uploads** of CRSP/Compustat/Wind/CSMAR files on Kaggle, GitHub, cloud drives or chat groups | (not listed) | ❌ Do not use: license violation; unknown provenance (you cannot check adjustments or survivorship); cannot be cited or released in a reproducibility package; takedowns break reproducibility; data editors and reviewers may reject the paper |

### 5.9 Data QA checklist (run before any narration)
- [ ] Adjusted vs raw prices identified; adjustment factors consistent across sources.
- [ ] No duplicate (stock, date); trading calendar per exchange; holidays excluded.
- [ ] Zero/negative prices, zero-volume days, suspensions flagged.
- [ ] Cross-source return mismatch rate reported (CN-A vs CN-B; US Yahoo vs Tiingo/EODHD on overlap).
- [ ] Point-in-time membership applied; coverage table per year.
- [ ] Every downloaded file hashed (SHA-256) and listed in `data/MANIFEST.csv` with source URL and access date.
- [ ] Qlib's `scripts/check_data_health.py` run on the bins and output archived.

### 5.10 Licensing table (fill in on run date)

| Source | License / terms | Redistribution of raw data | What you release |
|---|---|---|---|
| chenditc/investment_data | Repo Apache-2.0; vendor tables undocumented | No (vendor-derived parts) | Rebuild scripts + derived signals |
| BaoStock / AkShare / Tushare | Provider terms | Check | Scripts |
| Yahoo (yfinance) | Yahoo terms of service | No | Scripts |
| OSAP / Ken French / q-factors | Free academic | Usually yes with citation — check | Citation + download script |
| JKP | Non-commercial | Check | Script |
| Kaggle competition data | Competition rules | Usually no | Script + pointer |
| CRSP / Compustat / CSMAR / Wind | Subscription | No | Code + derived aggregates only if license allows |

---

## 6. Formula representation, operator library, executable semantics

### 6.1 Formula language options

| Option | Pros | Cons | Use |
|---|---|---|---|
| **A ✅ Qlib expression DSL** (`Mean($close, 5)`, `Ref($close, 1)`, `Corr($close, $volume, 10)`, …) | Native in Qlib; Alpha158 is written in it; used by most baselines | Qlib evaluates expressions per instrument: its `Rank(x, N)` is a *rolling time-series* rank, and cross-sectional ranking/normalization is applied as a data processor, not an expression operator. Alpha101/GTJA formulas need cross-sectional operators, so the DSL adds `CSRank`/`CSZScore` explicitly | Canonical surface form (extended) |
| B Alpha101/WorldQuant style (`ts_rank`, `decay_linear`, `rank`, `delta`) | Needed to present public formulas verbatim (pool K) | Ambiguous details across implementations | Input/translation layer |
| C Python/pandas code (CogAlpha, AlphaQT-Bench) | Most expressive | Static analysis much harder; hidden look-ahead (AlphaQT-Bench) | Optional extension, not first run |

**Recommendation:** one internal *typed expression tree* with parsers from A and B and serializers to A, B and plain math. All verification runs on the tree.

### 6.2 Typed operator specification (excerpt — full table in the repo)
Each operator carries: arity, argument types, window/lag parameters, time-series vs cross-sectional nature, **monotonicity per argument**, output range, NaN/warm-up behavior, unit rule.

| Operator | Kind | Monotone in arg 1? | Output range | Warm-up | Unit rule |
|---|---|---|---|---|---|
| `Ref(x, d)` (lag) | TS | ↑ | as x | d | as x |
| `Mean(x, n)`, `Sum(x, n)` | TS | ↑ | as x | n−1 | as x |
| `Std(x, n)` | TS | non-monotone | ≥ 0 | n−1 | as x |
| `Max/Min(x, n)` | TS | ↑ | as x | n−1 | as x |
| `CSRank(x)` (cross-sectional; DSL extension) | XS | ↑ | (0, 1] | 0 | dimensionless |
| `TsRank(x, n)` (= Qlib `Rank(x, n)`) | TS | ↑ (in current value) | (0, 1] | n−1 | dimensionless |
| `Corr(x, y, n)` | TS | non-monotone | [−1, 1] | n−1 | dimensionless |
| `x / y` | point | ↑ in x if y > 0; ↓ in y if x > 0 | — | 0 | unit(x)/unit(y) |
| `Log(x)` | point | ↑ | ℝ | 0 | requires dimensionless or log-units |
| `Sign(x)`, `Abs(x)` | point | step / non-monotone | — | 0 | — |

The unit rule implements the **dimensional-consistency check** (Alpha Jungle criticizes formulas that add a volume term to a log-price term); dimensionally inconsistent formulas are allowed in P3 (they occur in GP output) but flagged, because rationales for them are a distinct stress case.

### 6.3 Look-ahead impossibility — four layers
1. **Grammar:** negative lags/shifts and centered windows do not exist in the DSL.
2. **AST validator:** rejects any node whose time index exceeds t.
3. **Dynamic truncation test** (as in AlphaQT-Bench): compute the signal on the full panel and on ≥ 5 truncated prefixes; require exact equality on overlapping dates. Any discrepancy = causality violation.
4. **Label hygiene:** forward returns are computed in a separate module that the expression engine cannot import.

### 6.4 Verify the executor: two independent implementations
- Executor E1 = Qlib's expression engine for time-series operators, plus a thin cross-sectional layer for `CSRank`/`CSZScore`; Executor E2 = an independent vectorized NumPy/Polars implementation of everything, written from the operator spec.
- For every operator and for 500 random composite formulas: max |E1 − E2| ≤ 1e-8 after identical NaN handling, or cross-sectional rank correlation ≥ 0.9999 where float order matters.
- Property-based tests (e.g., the Hypothesis library) for operator laws: `CSRank`/`TsRank` invariance to monotone transforms, `Mean` linearity, `Ref` composition, window/NaN edge cases.
- Fixture panel (synthetic GBM + a real 50-stock slice) committed to the repo with expected outputs.

### 6.5 Canonical form, equivalence, deduplication
- **Symbolic normalization:** constant folding, commutative ordering, double-negation removal, `Ref` composition; optional equality saturation (e-graphs, `egg`) for algebraic identities.
- **Numerical equivalence:** two formulas are equivalent if exact-equal, or cross-sectional rank correlation ≥ 0.999 on ≥ 99% of dates (AlphaQT-Bench uses corr ≥ 0.999 as its functional-equivalence threshold; AlphaLogics uses a looser > 90% ranking/trend agreement for reconstruction).
- Used for: deduplication within pools, confirming SP variants are equivalent, confirming SA variants are *not*, identity-claim (C6) verification, novelty checks.

### 6.6 Complexity descriptors (for matching and as covariates)
Node count, depth, number of distinct fields, number of window parameters, maximum effective lookback, number of cross-sectional operators, number of non-monotone operators, dimensional-consistency flag. Fidelity almost certainly falls with complexity, so pools are matched on these descriptors (§7) and they enter the analysis model as covariates (§12).

---

## 7. Formula pools and generation protocols

### 7.1 Arm A — authorship protocols

**Common validity filter (all protocols):** parses → passes AST validator and truncation test → non-degenerate (cross-sectional std > 0 on ≥ 95% of dates; coverage ≥ 80% of member-days after warm-up) → not equivalent (§6.5) to another formula in the same pool. Do **not** filter on performance except in the explicitly labelled "mined" stratum.

| Protocol | Procedure | Strata / notes |
|---|---|---|
| **P1 Hypothesis-first** | Prompt with field list, operator list and one of 20 fixed research directions (e.g., "liquidity shocks", "intraday buying pressure", "volatility-scaled trend"); model writes a structured hypothesis (observation → mechanism → specification, as in AlphaAgent's idea agent), then a formula. Rationale = the hypothesis *plus* a post-hoc explanation requested after the formula is fixed | *P1-raw*: first valid formula per seed. *P1-mined*: ≤ 5 refinement rounds guided by validation-window IC (mirrors practice); log every trial for DSR/PBO |
| **P2 Formula-first (LLM)** | Same operator/field spec, no hypothesis; rationale requested in a fresh context by the same model; plus one cross-narration by a different family | Tests post-hoc rationalization of own formulas vs others' |
| **P3a GP** | Genetic programming on the DSL with training-window RankIC fitness and a parsimony penalty | Non-LLM authorship |
| **P3b Random grammar** | Sample trees from the DSL grammar | Matched to the P1/P2 complexity distribution by stratified sampling on node-count × depth bins |
| **P4 In-the-wild (optional)** | Run published frameworks with default settings and audit what they emit: AlphaAgent (GitHub `RndmVariableQ/AlphaAgent`) hypothesis/description/expression triples; RD-Agent factor scenario outputs; Alpha-R1 semantic profiles and AlphaLogics' market-logic library if released | External validity: real pipelines, real rationales |

### 7.2 Arm B — counter-recall pools

**Base set:** 60 public formulas expressible in OHLCV(+VWAP): 20 from Alpha101, 20 from GTJA-191, 20 from Alpha158, stratified by complexity tercile. Every variant below is generated by code, validated by the verifier, and stores the *targeted property* it is meant to change (or preserve).

| Pool | Construction | Validation | What faithful narration looks like |
|---|---|---|---|
| **K** Known-verbatim | Original notation, exactly as published | — | Correct description; recall is allowed to help |
| **K-named** | K + library ID shown ("WorldQuant Alpha#12") | — | Same as K; measures how much the *name* adds recall |
| **SP** Semantics-preserving | (i) notation translation (Alpha101 ↔ Qlib ↔ math), (ii) rename/inline intermediates, (iii) commutative reorder / algebraic identity, (iv) field anonymization with legend (`close → x3`) | Must be equivalent (§6.5) | Rationale claims should be **invariant** vs K (beyond sample-to-sample noise) |
| **SA-sign** | Negate the expression or the component that carries the directional claim | Verifier confirms direction property flips | Direction claims must flip |
| **SA-window** | Change a lookback (e.g., 5 → 60 days) | Verifier confirms lookback/horizon (and usually turnover) change | Horizon/turnover claims must update |
| **SA-field** | Swap a type-compatible field (close ↔ open; volume ↔ amount; high ↔ low) | Verifier confirms dependence/resemblance change | Field-dependence and resemblance claims must update |
| **NL** Misleading label | Unchanged K formula with a name/comment asserting a property the verifier REFUTES (e.g., "20-day momentum" on a reversal formula) | Label's property must be REFUTED | Rationale should follow the formula, not the label (CodeCrash-style test) |
| **N** Novel | Random-grammar + GP formulas generated **after the latest narrator cutoff**, complexity-matched to the base set | See novelty checks below | Same as any formula — the uncontaminated reference |

**Novelty checks for N (replaces "provably absent").**
1. Canonical form differs from every formula in the public libraries and in your own pools.
2. Maximum |time-average cross-sectional Spearman ρ| with every reference-library signal < 0.7 (pre-register).
3. Normalized-string search (whitespace/variable-name-insensitive) returns no hits in GitHub code search and in an n-gram index of open pretraining corpora (e.g., infini-gram over Dolma/RedPajama/The Pile).
4. Generation timestamp and random seed logged; formulas kept private until the study ends; narrated only by **local** models or endpoints whose terms exclude training on inputs.
5. On public release, embed a canary GUID in the files (BIG-bench practice) so future contamination can be detected.
Language in the paper: "absent from searchable public sources at generation time", never "provably absent from training data".

### 7.3 Pool sizes and run counts (main study, per narrator model)

| Arm | Formulas | Conditions per formula | k | Narrations |
|---|---|---|---|---|
| B: K, SP, N | 60 + 60 + 60 | A0 | 3 | 540 |
| B: SA (3 types) | 180 | A0 | 3 | 540 |
| B: K-named, NL | 30 + 30 | A0 | 3 | 180 |
| B: A2 subset (K + SA-sign) | 30 + 30 | A2 | 3 | 180 |
| A: P1-raw | 60 | A0, A1, A2 | 3 | 540 |
| A: P1-mined | 60 | A0 | 3 | 180 |
| A: P2 | 60 | A0, A1, A2 | 3 | 540 |
| A: P2 cross-narration (other family) | 60 | A0 | 3 | 180 |
| A: P3 (30 GP + 30 random, shared) | 60 | A0, A1, A2 | 3 | 540 |
| A: minimal-prompt subsample | 45 | A0 | 3 | 135 |
| **Total per model** (Arm B 1,440 + Arm A 2,115) | | | | **3,555** |

With 6 narrator models that is ≈ 21,300 narrations, plus authoring calls (≈ 60 P1 + ≤ 300 refinement + 60 P2 per model), one parser call per narration, and judge calls on a subsample (§13). Sizes follow from the power analysis in §12.4 (≈ 51 formulas per contrast arm under stated assumptions → 60 planned) and must be **re-derived from pilot estimates** before the main run.

---

## 8. Rationale elicitation

### 8.1 Prompt variants (verbatim templates in Appendix A)
- **Guided (primary):** asks the model to explain what the factor measures, which inputs drive it, the direction of its relation to future returns, its horizon, which known anomalies it resembles, whether it is novel, and when it should work. This mirrors the content of rationales produced by current systems (AlphaAgent descriptions, Alpha Jungle "alpha portraits", Alpha-R1 semantic profiles).
- **Minimal (robustness):** "Explain the rationale of this alpha factor." Checks that the guided prompt is not manufacturing claim types.
- Length cap: ≤ 250 words, free text (no forced claim list — structured elicitation is an ablation, §14).

### 8.2 Access levels

| Level | Narrator sees | What it tests |
|---|---|---|
| **A0** | Formula + glossary (field definitions, operator semantics) | Reading the formula unaided |
| **A1** | A0 + diagnostics table computed by the verifier on the training window: mean RankIC and t-stat, decile spread, rank autocorrelation, turnover, top-5 correlations with the reference library, factor loadings | Faithful *use of supplied evidence* (grounding) |
| **A2** | A0 + sandboxed tool: `compute_signal(expr)`, `corr_with(reference)`, `describe(expr)`, `perturb(expr, field, delta)` on training-window data only; ≤ 10 calls; all calls logged | Self-directed verification; whether models check before they claim |

A1 deliberately hands the model ground truth for several behavioral claims; the interesting quantity is how often it still contradicts the table.

### 8.3 Sampling and decoding
k = 3 samples at T = 0.7 (primary) plus one T = 0 reference; reasoning models at their default reasoning setting (recorded); identical max-token limits across models; seeds recorded where supported.

### 8.4 Randomization and counterbalancing
Randomize glossary order, field order and (in Arm B) the order in which a model sees K/SP/SA/NL variants of the same base formula (never in the same context). Latin-square assignment of prompt variants to formulas. No condition labels, pool names or the word "perturbed" ever appear in prompts.

### 8.5 Isolation and capture
Fresh context per narration; no memory or retrieval; refusals, empty and off-topic outputs logged and counted (not silently re-sampled more than once).

---

## 9. Claim extraction

### 9.1 Claim taxonomy and predicate templates (codebook excerpt; full codebook = Appendix C)

| Type | Subtype | Example sentence | Normalized predicate | Verified by |
|---|---|---|---|---|
| **C1 Mechanistic** | 1 Field dependence | "driven by trading volume" | `DEPENDS_ON(volume)` | AST + field-ablation magnitude |
| | 2 Direction w.r.t. an input | "stocks that rose recently score lower" | `SIGN(ret[5d], −)` | Monotonicity analysis → nudge test |
| | 3 Shape/monotonicity | "increasing in abnormal volume" | `MONO(abn_vol, ↑)` | Nudge test |
| | 4 Lookback/horizon | "uses a 20-day window", "short-term" | `LOOKBACK(20)`, `HORIZON(short)` | AST lookback; codebook bins |
| | 5 Cross-sectional vs time-series | "ranks stocks against each other" | `XSEC(true)` | AST |
| | 6 Invariance/units | "scale-free", "unit-less" | `INVARIANT(scale, price)` | Metamorphic test + unit rule |
| | 7 Range | "bounded between −1 and 1" | `RANGE(−1, 1)` | Static range analysis + empirical |
| | 8 Structure | "ratio of short to long moving average" | `STRUCT(ratio(MA_s, MA_l))` | AST pattern match |
| **C2 Behavioral** | 1 Resemblance | "captures short-term reversal" | `RESEMBLES(STREV_5d, +)` | Same-panel correlation (§10.3) |
| | 2 Exposure / independence | "independent of momentum", "low-volatility tilt" | `INDEPENDENT(MOM_12_1)`, `EXPOSED(VOL_20d, −)` | Equivalence test / loading CI |
| | 3 Turnover/persistence | "slow-moving, low turnover" | `TURNOVER(low)` | Rank autocorrelation, turnover |
| | 4 Conditional/regime | "works best in volatile markets" | `REGIME(high_vol, stronger)` | Interaction test |
| | 5 Predictive direction | "high values predict higher returns" | `PRED_SIGN(+, h)` | Sign of training-window RankIC |
| **C3 Originality** | 1 Novelty | "not captured by existing factors" | `NOVEL(library)` | Correlation ceiling + spanning test |
| | 2 Comparative | "improves on standard momentum" | `BETTER_THAN(MOM_12_1, IC)` | Paired OOS comparison |
| **C4 Performance** | magnitude/robustness | "robust, high IC", "stable across years" | `PERF(IC, high)` | OOS stats, HLZ t > 3, DSR |
| **C5 Theory/mechanism** | economic story | "because investors overreact to news" | `THEORY(overreaction)` | **UNVERIFIABLE** by execution (counted, never scored true/false) |
| **C6 Identity/provenance** | recall trace | "this is WorldQuant Alpha#12", "an RSI variant" | `IDENTITY(alpha101_012)`, `VARIANT_OF(RSI)` | Canonical form / numerical equivalence (§6.5) |

Every claim also carries slots: **scope/conditions** ("in small caps"), **hedge level** (absolute / typical / possible), **polarity**, **horizon**, **text span** (character offsets in the rationale).

### 9.2 Extraction approaches

| Option | Description | Pros | Cons |
|---|---|---|---|
| E1 Rule-based | Lexicons for fields/operators/anomaly names, regex for numbers and windows | Deterministic, high precision for C1.1, C1.4, C6 | Low recall for paraphrased C2/C3 claims |
| E2 Frozen LLM parser | Different family from all narrators; T = 0; JSON schema; Claimify-style instruction to extract only unambiguous claims and mark the rest AMBIGUOUS | High recall, handles paraphrase | Its own errors; must be validated |
| E3 Ensemble | Two LLM parsers (two families) + E1 slot filling; disagreements to a human | Most robust | Most expensive |
| **✅ First run** | E2 + E1 for slot filling; human adjudication on the validation sample; PPI correction (§9.4) | | |

### 9.3 Parser validation (before scaling up)
- **Sample:** ≥ 200 rationales, stratified by model × pool/protocol × access; refreshed with 50 more after the main run to check drift.
- **Annotators:** two people with finance + basic programming background, trained on the codebook with 20 practice items; a third adjudicates.
- **Gold:** annotators extract claims and assign predicates; verdicts for gold claims are produced by the verifier (annotators do not judge truth — that is the verifier's job).
- **Matching rule:** an extracted claim matches a gold claim if normalized predicates are equal and spans overlap.
- **Report:** extraction precision/recall/F1 by claim type; type accuracy; slot accuracy; inter-annotator agreement (Cohen's κ for two raters; Krippendorff's α with α ≥ 0.80 "reliable", ≥ 0.667 "tentative"; Gwet's AC1 where category prevalence is skewed).
- **Go/no-go:** proceed if F1 ≥ 0.80 for C1/C6 and ≥ 0.70 for C2/C3, and agreement ≥ the tentative threshold; otherwise revise the parser prompt/codebook and re-validate on fresh items.

### 9.4 Bias-corrected aggregate metrics with prediction-powered inference (PPI)
The parser is imperfect; PPI turns "imperfect but validated" into unbiased estimates.
For a rationale-level metric (e.g., claim precision of rationale r):
`θ̂_PPI = mean_{all r} f(auto_r) + mean_{r in gold set} [ f(gold_r) − f(auto_r) ]`,
where `f(auto_r)` uses parser-extracted claims and `f(gold_r)` uses human-extracted claims, both verified by the same verifier. Confidence intervals follow Angelopoulos et al. (2023) / PPI++; Boyeau et al. (2025) show this can substantially raise the effective human-labelled sample size. Apply per condition; report both naive and PPI-corrected values.

### 9.5 Normalization rules
- Vague terms are mapped through codebook bins (e.g., "short-term" = ≤ 21 trading days; "medium" = 22–126; "long" > 126). If a term has no bin → AMBIGUOUS.
- Claims about unobservable mechanisms → C5 (UNVERIFIABLE).
- Claims about things not in the formula's universe (e.g., "uses analyst revisions" for an OHLCV formula) are C1.1 claims and are **REFUTED** (the formula provably does not use that input).

---

## 10. Ground-truth verification engine

### 10.1 Verdicts

| Verdict | Meaning |
|---|---|
| **SUPPORTED** | Decision statistic lies entirely in the pre-registered "true" region (exact for static checks) |
| **REFUTED** | Lies entirely in the "false" region |
| **UNRESOLVED** | Interval straddles a boundary, or the property holds only conditionally when the claim was unconditional |
| **UNVERIFIABLE** | C5 theory claims; claims needing data outside the panel |
| **AMBIGUOUS** | Cannot be mapped to a predicate without guessing (§9.5) |

Precision is computed on SUPPORTED + REFUTED ("decidable") claims; the other shares are always reported alongside.

### 10.2 Mechanistic claims (C1, C6)

**Static analysis (exact).**
- Dependency set = leaf fields of the AST (after constant folding). `DEPENDS_ON(x)` is REFUTED if x is absent.
- Effective lookback = max over root-to-leaf paths of the sum, over operators on the path, of their lag d or (window n − 1) — compare to `LOOKBACK(n)` exactly for explicit numbers and via codebook bins for words.
- `XSEC` = presence of a cross-sectional operator (`CSRank`, `CSZScore`) on the path to the root.
- **Monotonicity abstract interpretation:** propagate a sign lattice {↑, ↓, 0, ±} from each leaf to the root using the operator table (§6.2): e.g., `CSRank`, `TsRank` (in the current value), `Mean`, `Sum`, `Ref` preserve; negation flips; division is ↑ in a numerator only if the denominator's sign is known positive (prices, volumes); `Std`, `Corr`, products of terms with unknown sign give ±. A definite ↑/↓ decides `SIGN`/`MONO` claims exactly; ± sends the claim to the nudge test.
- Units: dimensional consistency from the unit rule; `INVARIANT(scale, …)` claims decided by the metamorphic test below when not static.

**Nudge tests (near-exact, for ± cases and for direction claims about derived quantities like "recent returns").**
```
for c in sample(N=500 stock-date contexts from the training window, stratified by year and size):
    x0 = panel window needed by the formula at (stock_c, date_c)
    x1 = apply_perturbation(x0, claimed_input, delta)    # e.g., raise the last-5-day return path by +0.5 σ_i
                                                          #  by rescaling closes; keep other fields fixed
    d_c = signal(x1)[stock_c, date_c] − signal(x0)[stock_c, date_c]
share_pos = count(d > 0) / count(d ≠ 0);  report share_zero = count(d == 0)/N
SUPPORTED if Clopper-Pearson lower bound(share in claimed direction) ≥ 0.90
REFUTED   if Clopper-Pearson upper bound(share in claimed direction) ≤ 0.10
else UNRESOLVED (property is conditional)
```
For cross-sectional operators the perturbation is applied to one stock while the cross-section is held fixed, so rank changes are attributable. Deltas, sample sizes and bounds are pre-registered (§10.6).

**Metamorphic relations.** Multiply all prices by c > 0; multiply volume by c; add a constant to log prices; permute stock order (cross-sectional equivariance); shift dates (time equivariance); replace a field by its cross-sectional median (field ablation). Each relation either holds exactly or yields a violation magnitude used for `INVARIANT` and `DEPENDS_ON` strength.

**Optional formal check.** Encode small fragments in an SMT solver (Z3) with domain constraints (prices > 0, volume ≥ 0) to prove or refute sign claims that abstract interpretation leaves at ±.

**Driver attribution (for coverage metrics).** Variance-based global sensitivity: for each input field, replace it with a block-bootstrapped series from another stock-period (preserving its marginal behavior) and compute the total-effect index S_T of the signal's cross-sectional ranks. "True drivers" = fields with S_T ≥ 0.10 (pre-register). Used only for coverage/omission metrics, not for individual claim verdicts.

**Identity claims (C6).** `IDENTITY(lib_id)` SUPPORTED if the formula is canonical-equal or numerically equivalent (§6.5) to the named library formula; REFUTED otherwise. `VARIANT_OF(RSI)` SUPPORTED if the canonical form contains the codebook's RSI template or rank-correlates ≥ 0.9 with the reference RSI signal; REFUTED if ≤ 0.3; else UNRESOLVED.

### 10.3 Behavioral claims (C2, C3)

**Signal-level similarity (primary).** For reference signal g, compute the daily cross-sectional Spearman correlation ρ_t(f, g); summarize by the time average ρ̄ with a stationary block bootstrap CI (Politis–Romano; automatic block length) and, as a cross-check, a Newey–West t.

| Claim | SUPPORTED | REFUTED | Notes |
|---|---|---|---|
| `RESEMBLES(g, +)` | 95% CI lower bound of ρ̄ ≥ 0.30 | 95% CI upper bound ≤ 0.10 (also REFUTED-with-sign-error if CI upper ≤ −0.30) | Thresholds pre-registered; sensitivity ±50% |
| `INDEPENDENT(g)` | **TOST:** 90% CI of ρ̄ inside (−0.10, 0.10) | 95% CI entirely outside [−0.10, 0.10] | Non-rejection of ρ̄ = 0 is never treated as support |
| `EXPOSED(F, sign)` (return level) | Loading β of the formula's decile long–short return on factor F has HAC t ≥ 3 in the claimed sign and an absolute size above a pre-registered floor | t ≥ 3 in the opposite sign, or TOST shows the absolute loading below the floor | US: FF5 + MOM + ST-reversal (daily); CN: self-built daily factors, CH-3 monthly |
| `TURNOVER(low/high)` | Mean lag-1 rank autocorrelation CI above the 75th percentile of the reference library (low) / below the 25th (high); absolute definitions reported too | Opposite tail | Relative definition is more robust across markets |
| `REGIME(r, stronger)` | Interaction of daily RankIC with regime indicator (e.g., top tercile of market 20d realized volatility) has t ≥ 2 in the claimed direction in both train and test windows | Opposite sign t ≥ 2 | Most such claims are expected to be UNRESOLVED; that share is itself informative |
| `PRED_SIGN(s, h)` | Training-window mean RankIC has Newey–West t ≥ 2 in the claimed sign | t ≥ 2 in the opposite sign | Direction-of-effect claim, not a discovery claim (HLZ bar applies to C4) |

**Originality (C3.1 `NOVEL`).** SUPPORTED if (i) the 95% CI upper bound of max |ρ̄| against all reference signals < 0.50 **and** (ii) the signal's long–short return keeps a t ≥ 3 alpha after controls chosen by double-selection LASSO from the reference factor set (Feng, Giglio & Xiu 2020). REFUTED if any reference signal has ρ̄ CI lower bound ≥ 0.80, or an IDENTITY match exists. Otherwise UNRESOLVED.

**Alternative for behavioral claims (Option B).** Sequential falsification with e-values (POPPER, ICML 2025) gives anytime-valid error control when many behavioral claims are tested adaptively; worth it if A2 tool-using narrators generate many conditional claims.

### 10.4 Performance claims (C4)
- Evaluated **out of sample only**: the 2021–2024 test window and, separately, the post-cutoff window H_post (§5.2).
- "Significant / robust" claims require HLZ t > 3.0 on the OOS IC or long–short return.
- Sharpe-type claims: report Deflated Sharpe Ratio using the **actual number of trials** behind the formula (log every candidate in P1-mined and P3a runs).
- Families of candidates (P1-mined, P3a GP): Probability of Backtest Overfitting via CSCV (e.g., S = 16 partitions); Romano–Wolf stepdown or White's Reality Check / Hansen's SPA when a rationale claims "best of" a set.
- Transaction costs stated explicitly; for comparability use AlphaAgent's convention (CN: 5 bp buy, 15 bp sell; US: 5 bp sell) and report Alpha Jungle's 15 bp per trade as sensitivity. China limit-locked days untradable.

### 10.5 Theory claims (C5)
Never scored. Report their share per condition, and their co-occurrence with REFUTED mechanistic claims ("a confident story attached to a misread formula") — the operational form of the *narrative bias* identified by Kong et al. (2026).

### 10.6 Thresholds to pre-register (defaults and sensitivity ranges)

| Parameter | Default | Sensitivity |
|---|---|---|
| Nudge contexts N; delta | 500; +0.5 σ of the stock's own input | N = 200/1,000; delta 0.25/1.0 σ |
| Nudge SUPPORTED / REFUTED bounds | Clopper–Pearson lower ≥ 0.90 / upper ≤ 0.10 | 0.85 / 0.95 |
| Driver threshold S_T | 0.10 | 0.05 / 0.20 |
| Resemblance ρ̄ floor; refute ceiling | 0.30; 0.10 | 0.20–0.40 |
| Independence margin (TOST) | ±0.10 | ±0.05 / ±0.15 |
| Novelty ceiling / duplicate floor | 0.50 / 0.80 | 0.40–0.60 / 0.70–0.90 |
| Numerical equivalence | ρ ≥ 0.999 on ≥ 99% of dates | 0.995 |
| Performance discovery bar | HLZ t > 3.0 | t > 2.0 reported, not used for verdicts |
| Horizon bins | ≤ 5 / ≤ 21 / 22–126 / > 126 trading days | — |

### 10.7 Validating the verifier
1. **Planted-claims calibration set:** 300 synthetic (formula, claim, truth) triples whose truth is known by construction (e.g., build a formula that is ↑ in 5-day return and plant both the true and the false direction claim; build exact duplicates of library formulas for identity claims; build signals with controlled correlation to a reference via mixing for resemblance claims). Acceptance: ≥ 99% accuracy on decidable static items, ≥ 95% on statistical items with zero sign errors; publish the confusion matrix.
2. **Dual executors** agree (§6.4).
3. **Human spot-check:** 100 random verdicts reviewed by an annotator with the evidence bundle (static report, nudge histogram, correlation plots); disagreements investigated, not overruled.
4. **Unit and property tests** for every verifier routine; CI runs them on each commit.

---

## 11. Metrics (all with cluster-bootstrap or PPI confidence intervals)

Let C be all claims of a rationale and D ⊆ C the decidable ones (SUPPORTED ∪ REFUTED).

| Metric | Definition | Primary RQ |
|---|---|---|
| **Claim precision (CP)** | \|SUPPORTED\| / \|D\|; reported micro (pooled claims) and macro (mean over rationales) | All |
| Error rate by type | \|REFUTED of type t\| / \|D of type t\| | All |
| Decidability / unresolved / unverifiable / ambiguous shares | shares of \|C\| | All |
| Claim density | \|C\| and \|D\| per rationale (a model can buy precision by saying nothing) | All |
| **Driver coverage** | \|mentioned fields ∩ true drivers\| / \|true drivers\| (§10.2) | RQ1, RQ2 |
| **Dominant-driver omission** | 1[field with the largest S_T is not mentioned] | RQ1, RQ2 |
| **Exposure coverage / dominant-exposure omission** | Same, for reference signals with ρ̄ CI lower ≥ 0.30 | RQ1, RQ2 |
| **Counterfactual fidelity (SA)** | Among SA variants whose rationale mentions the targeted property: share where the property claim is SUPPORTED under the *perturbed* formula | RQ3 |
| **Recall-anchoring rate (SA)** | Share of SA rationales asserting the base formula's targeted property (SUPPORTED on base, REFUTED on perturbed) | RQ3 |
| **SP invariance** | Jaccard overlap of normalized predicate sets K vs SP, divided by the within-condition sample-to-sample Jaccard (≈ 1 means "as stable as resampling") | RQ3 |
| **Label-following rate (NL)** | Share of NL rationales asserting the misleading label's (REFUTED) property | RQ3 |
| **Identity-claim rate and accuracy** | Share of rationales with C6 claims; their precision by pool (identity claims in SA/N pools are recall leakage) | RQ3, RQ5 |
| **Counter-recall gaps** | Δ_SA = CP(K) − CP(SA); Δ_N = CP(K) − CP(N) (complexity-matched); Δ_SP = CP(K) − CP(SP) (expected ≈ 0) | RQ3, RQ5 |
| Evidence-use rate (A1) | Share of diagnostics-table facts restated correctly; contradiction rate against the table | RQ2 |
| Tool-check rate (A2) | Share of decidable claims preceded by a relevant tool call; precision of checked vs unchecked claims | RQ2 |
| Sample stability | Fleiss' κ over k samples on predicate presence; verdict variance | All |
| **Judge validity** | AUROC and Spearman ρ of judge score vs verified rationale-level CP; **false-accept rate** = share of rationales with ≥ 1 REFUTED claim scored ≥ the judge's acceptance threshold (e.g., c2 ≥ 0.8) | RQ4 |
| Fidelity–performance link | Correlation between formula-level CP and OOS RankIC (exploratory, with HLZ caveats) | RQ6 |

---

## 12. Statistical analysis plan

### 12.1 Estimands
Differences in claim precision (percentage points) between conditions, overall and by claim type; rates (anchoring, label-following, false-accept) with CIs; AUROC for judge validity.

### 12.2 Primary model (claim level)
Logistic GLMM on decidable claims:
`logit P(SUPPORTED) = condition + claim_type + condition × claim_type + complexity + hedge_level + market + (1 | formula) + (1 | rationale) + (1 | base_formula)` (Arm B) with model either as a fixed effect (few models) or a random effect (≥ 8 models).
Software: R `lme4`/`glmmTMB` + `marginaleffects`, or Python `bambi`/`pymc` (Bayesian) — report marginal effects, not log-odds.

### 12.3 Secondary and small-cell analyses
- Rationale-level CP with a **two-stage cluster bootstrap** (resample formulas, then rationales within formulas); clustered standard errors as in Miller (2024).
- Cells with fewer than a few hundred items: Bayesian beta-binomial or exact (Clopper–Pearson / Wilson) intervals rather than CLT intervals (Bowyer et al. 2025).
- Arm B is **paired** (each SA/SP/NL variant has a base formula): use paired differences or base-formula random effects — this is the most powerful contrast in the study.

### 12.4 Power analysis (worked example; replace assumptions with pilot estimates)
Assumptions: detect CP 0.70 vs 0.55 (15-point gap), power 0.80, two-sided.

| Step | Value | How |
|---|---|---|
| Independent claims per arm, α = 0.05 | ≈ 162 | Two-proportion formula |
| … with Bonferroni-equivalent α = 0.05/12 (z ≈ 2.87) | ≈ 285 | Same formula |
| Design effect for claims within rationales (m ≈ 6 claims/rationale, ICC ≈ 0.2) | DE = 1 + (m − 1)·ICC = 2.0 → ≈ 569 claims ≈ 95 rationales | Kish design effect |
| Design effect for k = 3 samples within formula (ICC ≈ 0.3) | DE = 1.6 → ≈ 152 rationales ≈ **51 formulas per arm** | Same |
| Planned | **60 formulas per arm** (§7.3) | Margin for exclusions |

Holm in a 6-test confirmatory family (§12.5) is less conservative than the Bonferroni line above, so this is a conservative plan. **Final N:** simulate the §12.2 GLMM with pilot-estimated m, ICCs and decidability rates (e.g., R `simr` or a custom simulator) and pick N giving ≥ 0.80 power for every confirmatory test.

### 12.5 Multiplicity: confirmatory vs exploratory
**Confirmatory family (Holm, FWER = 0.05), pre-registered:**
1. CF1 (RQ1): CP(P3) < CP(P1), complexity-adjusted.
2. CF2 (RQ2): CP(A2) > CP(A0) for C2–C4 claims.
3. CF3 (RQ3): CP on *targeted-property* claims lower in SA than K (paired).
4. CF4 (RQ3): Label-following rate in NL > its rate in K for the same property (paired).
5. CF5 (RQ4): Holistic judge (B1) AUROC for detecting rationales with ≥ 1 REFUTED claim < 0.90 (one-sided).
6. CF6 (RQ5): CP(N) ≠ CP(K), complexity-matched (two-sided).
**Exploratory:** every other contrast (per model, per claim subtype, per access level, per market…), Benjamini–Hochberg at FDR 0.10 (Benjamini–Yekutieli where dependence is strong), always labelled exploratory.

### 12.6 Performance-claim statistics
As §10.4 (HLZ t > 3, DSR with recorded trial counts, PBO via CSCV, stepdown/RC/SPA for "best-of" claims, pre- vs post-cutoff windows). Fidelity and performance are reported on separate axes.

### 12.7 Robustness battery
Threshold sensitivity (§10.6 ranges); second parser; minimal vs guided prompt; T = 0 vs T = 0.7; complexity-matched subsets; per market; train- vs test-window verdicts; absolute-only claims (dropping hedged ones); excluding A1 items whose answers were in the table; leave-one-model-out.

### 12.8 Reporting standards
Every rate with a CI and raw counts; effect sizes before p-values; forest plots by model; a table of all pre-registered vs exploratory results; deviations from the pre-registration listed in an appendix.

---

## 13. Baselines and comparisons (RQ4)

| ID | Baseline | Input | Output | What a difference from the verifier means |
|---|---|---|---|---|
| B1 | **Holistic LLM consistency judge** (AlphaAgent's c2-style description–expression score; c1 when a hypothesis exists) | Rationale + formula | Score in [0, 1] | The status-quo check misses (or flags) false rationales |
| B2 | Claim-level LLM judge, no execution | Each parsed claim + formula | TRUE / FALSE / CAN'T TELL | Is decomposition alone enough? |
| B3 | Claim-level LLM judge **with A2 tools** | Same + sandbox | Same | Is execution access (not our verifier) enough? If B3 ≈ verifier, cheap audits are possible — a useful positive result |
| B4 | **Reconstruction test** (AlphaLogics) | Rationale only → rebuilt formula | Equivalence (AlphaLogics' > 90% agreement; strict ρ ≥ 0.999) | 2 × 2 table: reconstructable × contains a REFUTED claim; reconstructable-but-false is the key cell |
| B5 | NLI/entailment model | Template-generated canonical description of the AST (premise) + claim (hypothesis) | Entail / contradict / neutral | Whether surface text matching suffices for C1 claims |
| B6 | Human experts (2) | 100 rationales, claim level, without then with the evidence bundle | Verdicts + time per item | Cost and accuracy of manual review |

Judges come from families not used as narrators in the main comparison; a same-family judge run measures self-preference. Report agreement per claim type, cost per audited claim (tokens, minutes), and qualitative case studies of disagreements.

---

## 14. Ablations (secondary, exploratory)
- Structured elicitation ("list your claims as bullet facts") vs free text — does asking for claims change their truth?
- Notation: Qlib DSL vs Alpha101 notation vs plain math (inside SP).
- Field anonymization with legend vs real field names.
- Reasoning effort settings (where configurable).
- Self-narration vs cross-narration of P2 formulas.
- Dimensionally inconsistent vs consistent formulas (P3).
- Length cap 150 vs 250 words.
- ChronoGPT-Instruct (vintage, no look-ahead) as an extra narrator on Arm B, **only if** it passes a pilot competence check (≥ 50% C1 precision on K).

---

## 15. Threats to validity and mitigations

| Threat | Type | Mitigation |
|---|---|---|
| Parser misses/misreads claims | Internal | Validation (§9.3), second parser, PPI correction (§9.4) |
| Verifier bugs | Internal | Dual executors, planted-claims calibration, unit/property tests, human spot-check (§10.7) |
| Thresholds chosen after seeing data | Internal | OSF pre-registration with timestamps; sensitivity ranges (§10.6) |
| Ambiguous natural language | Construct | Codebook bins, AMBIGUOUS bucket, Claimify-style extraction rule |
| "Fidelity" conflated with "profitability" | Construct | Separate axes; C4 only out of sample (§10.4) |
| Complexity confound between pools/protocols | Internal | Complexity matching (§7) + covariates (§12.2) |
| Prompt induces claim types | Construct | Minimal-prompt replication (§8.1) |
| Model drift / silent updates | External, reproducibility | Version pinning, dated logs, cached responses, local weights hashes |
| Free-tier models removed mid-study | External | Local inference for core roster; provider-independent cache |
| Survivorship and corporate actions | Statistical conclusion (C4) | Point-in-time membership, coverage reporting, Shumway-style bounds, paid-data replication |
| Look-ahead via model knowledge | Internal (C4) | Post-cutoff window; Novel pool; vintage-model ablation |
| Novel pool contamination by our own API calls | Internal | Local narration of N, data-use terms check, delayed release with canary |
| Selection on performance (P1-mined) | Internal | Separate stratum; trial logging for DSR/PBO |
| Market specificity | External | Two markets + optional JPX/crypto replication |
| Operator-set specificity | External | Two notations; optional code-based extension (CogAlpha/AlphaQT style) |

---

## 16. Reproducibility and ethics package

**Release (at publication):**
- Typed operator library, both executors, verifier, perturbation generator, parser prompts, judge prompts, codebook (Appendix C), thresholds file (pre-registered).
- All formula pools **including N**, with a canary GUID; complexity descriptors; novelty-check logs.
- All narrator/parser/judge call logs (JSONL), with model strings, dates, parameters, fingerprints, local weight hashes.
- Human annotation guidelines, raw annotations, adjudication log, agreement statistics.
- Data **rebuild scripts** (not licensed raw data), file hashes, coverage tables, and a licensing statement mapping each result to its data tier.
- Planted-claims calibration set with expected verdicts.
- Environment: lockfile + Dockerfile; hardware and GPU-hours report.

**Checklists:** REFORMS (Kapoor et al., *Science Advances* 2024) for ML-based science; NeurIPS paper checklist or ACL Responsible NLP checklist depending on venue; the bias checklist of Kong et al. (2026) for finance-specific items (look-ahead, survivorship, narrative, objective, cost).

**Pre-registration:** OSF (or AsPredicted) entry containing RQs, confirmatory tests, thresholds, sample-size rule, exclusion rules, analysis code skeleton — registered after the pilot and before the main run.

**Ethics:** no human subjects beyond annotators (consent + fair pay); no live trading; explicit statement that nothing in the paper is investment advice.

**Suggested repository layout**
```
rationale-fidelity/
├── configs/            # models.yaml (pinned), thresholds.yaml (pre-registered), prompts/
├── data/               # MANIFEST.csv, rebuild scripts, coverage reports (no raw licensed data)
├── dsl/                # grammar, typed operator table, parsers, serializers, canonicalizer
├── executors/          # e1_qlib/, e2_numpy/, tests/ (property-based), fixtures/
├── pools/              # generators for P1–P4 and K/SP/SA/NL/N, novelty checks, descriptors
├── narrate/            # runners for A0/A1/A2, sandbox tool, call logger, cache
├── parse/              # parser prompts, rule layer, schema, validation scripts
├── verify/             # static, nudge, metamorphic, behavioral, performance, identity
├── judges/             # B1–B6
├── analysis/           # GLMM, bootstrap, PPI, power simulation, figures
├── annotation/         # guidelines, samples, agreement scripts
└── paper/              # tables generated from analysis outputs only
```

---

## 17. Execution plan for the first (free) run

### 17.1 Phases and gates

| Phase | Weeks | Work | Gate to pass |
|---|---|---|---|
| 0 Setup | 1–2 | Environment (Kaggle/Colab/local); Qlib; CN data (chenditc bins + BaoStock cross-check); US data (yfinance + historical S&P 500 membership + partial delisted coverage); reference libraries; DSL + both executors; verifier unit tests; planted-claims set | Executors agree; calibration ≥ targets (§10.7); data QA checklist complete (§5.9) |
| 1 Pilot | 3 | 3 models (one per family) × ≈ 300 narrations each (Arm B: K, SP, SA-sign, N × 12 formulas; Arm A: P1, P3 × 12 formulas × A0/A2; k = 3); parser v1; 60 rationales double-annotated | Parser go/no-go (§9.3); estimates of m, ICCs, decidability, refusal rate, tokens/s |
| 2 Freeze | 4 | Final N by simulation; thresholds frozen; prompts frozen; OSF pre-registration | Registration timestamped |
| 3 Main run | 5–8 | All arms, full roster (§7.3), parser runs, verifier runs, judge baselines on a stratified 20% sample | ≥ 95% of planned cells complete; logs validated |
| 4 Validation & analysis | 9–10 | 200+ rationale annotation, PPI correction, GLMM, robustness battery | All confirmatory tests computed as registered |
| 5 Paid upgrade (optional) | later | CRSP/Norgate (US) or CSMAR (CN) for C4 and US behavioral checks; frontier closed models; longer post-cutoff window | Replication of confirmatory results |

### 17.2 Budget arithmetic (replace assumptions with pilot measurements)
- Narrations per model (main): 3,555 (§7.3). Assume ≈ 2,000 input and ≈ 450 output tokens on average (A2 multi-turn calls are larger): ≈ 7.1 M input and ≈ 1.6 M output tokens per model.
- Local generation time ≈ output tokens ÷ throughput: at an assumed 300 tokens/s (batched small models) ≈ 1.5 h of generation per model; at 30 tokens/s (single-stream ~30B 4-bit) ≈ 15 h. Reasoning models multiply output tokens — budget separately. Prefill time adds to this; measure in the pilot.
- Free API tiers: days ≈ calls ÷ daily cap (e.g., 3,555 calls at 1,000 requests/day ≈ 3.6 days per model, before parser calls).
- Human annotation: ≈ 260 rationales × ≈ 4 minutes × 2 annotators ≈ 35 person-hours, plus adjudication.

### 17.3 Concrete first-week commands (illustrative; check current docs)
```bash
# China A-shares: community Qlib bins (Qlib README points here while the official dataset is disabled)
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/cn_data   # from the Qlib repo

# Cross-check source and point-in-time constituents
pip install baostock pyqlib yfinance openassetpricing
```
```python
import baostock as bs
bs.login()
members = bs.query_zz500_stocks(date="2019-06-28").get_data()       # CSI500 members on that date
universe = bs.query_all_stock(day="2019-06-28").get_data()          # all securities trading that day
k = bs.query_history_k_data_plus("sh.600000", "date,open,high,low,close,volume,amount",
                                  start_date="2015-01-01", end_date="2024-12-31",
                                  frequency="d", adjustflag="1").get_data()   # check adjustflag semantics in docs
bs.logout()
```

---

## 18. Explicit non-claims (state in the paper)
- No claim about the validity of any economic theory (C5 claims are counted, never scored).
- Faithful ≠ profitable: fidelity and performance are separate axes.
- No new alpha-mining method is proposed.
- Not the first to check factor explanations: AlphaAgent scores description–expression consistency with an LLM, XALPHA and QuantaAlpha enforce alignment with LLM checks, AlphaLogics reconstructs formulas from explanations, AlphaQT-Bench verifies code against instructions by execution. The contribution is **claim-level, execution-verified truth of rationale text**, plus the **counter-recall** design.
- Results on free data are labelled with their data tier; survivorship-sensitive conclusions (C4) are flagged until replicated on paid data.

---

## 19. References

See the specification's reference list (§19.1–19.8): LLM alpha mining (AlphaAgent, Alpha Jungle, R&D-Agent-Quant, CogAlpha, AlphaLogics, FactorMiner, QuantaAlpha, FactorEngine, XALPHA, Alpha-R1, Chain-of-Alpha, FAMA, AlphaGen, Zhang et al. survey); benchmarks (AlphaBench, AlphaQT-Bench, AlphaEval, FINSABER, Kong et al.); memorization and look-ahead (Lopez-Lira et al., Profit Mirage, ChronoBERT/ChronoGPT, Glasserman & Lin, Sarkar & Vafa); faithfulness and claims (Turpin et al., Lanham et al., Chen et al., Matton et al., Atanasova et al., FActScore, SAFE, Claimify, CodeCrash, Wu et al., GSM-Symbolic, CRUXEval, POPPER, infini-gram, Oren et al., BIG-bench); statistics (Miller, Bowyer et al., Card et al., Angelopoulos et al., Boyeau et al., Krippendorff, Gwet, Cohen, Lakens, Politis & Romano, Politis & White, Newey & West, Benjamini–Hochberg, Benjamini–Yekutieli, Holm, Kish, Green & MacLeod); asset pricing (Harvey, Liu & Zhu; Harvey & Liu; Bailey & López de Prado; Bailey et al.; White; Hansen; Romano & Wolf; Chordia et al.; Feng, Giglio & Xiu; Jensen, Kelly & Pedersen; Hou, Xue & Zhang; Chen & Zimmermann; Fama & French; Liu, Stambaugh & Yuan; Shumway; Kakushadze; Guotai Junan; Yang et al. Qlib); software verification (Chen et al. metamorphic testing; egg; Z3; Saltelli et al.; REFORMS; Kapoor & Narayanan; Chen, Zaharia & Zou). Data and platform pages accessed 6 Oct 2026 are listed in §19.8 of the specification and in `data/README.md`.

## Appendices A–G

Implemented verbatim in the repository: Appendix A (narrator prompts) → `configs/prompts/narrator_*.txt`; Appendix B (parser prompt and schema) → `configs/prompts/parser_instruction.txt`, `parse/schema.py`; Appendix C (codebook) → `configs/codebook.yaml`; Appendix D (record schemas) → `pools/records.py`, `narrate/runner.py`, `verify/__main__.py`; Appendix E (pre-registration template) → `paper/preregistration.md`; Appendix F (planted-claims recipes) → `verify/calibration.py`; Appendix G (search log) → `paper/search_log.md`.
