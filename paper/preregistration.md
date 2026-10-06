# Pre-registration (Appendix E) — fill after the pilot, register on OSF before the main run

**Study:** Execution-verified rationale fidelity for LLM alpha discovery (Experimental Pipeline v2).
**Registration date:** `YYYY-MM-DD` · **Registry:** OSF / AsPredicted · **Code commit:** `<git sha>`

## E.1 Hypotheses (confirmatory family, Holm at FWER 0.05) — `analysis/confirmatory.py`

| ID | RQ | Hypothesis | Direction | Test |
|---|---|---|---|---|
| CF1 | RQ1 | Claim precision is lower for P3 (non-LLM formulas narrated post hoc) than for P1 (hypothesis-first), complexity-adjusted | one-sided (less) | claim-level GLMM, condition effect, random intercepts formula / rationale |
| CF2 | RQ2 | CP(A2) > CP(A0) for C2–C4 claims | one-sided (greater) | same GLMM on Arm-A C2–C4 claims |
| CF3 | RQ3 | CP on targeted-property claims is lower in SA than in K (paired by base formula) | one-sided (less) | GLMM with base-formula random effect |
| CF4 | RQ3 | Label-following rate in NL > its rate in K for the same property (paired) | one-sided (greater) | paired bootstrap over base formulas (Wilcoxon reported) |
| CF5 | RQ4 | Holistic judge (B1) AUROC for detecting rationales with ≥ 1 REFUTED claim < 0.90 | one-sided (less) | bootstrap of AUROC |
| CF6 | RQ5 | CP(N) ≠ CP(K), complexity-matched | two-sided | GLMM |

Exploratory: every other contrast (per model, per claim subtype, per access level, per market…),
Benjamini–Hochberg at FDR 0.10 (Benjamini–Yekutieli where dependence is strong), always labelled exploratory.

## E.2 Outcomes

* Primary: claim precision on decidable claims (SUPPORTED / (SUPPORTED + REFUTED)), micro and macro.
* Secondary (§11): error rate by type; decidable / unresolved / unverifiable / ambiguous shares; claim
  density; driver coverage and dominant-driver omission; exposure coverage; counterfactual fidelity and
  recall-anchoring rate (SA); SP invariance; label-following rate (NL); identity-claim rate and accuracy;
  counter-recall gaps Δ_SA, Δ_N, Δ_SP; evidence use (A1); tool-check rate (A2); sample stability; judge
  validity (AUROC, Spearman, false-accept rate); fidelity–performance link (exploratory).

## E.3 Thresholds

`configs/thresholds.yaml`, SHA-256: `python -c "import configs; print(configs.thresholds_hash())"` → `<hash>`.
Sensitivity ranges in the same file are the robustness battery (§12.7).

## E.4 Sample sizes

Worked example `python -m analysis power` (≈ 51 formulas per arm → 60 planned). Final N from the GLMM
simulation with pilot-estimated m, ICCs (variance components) and decidability:
`analysis.power.required_formulas(sd_formula=…, sd_rationale=…, m=…, p1=…, p2=…, method="glmm")` → `<N>`.
No optional stopping.

## E.5 Exclusion rules

* Formulas failing the validity filter (§7.1) never enter a pool; base-set swaps are logged in `pools_report.json`.
* Refusals / empty / off-topic outputs are counted and re-sampled once; never replaced beyond one retry.
* Parser failures (unparsable JSON) are counted; affected rationales enter only the PPI gold-set analysis.
* The Novel pool is narrated only by local models or endpoints whose terms exclude training on inputs.

## E.6 Analysis code skeleton

`python -m analysis run --run-dir runs/main --n-boot 1000` (fixed seed 20261006 from `configs/study.yaml`).
R equivalent of the primary model: `python -m analysis r-script --run-dir runs/main`.

## E.7 Robustness checks and deviations

Planned robustness (§12.7): threshold sensitivity; second parser; minimal vs guided prompt; T = 0 vs
T = 0.7; complexity-matched subsets; per market; train- vs test-window verdicts; absolute-only claims;
excluding A1 items answered by the table; leave-one-model-out.

Any change to RQs, tests, thresholds, sample-size rule, exclusion rules or analysis code after
registration is a deviation and is listed with its reason in the paper's appendix.
