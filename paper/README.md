# paper/

Tables are generated from analysis outputs only (`python -m paper tables --run-dir runs/main`), never
typed by hand. Every rate carries its CI and raw counts, effect sizes precede p-values, pre-registered
and exploratory results are separated, and each table states the data tier it was computed on.

* `make_tables.py` — Markdown tables from `runs/<run>/analysis/results.json`.
* `preregistration.md` — Appendix E template (register after the pilot, before the main run).
* `search_log.md` — Appendix G literature-search log (re-run before submission).

## Checklists to complete at submission

* REFORMS (Kapoor et al., *Science Advances* 2024) for ML-based science.
* NeurIPS paper checklist or ACL Responsible NLP checklist, depending on venue.
* Finance bias checklist of Kong et al. (2026): look-ahead, survivorship, narrative, objective, cost.

## Explicit non-claims (§18) — state in the paper

* No claim about the validity of any economic theory (C5 claims are counted, never scored).
* Faithful ≠ profitable: fidelity and performance are separate axes.
* No new alpha-mining method is proposed.
* Not the first to check factor explanations: AlphaAgent scores description–expression consistency with
  an LLM, XALPHA and QuantaAlpha enforce alignment with LLM checks, AlphaLogics reconstructs formulas
  from explanations, AlphaQT-Bench verifies code against instructions by execution. The contribution is
  claim-level, execution-verified truth of rationale text, plus the counter-recall design.
* Results on free data are labelled with their data tier; survivorship-sensitive conclusions (C4) are
  flagged until replicated on paid data.

## Ethics

No human subjects beyond annotators (informed consent, fair pay); no live trading; nothing in the paper
is investment advice.
