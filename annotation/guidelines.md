# Annotation guidelines — claim extraction (§9.3)

Annotators: two people with finance and basic programming background, trained on the codebook
(`configs/codebook.yaml`) with 20 practice items; a third person adjudicates.

**You extract claims; you never judge their truth.** Verdicts for gold claims are produced by the
verifier.

## Unit of annotation

An *atomic claim* is the smallest statement about the factor that could be true or false on its own:
"driven by volume and close" is two claims (`DEPENDS_ON(volume)`, `DEPENDS_ON(close)`).

For every claim record, in `sheet_<you>.csv`:

| column | content |
|---|---|
| `rationale_id`, `claim_no` | identify the claim (number claims 0, 1, 2 … within a rationale) |
| `span_start`, `span_end` | character offsets of the exact supporting text in the rationale |
| `text` | the copied span |
| `type` | C1 mechanistic, C2 behavioral, C3 originality, C4 performance, C5 theory, C6 identity |
| `predicate`, `args_json` | one predicate from the codebook with its arguments (see `configs/prompts/parser_instruction.txt`) |
| `scope` | condition such as "small caps", else empty |
| `hedge` | `absolute` / `typical` ("tends to", "usually") / `possible` ("may", "could") |
| `polarity` | `affirm`, or `deny` for negated claims ("does not depend on volume") |
| `horizon` | horizon bin of a predictive claim, else empty |
| `ambiguous` | `true` if the claim cannot be mapped to a predicate without guessing |

## Rules

1. **Vague terms** map through the codebook bins: "short-term" = ≤ 21 trading days, "medium" =
   22–126, "long" > 126; "very short" ≤ 5. A vague term with no bin → `ambiguous = true`.
2. **Anomaly names** map to codebook terms (Appendix C): "short-term reversal", "momentum",
   "low volatility", "illiquidity", "abnormal volume", "52-week high", … A named effect not in the
   codebook → `ambiguous = true`.
3. **Mechanism stories** ("because investors overreact") are `THEORY` (C5) — record them; they are
   counted, never scored.
4. **Inputs outside the panel** ("uses analyst revisions", "earnings surprise") are `DEPENDS_ON`
   claims with the codebook key of that input (C1.1).
5. **Identity claims** ("this is Alpha#12", "an RSI variant") are C6 — always record them.
6. **Direction claims** need an explicit input and direction ("stocks that rose recently score lower"
   → `SIGN(ret_5d, -)`; with no window stated use the codebook default of 5 days for "recent").
7. **Predictive claims** ("high values predict higher returns over the next week") → `PRED_SIGN`
   with `horizon` in days when stated.
8. Do not split a claim across sentences; do not merge two claims because they share a sentence.

## Agreement

Reported per batch: Cohen's κ (two raters), Krippendorff's α (≥ 0.80 reliable, ≥ 0.667 tentative)
and Gwet's AC1 (skewed prevalence). Proceed to the main run only if parser F1 ≥ 0.80 for C1/C6,
≥ 0.70 for C2/C3, and agreement ≥ the tentative threshold.

## Spot-check of verdicts (§10.7.3)

`spotcheck.csv` lists 100 random verdicts with the evidence bundle (static report, nudge statistics,
correlation statistics). Mark `reviewer_agrees` Y/N and comment. Disagreements are investigated by
re-running the verifier routine, never overruled by hand.

## Time and pay

≈ 4 minutes per rationale per annotator (≈ 260 rationales × 2 annotators ≈ 35 person-hours plus
adjudication). Annotators give informed consent and are paid fairly.
