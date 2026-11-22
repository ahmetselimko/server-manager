# Pre-registration: fleet-size sensitivity on the Azure 2019 VM trace

**Written before running the analysis.** The commit that adds this file is pushed before any K=5 or K=80 result exists. Any deviation from this plan will be reported explicitly in the results.

## Motivation
In the K=20 run (30 fleets × 20 subscriptions), most real fleets were smooth: only 8/30 had ≥0.5% SLA violations under reactive scaling. Smaller fleets aggregate fewer tenants and should be burstier, so a predictive safety margin has more room to help.

## Fixed design (unchanged from the K=20 run)
- **Data:** Azure Public Dataset V2, aggregated subscription × 5-min used cores (`data_prep/`).
- **Candidates:** subscriptions active in ≥95% of ticks. Fleets are 30 disjoint random subsets, seed 12345.
- **Fleet sizes:** K ∈ {5, 20, 80} subscriptions per fleet. K=20 is the existing run, which is deterministic and is reused.
- **Scaling and split:**
  - Each fleet is scaled by its own training-period peak.
  - Days 0–20 are used for training, days 21–29 for testing.
  - Hyperparameters are identical to the synthetic study.
- **Policies:** `naive`, `rf`, `lin`, `linq` (conditional CQR margin). Each is swept over margin quantiles 0.50–0.99 with the greedy selector.

## Primary metric
For each fleet and each pair of methods (A, B), compute the mean SLA violation % of each method over their common extra-energy range, intersected with [0%, 10%] above the reactive baseline. Linear interpolation is done on 200 grid points. The paired difference is A − B, and fleets whose common range is narrower than 0.5 pts are excluded and counted.

## Hypotheses and tests
- **H1 (main):** For every K, `linq` − `naive` < 0.
  - Test: paired bootstrap 95% CI over fleets (5000 resamples, seed 0).
  - H1 is supported for a given K if the CI excludes 0.
- **H2 (trend):** The advantage of `linq` over `naive` grows with fleet difficulty.
  - Test: pool all fleets from the three K values and compute Spearman's ρ between the fleet's reactive SLA violation % and the paired difference (`linq` − `naive`).
  - One-sided permutation test (10,000 permutations, seed 0). Supported if ρ < 0 and p < 0.05.
  - Also reported: the mean difference per K, in the expected order K=5 < K=20 < K=80. This is descriptive only.
- **H3 (secondary, no directional claim):** `linq` − `rf` and `linq` − `lin`, with the same CI method, reported for each K.

## Pre-defined subgroup
"Hard" fleets are those with reactive SLA ≥ 0.5%. This threshold is fixed here. Results for hard fleets are reported per K and pooled, labelled as a subgroup analysis.

## What will be reported regardless of outcome
- All three K values.
- All CIs and the number of excluded fleets.
- The ρ value and its p-value.
- Whether each hypothesis was supported.
