# Results: fleet-size sensitivity (Azure 2019 VM trace)

This report follows [`preregistration_fleet_size.md`](preregistration_fleet_size.md), which was committed before the K=5 and K=80 runs (commit `8cf19fe`, 2026-11-18 13:13 +03). **There were no deviations from the plan.** One additional analysis was added afterwards; it is **not** pre-registered and is labelled *exploratory*.

To reproduce:

```bash
python azure_eval.py --k 5  --out logs/azure_pareto_k5
python azure_eval.py --k 80 --out logs/azure_pareto_k80
python fleet_size_analysis.py      # uses logs/azure_pareto_k{5,20,80}.json
```

## Primary metric
The primary metric is each fleet's mean SLA violation % over the common extra-energy range ∩ [0%, 10%]. Differences are paired, with bootstrap 95% CIs over fleets.

| K | Reactive SLA (mean) | Hard fleets | `linq` − `naive` | `linq` − `rf` | `linq` − `lin` |
|---|---|---|---|---|---|
| 5 | 0.61% | 7/30 | **−0.12 [−0.26, −0.03]** (−31%) | **−0.08 [−0.16, −0.03]** (−23%) | **−0.06 [−0.11, −0.03]** |
| 20 | 1.28% | 8/30 | **−0.17 [−0.35, −0.03]** (−29%) | −0.28 [−0.81, +0.05] (−40%) | **−0.08 [−0.17, −0.02]** |
| 80 | 0.64% | 5/30 | **−0.14 [−0.32, −0.01]** (−43%) | −0.01 [−0.04, +0.03] (−4%) | −0.03 [−0.09, +0.01] |

Bold entries are those whose CI excludes 0. Between 1 and 4 fleets per comparison were excluded because their common energy range was narrower than 0.5 pts; exact counts are in `logs/fleet_size_analysis.json`.

## Hypotheses
- **H1 (`linq` − `naive` < 0 for every K): supported for K = 5, 20 and 80.**
- **H2 (the advantage grows with fleet difficulty): supported.**
  - Spearman ρ = −0.62, one-sided permutation p = 0.0001, n = 85 fleets.
  - The descriptive ordering by K (K=5 < K=20 < K=80) did **not** hold.
  - The premise behind it was wrong: smaller fleets were *not* harder on average. Mean reactive SLA was 0.61% at K=5, 1.28% at K=20 and 0.64% at K=80, because each fleet is sized to its own training peak.
  - Fleet difficulty (reactive SLA), not fleet size, is what predicts the gain.
- **H3 (secondary):**
  - Against `rf`, `linq` is significantly better at K=5 (−23%), directionally better at K=20 (−40%, CI includes 0), and on par at K=80.
  - The conditional margin's own contribution (`linq` − `lin`) is significant at K=5 and K=20, but not at K=80.

## Exploratory (not pre-registered): absolute vs relative gain
H2 was registered on the *absolute* difference. Fleets with more violations naturally show larger absolute differences, so part of ρ = −0.62 is a scale effect.

Using the *relative* difference instead (`linq` − `naive`) / `naive`, over the 50 fleets where `naive` > 0.02%:
- **Correlation:** ρ = −0.19, one-sided p = 0.095, which is **not significant**.
- **Median relative reduction:** −18%.
- **By difficulty:**

| Reactive SLA | Median relative reduction | `linq` better in |
|---|---|---|
| < 0.25% | −15% | 15/21 fleets |
| 0.25–1% | −29% | 13/17 fleets |
| ≥ 1% | −30% | 11/12 fleets |

## Conclusion
- **Consistent effect:** on real Azure workloads, the conditional (CQR) safety margin reduces SLA violations compared with a fixed margin on last-value persistence, at every fleet size tested. The median relative reduction is −18%.
- **Where the absolute gain sits:** in the hardest fleets.
- **Relative gain:** it tends to be larger in harder fleets, but that trend is not statistically conclusive.
- **Against Random Forest:** the advantage depends on the fleet. It is significant for small fleets (K=5) and vanishes for large, smooth fleets (K=80).
