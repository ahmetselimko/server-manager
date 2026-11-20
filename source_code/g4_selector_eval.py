"""
g4_selector_eval.py — G4 kabul testi: enerji/geçiş duyarlı seçici (eopt) vs greedy vs eski opt
==============================================================================================

Kabul kriterleri (PROJE.md, G4), seçici dışındaki her şey sabitken (talep kaynağı linq, τ=0.90):
  K1  enerji(eopt) ≤ enerji(greedy) × 1.01
  K2  uyandırma(eopt) ≤ uyandırma(greedy) × 1.2
  K3  100 sunucuda karar süresi < 50 ms (p95)
λ yalnızca eğitim seed'inde (1000) seçildi; test: 30 seed × 7 gün (2001–2030).
Ek olarak "statik üst sınır": geçiş olmadan, aynı talep için seçilen kümelerin anlık gücü.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import copy
import json
import time
from multiprocessing import Pool

import numpy as np

import evaluate_policies as ep
from core import energy_selector as es
from core.server_profiles import create_server_fleet

LAMBDA_GRID = [0, 2, 5, 10, 20, 50]
SOURCE, LEVEL = "linq", 0.90
_G = {}


def _init(f, lam):
    _G["f"], ep.EOPT_LAMBDA = f, lam


def _run(seed):
    tr = ep.generate_trace(seed, 7, len(create_server_fleet()))
    f = _G["f"]; f.set_level(LEVEL)
    return {sel: ep.simulate(f"{SOURCE}_{sel}", tr, f) for sel in ("greedy", "eopt", "opt")}


def select_lambda(f):
    """Yalnızca eğitim seed'i (1000): enerjiyi en aza indiren λ (eşitlikte küçük olan)."""
    tr = ep.generate_trace(1000, 7, len(create_server_fleet()))
    f.set_level(LEVEL)
    base = ep.simulate(f"{SOURCE}_greedy", tr, f)["energy_kwh"]
    rows = []
    for lam in LAMBDA_GRID:
        ep.EOPT_LAMBDA = lam
        r = ep.simulate(f"{SOURCE}_eopt", tr, f)
        rows.append({"lambda": lam, "delta_energy_pct": 100 * (r["energy_kwh"] / base - 1),
                     "wakeups": r["wakeups"], "sla": r["sla_violation_pct"]})
    best = min(rows, key=lambda r: (round(r["delta_energy_pct"], 6), r["lambda"]))
    return best["lambda"], rows


def static_bound(n=2000):
    fleet = create_server_fleet(); cap = sum(s.max_requests_per_sec for s in fleet)
    rng = np.random.default_rng(1)

    def power(S, dem):
        c = sum(s.max_requests_per_sec for s in S); rho = min(1.0, dem / c); w = 0.0
        for s in S:
            s.current_cpu_pct, s.current_temp_c = rho * 100, 40.0
            w += s.compute_power_draw()
        return w
    d = []
    for _ in range(n):
        dem = rng.uniform(0.02, 0.6) * cap
        d.append(power(es.select_energy(fleet, dem, dem, set(), 0.0, ep.TARGET_UTIL, ep.MIN_ACTIVE), dem)
                 / power(ep.select_greedy(fleet, dem), dem) - 1)
    d = np.array(d)
    return {"mean_pct": float(100 * d.mean()), "min_pct": float(100 * d.min()),
            "share_better": float((d < -1e-9).mean())}


def timing():
    big = []
    for r in range(10):
        for s in create_server_fleet():
            s = copy.copy(s); s.server_id = f"{s.server_id}-{r}"; big.append(s)
    cap = sum(s.max_requests_per_sec for s in big); rng = np.random.default_rng(0); ts = []
    for _ in range(100):
        req = rng.uniform(0.1, 0.8) * cap * 0.7
        on = {s.server_id for s in big if rng.random() < 0.4}
        t0 = time.perf_counter(); es.select_energy(big, req, req * 0.9, on, 2.0, 0.7, 2, solver="local")
        ts.append(time.perf_counter() - t0)
    return {"median_ms": float(1000 * np.median(ts)), "p95_ms": float(1000 * np.percentile(ts, 95))}


def main():
    f = {x.name: x for x in ep.train_forecasters(1000, 30, False, len(create_server_fleet()),
                                                 use_rf=False)}[SOURCE]
    lam, lam_rows = select_lambda(f)
    print(f"λ seçimi (eğitim seed 1000): " + " ".join(f"λ={r['lambda']}:{r['delta_energy_pct']:+.1f}%"
                                                     for r in lam_rows) + f" → λ={lam}")
    with Pool(os.cpu_count() or 1, initializer=_init, initargs=(f, lam)) as pool:
        per = pool.map(_run, [2001 + i for i in range(30)])

    rng = np.random.default_rng(0)

    def paired(a, b):
        d = np.array([100 * (x[a]["energy_kwh"] / x[b]["energy_kwh"] - 1) for x in per])
        bm = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
        return float(d.mean()), [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]

    summ = {}
    for sel in ("greedy", "eopt", "opt"):
        summ[sel] = {k: float(np.mean([x[sel][k] for x in per]))
                     for k in ("energy_kwh", "sla_violation_pct", "wakeups", "avg_active_servers")}
    de_eopt, ci_eopt = paired("eopt", "greedy")
    de_opt, ci_opt = paired("opt", "greedy")
    sb, tm = static_bound(), timing()
    k1 = summ["eopt"]["energy_kwh"] <= summ["greedy"]["energy_kwh"] * 1.01
    k2 = summ["eopt"]["wakeups"] <= summ["greedy"]["wakeups"] * 1.2
    k3 = tm["p95_ms"] < 50

    print(f"\n{'seçici':<8} {'enerji kWh':>10} {'SLA %':>7} {'uyandırma':>10} {'aktif':>6}")
    for sel, r in summ.items():
        print(f"{sel:<8} {r['energy_kwh']:>10.1f} {r['sla_violation_pct']:>7.2f} {r['wakeups']:>10.0f} "
              f"{r['avg_active_servers']:>6.2f}")
    print(f"\nΔEnerji vs greedy: eopt {de_eopt:+.2f}% [{ci_eopt[0]:+.2f},{ci_eopt[1]:+.2f}] | "
          f"eski opt {de_opt:+.2f}% [{ci_opt[0]:+.2f},{ci_opt[1]:+.2f}]")
    print(f"Statik üst sınır (geçiş yok): eopt/greedy güç ort. {sb['mean_pct']:+.2f}% "
          f"(en iyi {sb['min_pct']:+.2f}%, eopt daha iyi {100*sb['share_better']:.0f}% örnekte)")
    print(f"Karar süresi (100 sunucu): medyan {tm['median_ms']:.1f} ms, p95 {tm['p95_ms']:.1f} ms")
    print(f"\nKABUL: K1 enerji {'✓' if k1 else '✗'} | K2 uyandırma {'✓' if k2 else '✗'} | "
          f"K3 süre {'✓' if k3 else '✗'} → {'KABUL' if (k1 and k2 and k3) else 'RED — seçici greedy kalır'}")
    json.dump({"schema_version": 1, "source": SOURCE, "level": LEVEL, "lambda": lam,
               "lambda_selection_train_seed_1000": lam_rows, "seeds": [2001 + i for i in range(30)],
               "summary": summ, "delta_energy_vs_greedy": {"eopt": [de_eopt, ci_eopt], "opt": [de_opt, ci_opt]},
               "static_bound": sb, "timing_100_servers": tm,
               "acceptance": {"K1_energy": bool(k1), "K2_wakeups": bool(k2), "K3_time": bool(k3)}},
              open("logs/g4_selector.json", "w"), indent=2, ensure_ascii=False)
    print("→ logs/g4_selector.json")


if __name__ == "__main__":
    main()
