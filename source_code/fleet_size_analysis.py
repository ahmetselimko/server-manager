"""
fleet_size_analysis.py — Ön kayıtlı filo büyüklüğü analizi (docs/preregistration_fleet_size.md)
================================================================================================

Girdi : logs/azure_pareto_k{5,20,80}.json  (azure_eval.py çıktıları)
Çıktı : logs/fleet_size_analysis.json, logs/fleet_size_analysis.png

Birincil ölçüt (ön kayıttaki tanımla birebir):
  Her filo ve (A, B) yöntem çifti için, iki yöntemin ortak ek-enerji aralığı ∩ [%0, %10]
  üzerinde (200 noktalı doğrusal enterpolasyon) ortalama SLA ihlali %; fark = A − B.
  Ortak aralığı 0.5 puandan dar olan filolar dışlanır ve sayılır.
H1 : her K için linq − naive < 0  (eşleştirilmiş bootstrap %95 GA, 5000 örnek, seed 0)
H2 : tüm filolar birleştirilince Spearman ρ(reaktif SLA, linq − naive) < 0,
     tek yönlü permütasyon testi (10.000, seed 0), p < 0.05
H3 : linq − rf ve linq − lin (yön iddiası yok)
Alt grup: "zor" filo = reaktif SLA ≥ %0.5 (önceden sabitlendi)
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

K_VALUES   = [5, 20, 80]
E_LO, E_HI = 0.0, 10.0
GRID       = 200
MIN_RANGE  = 0.5
N_BOOT     = 5000
N_PERM     = 10000
HARD_SLA   = 0.5
PAIRS      = [("linq", "naive"), ("linq", "rf"), ("linq", "lin"), ("rf", "naive")]


def _curve(fleet: Dict, m: str) -> Tuple[np.ndarray, np.ndarray]:
    e0 = fleet["reactive"]["energy_kwh"]
    e = np.array([100.0 * (p["energy_kwh"] / e0 - 1.0) for p in fleet["curves"][m]])
    s = np.array([p["sla_violation_pct"] for p in fleet["curves"][m]])
    o = np.argsort(e)
    return e[o], s[o]


def mean_sla_pair(fleet: Dict, a: str, b: str) -> Optional[Tuple[float, float]]:
    ea, sa = _curve(fleet, a)
    eb, sb = _curve(fleet, b)
    lo, hi = max(ea[0], eb[0], E_LO), min(ea[-1], eb[-1], E_HI)
    if hi - lo < MIN_RANGE:
        return None
    g = np.linspace(lo, hi, GRID)
    return float(np.interp(g, ea, sa).mean()), float(np.interp(g, eb, sb).mean())


def paired_ci(diff: np.ndarray, rng) -> Dict:
    if len(diff) < 3:
        return {"mean": None, "ci95": None, "n": int(len(diff))}
    bm = diff[rng.integers(0, len(diff), (N_BOOT, len(diff)))].mean(axis=1)
    return {"mean": float(diff.mean()),
            "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            "n": int(len(diff))}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def compare(fleets: List[Dict], a: str, b: str, rng) -> Dict:
    rows = [(f, mean_sla_pair(f, a, b)) for f in fleets]
    kept = [(f, v) for f, v in rows if v is not None]
    diff = np.array([v[0] - v[1] for _, v in kept])
    base = np.array([v[1] for _, v in kept])
    out = paired_ci(diff, rng)
    out.update({"excluded": len(rows) - len(kept),
                "wins": int((diff < -1e-9).sum()), "losses": int((diff > 1e-9).sum()),
                "ties": int((np.abs(diff) <= 1e-9).sum()),
                "mean_a": float(np.mean([v[0] for _, v in kept])) if kept else None,
                "mean_b": float(base.mean()) if kept else None,
                "relative_pct": (float(100 * diff.mean() / base.mean())
                                 if kept and base.mean() > 0 else None)})
    return out


def main():
    rng = np.random.default_rng(0)
    data = {}
    for k in K_VALUES:
        p = Path(f"logs/azure_pareto_k{k}.json")
        data[k] = json.load(open(p))["per_fleet"]

    res: Dict = {"per_k": {}, "preregistration": "docs/preregistration_fleet_size.md"}
    print("Birincil ölçüt: ortak ΔE∩[0,10]% aralığında ortalama SLA % — fark [%95 GA]\n")
    for k in K_VALUES:
        fl = data[k]
        hard = [f for f in fl if f["reactive"]["sla_violation_pct"] >= HARD_SLA]
        r = {"n_fleets": len(fl), "n_hard": len(hard),
             "reactive_sla_mean": float(np.mean([f["reactive"]["sla_violation_pct"] for f in fl])),
             "oracle_sla_mean": float(np.mean([f["oracle"]["sla_violation_pct"] for f in fl])),
             "test_cv_mean": float(np.mean([f["stats_test"]["cv"] for f in fl])),
             "test_ac1_mean": float(np.mean([f["stats_test"]["ac_lag1"] for f in fl])),
             "all": {}, "hard": {}}
        for a, b in PAIRS:
            r["all"][f"{a}-{b}"] = compare(fl, a, b, rng)
            r["hard"][f"{a}-{b}"] = compare(hard, a, b, rng)
        res["per_k"][str(k)] = r
        print(f"K={k:<3} | reaktif SLA ort={r['reactive_sla_mean']:.2f}% | zor filo={len(hard)}/{len(fl)} "
              f"| test CV={r['test_cv_mean']:.2f} ac1={r['test_ac1_mean']:.2f}")
        for grp in ("all", "hard"):
            for pair, c in r[grp].items():
                if c["mean"] is None:
                    print(f"   {grp:<4} {pair:<11} n={c['n']} (yetersiz)")
                    continue
                sig = "✓" if c["ci95"][1] < 0 else ("✗" if c["ci95"][0] > 0 else "·")
                print(f"   {grp:<4} {pair:<11} {c['mean_a']:.3f} vs {c['mean_b']:.3f} | fark={c['mean']:+.3f} "
                      f"[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] {sig} | göreli={c['relative_pct']:+.0f}% "
                      f"| kazanma {c['wins']}/{c['n']} (eşit {c['ties']}, dışlanan {c['excluded']})")

    # H1
    h1 = {str(k): res["per_k"][str(k)]["all"]["linq-naive"] for k in K_VALUES}
    h1_ok = {k: (v["ci95"] is not None and v["ci95"][1] < 0) for k, v in h1.items()}

    # H2: tüm filolar birleşik
    xs, ys, ks = [], [], []
    for k in K_VALUES:
        for f in data[k]:
            v = mean_sla_pair(f, "linq", "naive")
            if v is not None:
                xs.append(f["reactive"]["sla_violation_pct"]); ys.append(v[0] - v[1]); ks.append(k)
    xs, ys = np.array(xs), np.array(ys)
    rho = spearman(xs, ys)
    prng = np.random.default_rng(0)
    perm = np.array([spearman(xs, prng.permutation(ys)) for _ in range(N_PERM)])
    p_one = float((np.sum(perm <= rho) + 1) / (N_PERM + 1))
    h2_ok = rho < 0 and p_one < 0.05
    order = [h1[str(k)]["mean"] for k in K_VALUES]
    res["H1"] = {"per_k": h1, "supported": h1_ok}
    res["H2"] = {"spearman_rho": rho, "p_one_sided": p_one, "n": int(len(xs)),
                 "supported": bool(h2_ok), "mean_diff_by_k": dict(zip(map(str, K_VALUES), order)),
                 "expected_order_holds": bool(all(order[i] <= order[i + 1] for i in range(len(order) - 1)))}
    res["points"] = {"reactive_sla": xs.tolist(), "linq_minus_naive": ys.tolist(), "k": ks}

    # [KEŞİF — ön kayıtta yok] Mutlak farkın ölçek etkisini ayırmak için göreli kazanç
    xr, rel = [], []
    for k in K_VALUES:
        for f in data[k]:
            v = mean_sla_pair(f, "linq", "naive")
            if v is not None and v[1] > 0.02:      # naive ≈ 0 iken göreli fark tanımsız
                xr.append(f["reactive"]["sla_violation_pct"]); rel.append((v[0] - v[1]) / v[1])
    xr, rel = np.array(xr), np.array(rel)
    rho_r = spearman(xr, rel)
    prng2 = np.random.default_rng(0)
    perm_r = np.array([spearman(xr, prng2.permutation(rel)) for _ in range(N_PERM)])
    p_r = float((np.sum(perm_r <= rho_r) + 1) / (N_PERM + 1))
    bins = []
    for lo, hi in [(0, 0.25), (0.25, 1.0), (1.0, np.inf)]:
        m = (xr >= lo) & (xr < hi)
        bins.append({"reactive_sla_range": [lo, None if np.isinf(hi) else hi], "n": int(m.sum()),
                     "median_relative_pct": float(100 * np.median(rel[m])),
                     "linq_better": int((rel[m] < 0).sum())})
    res["exploratory_relative_H2"] = {
        "note": "Ön kayıtta YOK; mutlak farkın ölçek etkisini ayırmak için keşif amaçlı.",
        "spearman_rho": rho_r, "p_one_sided": p_r, "n": int(len(xr)),
        "median_relative_pct": float(100 * np.median(rel)), "by_difficulty": bins}

    print(f"\nH1 (linq − naive < 0, her K): " +
          " ".join(f"K={k}:{'DESTEKLENDİ' if ok else 'desteklenmedi'}" for k, ok in h1_ok.items()))
    print(f"H2 (ρ<0, tek yönlü p<0.05): ρ={rho:+.3f}, p={p_one:.4f}, n={len(xs)} → "
          f"{'DESTEKLENDİ' if h2_ok else 'desteklenmedi'} | K'ya göre ortalama fark "
          f"{', '.join(f'K={k}:{d:+.3f}' for k, d in zip(K_VALUES, order))} "
          f"(beklenen sıra {'tutuyor' if res['H2']['expected_order_holds'] else 'tutmuyor'})")
    ex = res["exploratory_relative_H2"]
    print(f"[KEŞİF] göreli kazanç: ρ={ex['spearman_rho']:+.3f}, p={ex['p_one_sided']:.4f}, n={ex['n']}, "
          f"medyan={ex['median_relative_pct']:+.0f}% | " +
          " ".join(f"SLA{b['reactive_sla_range']}: {b['median_relative_pct']:+.0f}% "
                   f"({b['linq_better']}/{b['n']})" for b in ex["by_difficulty"]))

    Path("logs").mkdir(exist_ok=True)
    json.dump(res, open("logs/fleet_size_analysis.json", "w"), indent=2, ensure_ascii=False)
    plot(res)
    print("\n→ logs/fleet_size_analysis.json, logs/fleet_size_analysis.png")


INK, INK2, GRID_C, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
K_COLORS  = {5: "#2a78d6", 20: "#eb6834", 80: "#1baf7a"}
K_MARKERS = {5: "o", 20: "s", 80: "^"}


def plot(res: Dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = res["points"]
    x, y, k = np.array(pts["reactive_sla"]), np.array(pts["linq_minus_naive"]), np.array(pts["k"])
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=160)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.axhline(0, color=INK2, lw=1)
    ax.axvline(HARD_SLA, color=GRID_C, lw=1.2, ls="--")
    ax.text(HARD_SLA, 1.0, " zor filo eşiği (önceden sabit)", transform=ax.get_xaxis_transform(),
            fontsize=8, color=INK2, va="top")
    for kv in K_VALUES:
        m = k == kv
        ax.scatter(np.maximum(x[m], 1e-3), y[m], s=38, marker=K_MARKERS[kv], color=K_COLORS[kv],
                   edgecolor=SURF, linewidth=1.2, zorder=3, label=f"K={kv} abonelik/filo")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("Filonun zorluğu: reaktif ölçeklemede SLA ihlali (%)", color=INK2, fontsize=9.5)
    ax.set_ylabel("Koşullu pay − son değer (SLA puanı)\n(negatif = koşullu pay daha iyi)",
                  color=INK2, fontsize=9.5)
    h2 = res["H2"]
    ax.set_title("Azure: koşullu payın kazancı zor filolarda büyüyor mu?", color=INK, fontsize=11,
                 loc="left", pad=12)
    ex = res.get("exploratory_relative_H2", {})
    ax.text(0.0, 1.01, f"ön kayıtlı H2 (mutlak fark): ρ = {h2['spearman_rho']:+.2f}, p = "
            f"{h2['p_one_sided']:.4f}, n = {h2['n']} · keşif (göreli fark): ρ = "
            f"{ex.get('spearman_rho', float('nan')):+.2f}, p = {ex.get('p_one_sided', float('nan')):.3f}",
            transform=ax.transAxes, fontsize=8, color=INK2)
    ax.grid(True, color=GRID_C, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID_C)
    ax.tick_params(colors=INK2, labelsize=8.5)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig("logs/fleet_size_analysis.png", facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    main()
