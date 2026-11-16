"""
pareto_margin.py — G2: Güvenlik payı taraması ve SLA–enerji Pareto eğrisi
===========================================================================

Soru: Tahminci (RF/LSTM), "son değer + pay" (naive) yönteminden gerçekten daha iyi mi,
yoksa farkı sadece farklı bir işletme noktasında (farklı pay) çalışmasından mı geliyor?

Yöntem
  * Her tahminci için güvenlik payı, doğrulama kalıntılarının q kantili olarak taranır
    (q ∈ QUANTILES). Her (yöntem, q) için 30 seed × 7 gün simülasyon (greedy seçici).
  * Her seed'de enerji, o seed'in reaktif baseline'ına (observed_greedy) göre ΔEnerji %
    olarak ifade edilir → (ΔEnerji, SLA%) eğrisi.
  * EŞ ENERJİ karşılaştırması: her seed'de eğriler ortak ΔEnerji düzeylerinde
    doğrusal interpolasyonla okunur; yöntem − naive farkı seed'ler üzerinde
    eşleştirilmiş bootstrap ile %95 GA'lı raporlanır.
  * EŞ SLA karşılaştırması: hedef SLA düzeyine ulaşmak için gereken ΔEnerji.

Çıktılar
  logs/pareto_margin.json   — ham noktalar + eş-enerji / eş-SLA tabloları
  logs/pareto_margin.png    — Pareto grafiği

Kullanım:
  python pareto_margin.py                  # 30 seed × 7 gün (çok çekirdekli)
  python pareto_margin.py --seeds 4 --days 2 --no-lstm   # hızlı deneme
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np

import evaluate_policies as ep
from core.server_profiles import create_server_fleet

QUANTILES      = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
ENERGY_LEVELS  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]      # ΔEnerji % (eş enerji)
SLA_LEVELS     = [1.5, 1.0, 0.75, 0.5]                      # SLA % (eş SLA)
N_BOOT         = 2000

_G: Dict = {}   # çalışan süreçlerde paylaşılan tahminciler


def _init(forecasters):
    _G["f"] = {f.name: f for f in forecasters}


def _run_seed(args):
    seed, days = args
    tr = ep.generate_trace(seed, days, len(create_server_fleet()))
    out = {"seed": seed,
           "reactive": ep.simulate("observed_greedy", tr),
           "oracle":   ep.simulate("oracle_greedy", tr),
           "curves":   {}}
    for name, f in _G["f"].items():
        pts = []
        for q in QUANTILES:
            f.margin = f.margin_at(q)
            r = ep.simulate(f"{name}_greedy", tr, f)
            pts.append({"q": q, "margin": f.margin, "energy_kwh": r["energy_kwh"],
                        "sla_violation_pct": r["sla_violation_pct"],
                        "dropped_req_pct": r["dropped_req_pct"],
                        "wakeups": r["wakeups"]})
        out["curves"][name] = pts
    return out


def _interp_sla_at_energy(de: np.ndarray, sla: np.ndarray, level: float) -> float:
    o = np.argsort(de)
    de, sla = de[o], sla[o]
    if level < de[0] or level > de[-1]:
        return np.nan
    return float(np.interp(level, de, sla))


def _interp_energy_at_sla(de: np.ndarray, sla: np.ndarray, level: float) -> float:
    o = np.argsort(sla)
    sla_s, de_s = sla[o], de[o]
    if level < sla_s[0] or level > sla_s[-1]:
        return np.nan
    return float(np.interp(level, sla_s, de_s))


def _paired(a: np.ndarray, b: np.ndarray, rng) -> Dict:
    """a − b, seed'ler üzerinde eşleştirilmiş; NaN içeren seed'ler atlanır."""
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    if len(d) < 5:
        return {"mean": None, "ci95": None, "n": int(len(d))}
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    bm = d[idx].mean(axis=1)
    return {"mean": float(d.mean()),
            "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            "n": int(len(d))}


def analyse(per_seed: List[Dict], methods: List[str]) -> Dict:
    rng = np.random.default_rng(0)
    n = len(per_seed)
    # seed bazında (ΔE, SLA) dizileri
    de, sla = {}, {}
    for m in methods:
        de[m]  = np.array([[100.0 * (p["energy_kwh"] / s["reactive"]["energy_kwh"] - 1.0)
                            for p in s["curves"][m]] for s in per_seed])
        sla[m] = np.array([[p["sla_violation_pct"] for p in s["curves"][m]] for s in per_seed])

    curve_mean = {m: [{"q": q, "delta_energy_pct": float(de[m][:, i].mean()),
                       "sla_violation_pct": float(sla[m][:, i].mean())}
                      for i, q in enumerate(QUANTILES)] for m in methods}

    iso_energy = {}
    for lvl in ENERGY_LEVELS:
        vals = {m: np.array([_interp_sla_at_energy(de[m][i], sla[m][i], lvl) for i in range(n)])
                for m in methods}
        both = np.all([np.isfinite(v) for v in vals.values()], axis=0)   # tüm yöntemlerde tanımlı seed'ler
        row = {m: (float(v[both].mean()) if both.any() else None) for m, v in vals.items()}
        row["n_seeds_common"] = int(both.sum())
        row["diff_vs_naive"] = {m: _paired(vals[m], vals["naive"], rng)
                                for m in methods if m != "naive"}
        iso_energy[str(lvl)] = row

    iso_sla = {}
    for lvl in SLA_LEVELS:
        vals = {m: np.array([_interp_energy_at_sla(de[m][i], sla[m][i], lvl) for i in range(n)])
                for m in methods}
        both = np.all([np.isfinite(v) for v in vals.values()], axis=0)   # tüm yöntemlerde tanımlı seed'ler
        row = {m: (float(v[both].mean()) if both.any() else None) for m, v in vals.items()}
        row["n_seeds_common"] = int(both.sum())
        row["diff_vs_naive"] = {m: _paired(vals[m], vals["naive"], rng)
                                for m in methods if m != "naive"}
        iso_sla[str(lvl)] = row

    ref = {
        "reactive_sla_pct": float(np.mean([s["reactive"]["sla_violation_pct"] for s in per_seed])),
        "oracle_sla_pct":   float(np.mean([s["oracle"]["sla_violation_pct"] for s in per_seed])),
        "oracle_delta_energy_pct": float(np.mean(
            [100.0 * (s["oracle"]["energy_kwh"] / s["reactive"]["energy_kwh"] - 1.0)
             for s in per_seed])),
    }
    return {"curve_mean": curve_mean, "iso_energy": iso_energy, "iso_sla": iso_sla,
            "reference": ref}


# ── Grafik ────────────────────────────────────────────────────────────────────
COLORS  = {"naive": "#2a78d6", "rf": "#eb6834", "lstm": "#1baf7a"}   # doğrulanmış kategorik
MARKERS = {"naive": "o", "rf": "s", "lstm": "^"}
LABELS  = {"naive": "Son değer + pay", "rf": "Random Forest + pay", "lstm": "LSTM + pay"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot(res: Dict, methods: List[str], path: str, n_seeds: int, days: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=160)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ref = res["reference"]

    for m in methods:
        c = res["curve_mean"][m]
        x = [p["delta_energy_pct"] for p in c]; y = [p["sla_violation_pct"] for p in c]
        ax.plot(x, y, color=COLORS[m], lw=2, zorder=3, label=LABELS[m])
        ax.scatter(x, y, s=34, marker=MARKERS[m], color=COLORS[m], edgecolor=SURF,
                   linewidth=1.5, zorder=4)
        # seçici doğrudan etiketler: eğrilerin birbirinden ayrıldığı noktalarda
        q_lbl, off, ha = {"naive": (0.90, (9, -3), "left"),
                          "lstm":  (0.85, (9, 2), "left"),
                          "rf":    (0.95, (-8, -12), "right")}[m]
        i = QUANTILES.index(q_lbl)
        ax.annotate(LABELS[m], (x[i], y[i]), xytext=off, textcoords="offset points",
                    fontsize=8.5, color=INK2, ha=ha, va="center")

    ax.scatter([0.0], [ref["reactive_sla_pct"]], s=70, marker="D", color=INK, zorder=5)
    ax.annotate("Reaktif (HPA benzeri)", (0.0, ref["reactive_sla_pct"]), xytext=(8, 0),
                textcoords="offset points", fontsize=8.5, color=INK, va="center")
    ax.scatter([ref["oracle_delta_energy_pct"]], [ref["oracle_sla_pct"]], s=90, marker="*",
               color=INK, zorder=5)
    ax.annotate("Kusursuz tahmin (oracle)", (ref["oracle_delta_energy_pct"], ref["oracle_sla_pct"]),
                xytext=(8, 4), textcoords="offset points", fontsize=8.5, color=INK)

    ax.set_xlabel("Ek enerji, reaktif baseline'a göre (%)", color=INK2, fontsize=9.5)
    ax.set_ylabel("SLA ihlali (tick'lerin %'si)", color=INK2, fontsize=9.5)
    ax.set_title("Güvenlik payı taraması: SLA–enerji dengesi (sol-alt daha iyi)",
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.text(0.0, 1.01, f"{n_seeds} seed × {days:g} gün ortalaması · greedy seçici · "
            f"pay = doğrulama hatasının q kantili (q=0.50…0.99)",
            transform=ax.transAxes, fontsize=8, color=INK2)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_ylim(bottom=-0.05)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--train-days", type=float, default=30.0)
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", type=str, default="logs/pareto_margin")
    args = ap.parse_args()

    print("Tahminciler eğitiliyor (eğitim seed=1000)…")
    forecasters = ep.train_forecasters(1000, args.train_days, not args.no_lstm,
                                       len(create_server_fleet()))
    methods = [f.name for f in forecasters]
    seeds = [2001 + i for i in range(args.seeds)]
    print(f"Tarama: {len(methods)} yöntem × {len(QUANTILES)} kantil × {len(seeds)} seed "
          f"({args.workers} çekirdek)…")
    with Pool(args.workers, initializer=_init, initargs=(forecasters,)) as pool:
        per_seed = pool.map(_run_seed, [(s, args.days) for s in seeds])

    res = analyse(per_seed, methods)

    print("\nEŞ ENERJİDE SLA ihlali (%), yöntem − naive farkı [%95 GA]:")
    for lvl, row in res["iso_energy"].items():
        line = f"  ΔE=+{float(lvl):>4.1f}% | " + " ".join(
            f"{m}={row[m]:.2f}" if row[m] is not None else f"{m}=—" for m in methods)
        for m, d in row["diff_vs_naive"].items():
            if d["mean"] is not None:
                line += f" | {m}−naive={d['mean']:+.2f} [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}] (n={d['n']})"
        print(line)
    print("\nEŞ SLA'DA gereken ek enerji (%), yöntem − naive farkı [%95 GA]:")
    for lvl, row in res["iso_sla"].items():
        line = f"  SLA={float(lvl):>4.2f}% | " + " ".join(
            f"{m}={row[m]:+.2f}" if row[m] is not None else f"{m}=—" for m in methods)
        for m, d in row["diff_vs_naive"].items():
            if d["mean"] is not None:
                line += f" | {m}−naive={d['mean']:+.2f} [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}] (n={d['n']})"
        print(line)
    r = res["reference"]
    print(f"\nReferans: reaktif SLA={r['reactive_sla_pct']:.2f}% | oracle SLA="
          f"{r['oracle_sla_pct']:.2f}% @ ΔE=+{r['oracle_delta_energy_pct']:.2f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"schema_version": 1,
                   "config": {"seeds": seeds, "days": args.days, "quantiles": QUANTILES,
                              "energy_levels": ENERGY_LEVELS, "sla_levels": SLA_LEVELS,
                              "selector": "greedy", "train_seed": 1000},
                   "forecasters": {f.name: f.val_metrics for f in forecasters},
                   "analysis": res, "per_seed": per_seed}, fh, indent=2, ensure_ascii=False)
    plot(res, methods, args.out + ".png", len(seeds), args.days)
    print(f"\n→ {args.out}.json, {args.out}.png")


if __name__ == "__main__":
    main()
