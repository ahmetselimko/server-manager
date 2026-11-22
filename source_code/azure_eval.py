"""
azure_eval.py — G5/G6: Gerçek trafik (Azure Public Dataset V2, 2019) üzerinde dış geçerlilik
==============================================================================================

Veri: data_prep/azure_aggregate.py ile üretilen abonelik × 5-dk talep matrisi
      (D[s,t] = Σ avg_cpu/100 × çekirdek; 6.687 abonelik × 8.640 tick = 30 gün).

Protokol (sentetik deneyle birebir aynı ölçütler):
  * FİLO: Rastgele seçilen K aboneliğin toplam talebi bir "filo"nun talebidir. N_FLEETS filo,
    birbirinden AYRIK abonelik kümeleriyle kurulur (bağımsız örnekler). Yalnızca 30 günün
    en az %MIN_ACTIVE'inde talebi olan abonelikler aday olur.
  * NORMALİZASYON: x = D_filo / max(D_filo[eğitim]); yani filo, eğitim dönemindeki zirvesine
    göre boyutlandırılır (kapasite planlaması). Test verisi ölçeklemeye katılmaz. x ≤ X_CLIP.
  * ZAMAN AYRIMI: gün 0–20 eğitim (kendi içinde %80/%20 doğrulama), gün 21–29 test.
    21 gün = 3 hafta olduğundan saat ve haftanın günü özellikleri kaymaz.
  * Her filonun tahmincileri YALNIZCA kendi eğitim döneminde eğitilir (seed 1000 sentetik
    veri kullanılmaz). Hiperparametreler sentetik deneydekiyle aynıdır; Azure test
    dönemine bakılarak hiçbir şey seçilmez.
  * Değerlendirme: pareto_margin.py ile aynı kantil taraması, eş-enerji / eş-SLA
    karşılaştırması ve eşleştirilmiş bootstrap (filolar üzerinde).

Kullanım:
  python azure_eval.py                         # 30 filo
  python azure_eval.py --fleets 4 --quick      # hızlı deneme
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np

import evaluate_policies as ep
import pareto_margin as pm
from core.server_profiles import create_server_fleet

TICKS_PER_DAY = 288
TRAIN_DAYS    = 21
N_FLEETS      = 30
K_SUBS        = 20          # filo başına abonelik sayısı
MIN_ACTIVE    = 0.95        # adayın talebi olan tick oranı (30 gün)
X_CLIP        = 1.5
FLEET_SEED    = 12345
METHODS       = ["naive", "rf", "lin", "linq"]


def build_fleets(D: np.ndarray, n_fleets: int, k: int, seed: int = FLEET_SEED):
    active = (D > 0).mean(axis=1) >= MIN_ACTIVE
    cand = np.flatnonzero(active)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(cand)
    if len(perm) < n_fleets * k:
        raise ValueError(f"aday abonelik yetersiz: {len(perm)} < {n_fleets}×{k}")
    return [np.sort(perm[i * k:(i + 1) * k]) for i in range(n_fleets)], int(len(cand))


def fleet_series(D: np.ndarray, subs: np.ndarray):
    raw = D[subs].sum(axis=0).astype(float)
    split = TRAIN_DAYS * TICKS_PER_DAY
    scale = raw[:split].max()
    x = np.clip(raw / scale, 0.0, X_CLIP)
    return x[:split], x[split:], raw, scale


def describe(x: np.ndarray) -> Dict:
    d = np.diff(x)
    ac = lambda lag: float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
    return {"mean": float(x.mean()), "cv": float(x.std() / max(x.mean(), 1e-9)),
            "ac_lag1": ac(1), "ac_1h": ac(12), "ac_1d": ac(TICKS_PER_DAY),
            "mean_abs_step": float(np.abs(d).mean()), "p99_abs_step": float(np.percentile(np.abs(d), 99))}


_D = {}


def _init(D):
    _D["D"] = D


def _run_fleet(args):
    fid, subs, quick = args
    D = _D["D"]
    x_tr, x_te, raw, scale = fleet_series(D, subs)
    n_srv = len(create_server_fleet())
    fs = ep.train_forecasters_on(x_tr, np.zeros(len(x_tr), dtype=int), use_lstm=False,
                                 use_rf=not quick, verbose=False)
    fmap = {f.name: f for f in fs}
    tr = ep.trace_from_series(x_te, None, n_srv, failure_seed=5000 + fid)
    out = {"seed": fid, "subs": subs.tolist(), "scale_cores": float(scale),
           "stats_test": describe(x_te), "stats_train": describe(x_tr),
           "forecasters": {f.name: f.val_metrics for f in fs},
           "reactive": ep.simulate("observed_greedy", tr),
           "oracle": ep.simulate("oracle_greedy", tr), "curves": {}}
    for name in [m for m in METHODS if m in fmap]:
        f, pts = fmap[name], []
        for q in pm.QUANTILES:
            f.set_level(q)
            r = ep.simulate(f"{name}_greedy", tr, f)
            pts.append({"q": q, "energy_kwh": r["energy_kwh"],
                        "sla_violation_pct": r["sla_violation_pct"],
                        "dropped_req_pct": r["dropped_req_pct"], "wakeups": r["wakeups"]})
        out["curves"][name] = pts
    lq = out["forecasters"].get("linq", {})
    print(f"  filo {fid:>2}: zirve={scale:,.0f} çekirdek | test ac1={out['stats_test']['ac_lag1']:.3f} "
          f"cv={out['stats_test']['cv']:.2f} | lin R²={out['forecasters']['lin']['r2']:.3f} "
          f"| linq kapsama(0.9)={lq.get('coverage', {}).get('0.9', float('nan')):.3f} "
          f"| reaktif SLA={out['reactive']['sla_violation_pct']:.2f}%", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/azure_sub_demand.npz")
    ap.add_argument("--fleets", type=int, default=N_FLEETS)
    ap.add_argument("--k", type=int, default=K_SUBS)
    ap.add_argument("--quick", action="store_true", help="RF'yi atla (hızlı deneme)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="logs/azure_pareto")
    args = ap.parse_args()

    z = np.load(args.data, allow_pickle=True)
    D = z["D"]
    if (D[:, -1] == 0).all():          # izin son 5 dakikası veri setinde yok (son ts=2591400)
        D = D[:, :-1]
    fleets, n_cand = build_fleets(D, args.fleets, args.k)
    print(f"Azure: {D.shape[0]:,} abonelik × {D.shape[1]:,} tick | aday={n_cand:,} | "
          f"{args.fleets} filo × {args.k} abonelik (ayrık)")
    with Pool(args.workers, initializer=_init, initargs=(D,)) as pool:
        per = pool.map(_run_fleet, [(i, f, args.quick) for i, f in enumerate(fleets)])

    methods = [m for m in METHODS if m in per[0]["curves"]]
    res = pm.analyse(per, methods)

    def _fmt(d):
        return (f"{d['mean']:+.2f} [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}] n={d['n']}"
                if d["mean"] is not None else f"— (n={d['n']})")

    for title, key, unit in [("EŞ ENERJİDE SLA ihlali (%)", "iso_energy", "ΔE=+{:.1f}%"),
                             ("EŞ SLA'DA gereken ek enerji (%)", "iso_sla", "SLA={:.2f}%")]:
        print(f"\n{title} — ortak filo ortalaması; farklar eşleştirilmiş [%95 GA]:")
        for lvl, row in res[key].items():
            vals = " ".join(f"{m}={row[m]:.2f}" if row[m] is not None else f"{m}=—" for m in methods)
            print(f"  {unit.format(float(lvl)):<11} | {vals}")
            for base in [b for b in pm.BASELINES if b in methods]:
                diffs = "  ".join(f"{m}−{base}={_fmt(d)}" for m, d in row[f"diff_vs_{base}"].items()
                                  if m not in pm.BASELINES or base == "naive")
                print(f"  {'':<11}   {diffs}")
    r = res["reference"]
    print(f"\nReferans: reaktif SLA={r['reactive_sla_pct']:.2f}% | oracle SLA="
          f"{r['oracle_sla_pct']:.2f}% @ ΔE=+{r['oracle_delta_energy_pct']:.2f}%")
    st = {k: float(np.mean([p["stats_test"][k] for p in per])) for k in per[0]["stats_test"]}
    print("Test dönemi talep istatistikleri (filo ortalaması): " +
          " ".join(f"{k}={v:.3f}" for k, v in st.items()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"schema_version": 1, "dataset": "Azure Public Dataset V2 (2019 VM trace)",
                   "config": {"fleets": args.fleets, "k_subs": args.k, "train_days": TRAIN_DAYS,
                              "min_active": MIN_ACTIVE, "x_clip": X_CLIP, "fleet_seed": FLEET_SEED,
                              "quantiles": pm.QUANTILES, "selector": "greedy",
                              "n_candidate_subs": n_cand, "config_sha256": pm.config_hash(TRAIN_DAYS)},
                   "test_stats_mean": st, "analysis": res, "per_fleet": per},
                  fh, indent=2, ensure_ascii=False)
    pm.plot(res, methods, args.out + ".png", len(per), 30 - TRAIN_DAYS,
            title="Azure VM izi (2019): SLA–enerji dengesi (sol-alt daha iyi)",
            subtitle=f"{len(per)} filo × {30 - TRAIN_DAYS} günlük test · her filo kendi ilk "
                     f"{TRAIN_DAYS} gününde eğitildi · greedy seçici")
    print(f"\n→ {args.out}.json, {args.out}.png")


if __name__ == "__main__":
    main()
