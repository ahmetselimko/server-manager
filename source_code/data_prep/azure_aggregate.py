"""
azure_aggregate.py — Azure Public Dataset V2 (2019) VM CPU izini abonelik bazında toplar
=========================================================================================

Kaynak: https://github.com/Azure/AzurePublicDataset (AzurePublicDatasetV2.md)
  * vm_cpu_readings-file-*-of-195.csv.gz : timestamp, vm id, min cpu, max cpu, avg cpu (5 dk)
  * vmtable.csv.gz                       : vm id, subscription id, …, core bucket (10. sütun)

Çıktı: abonelik × 5-dakikalık tick matrisi
    D[s, t] = Σ_{VM ∈ s} (avg_cpu[VM, t] / 100) × çekirdek[VM]       ("kullanılan çekirdek")
Ham dosyalar ~156 GB olduğundan diske yazılmaz; her dosya indirilirken akış halinde
işlenir ve yalnızca (anahtar, toplam) çiftleri kaydedilir.

Kullanım:
  python azure_aggregate.py index  vmtable.csv.gz  out_dir          # VM → abonelik/çekirdek dizini
  curl -sSL <url> | python azure_aggregate.py part out_dir <i>       # tek dosyayı akıştan işle
  python azure_aggregate.py merge  out_dir                           # parçaları birleştir
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TICK = 300
N_TICKS = 30 * 24 * 12          # 30 gün × 288 = 8640
CORE_MAP = {">24": 30.0}        # açık uçlu kova için temsilî değer (raporda belirtilmeli)


def cmd_index(vmtable_path: str, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    vt = pd.read_csv(vmtable_path, header=None, usecols=[0, 1, 9],
                     names=["vm", "sub", "cores"], dtype=str)
    cores = vt["cores"].map(lambda c: CORE_MAP.get(c, None) or float(c)).astype(np.float32)
    sub_codes, sub_uniques = pd.factorize(vt["sub"])
    pd.to_pickle({"vm_index": pd.Index(vt["vm"].values),
                  "sub_code": sub_codes.astype(np.int32),
                  "cores": cores.values,
                  "subs": np.asarray(sub_uniques)}, out / "vm_index.pkl")
    print(f"VM={len(vt):,}  abonelik={len(sub_uniques):,}  "
          f"çekirdek kovaları={sorted(vt['cores'].unique())}")


def cmd_part(out_dir: str, i: int):
    out = Path(out_dir)
    t0 = time.time()
    ix = pd.read_pickle(out / "vm_index.pkl")
    vm_index, sub_code, cores = ix["vm_index"], ix["sub_code"], ix["cores"]
    n_sub = len(ix["subs"])
    acc: dict = {}
    rows = missing = 0
    tmin, tmax = 1 << 62, -1
    reader = pd.read_csv(sys.stdin.buffer, compression="gzip", header=None,
                         usecols=[0, 1, 4], names=["ts", "vm", "avg"],
                         dtype={"ts": np.int64, "vm": str, "avg": np.float64},
                         chunksize=1_000_000)
    for ch in reader:
        pos = vm_index.get_indexer(ch["vm"].values)
        ok = pos >= 0
        missing += int((~ok).sum())
        pos = pos[ok]
        tick = (ch["ts"].values[ok] // TICK).astype(np.int64)
        w = ch["avg"].values[ok] / 100.0 * cores[pos]
        key = sub_code[pos].astype(np.int64) * N_TICKS + tick
        s = pd.Series(w).groupby(key).sum()
        for k, v in zip(s.index.values, s.values):
            acc[k] = acc.get(k, 0.0) + v
        rows += len(ch)
        tmin, tmax = min(tmin, int(ch["ts"].min())), max(tmax, int(ch["ts"].max()))
    keys = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
    vals = np.fromiter(acc.values(), dtype=np.float64, count=len(acc))
    np.savez_compressed(out / f"part_{i:03d}.npz", keys=keys, vals=vals,
                        rows=rows, missing=missing, tmin=tmin, tmax=tmax, n_sub=n_sub)
    print(f"part {i}: satır={rows:,} eşleşmeyen VM={missing:,} ts=[{tmin},{tmax}] "
          f"anahtar={len(keys):,} süre={time.time()-t0:.0f}s", flush=True)


def cmd_merge(out_dir: str):
    out = Path(out_dir)
    ix = pd.read_pickle(out / "vm_index.pkl")
    n_sub = len(ix["subs"])
    D = np.zeros(n_sub * N_TICKS, dtype=np.float64)
    parts = sorted(out.glob("part_*.npz"))
    rows = missing = 0
    for p in parts:
        z = np.load(p)
        np.add.at(D, z["keys"], z["vals"])
        rows += int(z["rows"]); missing += int(z["missing"])
    D = D.reshape(n_sub, N_TICKS).astype(np.float32)
    np.savez_compressed(out / "azure_sub_demand.npz", D=D, subs=ix["subs"])
    tot = D.sum(axis=0)
    print(f"parça={len(parts)} satır={rows:,} eşleşmeyen={missing:,} "
          f"toplam talep (çekirdek) ort={tot.mean():.0f} min={tot.min():.0f} max={tot.max():.0f}")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "index":
        cmd_index(sys.argv[2], sys.argv[3])
    elif c == "part":
        cmd_part(sys.argv[2], int(sys.argv[3]))
    elif c == "merge":
        cmd_merge(sys.argv[2])
