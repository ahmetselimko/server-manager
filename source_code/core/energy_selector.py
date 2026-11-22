"""
energy_selector.py — G4: Enerji ve geçiş maliyetine duyarlı sunucu kümesi seçici
================================================================================

Problem (her karar anında):
    min_S   Σ_{s∈S} [ idle_s + dyn_s · ρ_S^α ]  +  λ · Σ_{s∈S∖P} idle_s
    s.t.    cap(S) ≥ R_req / U_target,   |S| ≥ MIN_ACTIVE,   S ⊆ uygun sunucular

    ρ_S   = min(1, d̂ / cap(S))  — beklenen talep d̂'nin kapasiteye oranı (orantılı dağıtım)
    α     = 1.4                — sunucu güç modelindeki üs (server_profiles.compute_power_draw)
    P     = şu an açık (veya açılmakta olan) sunucular;  S∖P uyandırılacak olanlar
    λ     = uyandırma cezası (boşta güç cinsinden); geçiş (churn) maliyetini temsil eder.
            Yalnızca eğitim seed'inde seçilir.

İki çözücü:
  * exact  : n ≤ EXACT_MAX_N için tüm 2^n alt kümeyi numpy ile vektörize tarar (kesin optimum).
  * local  : büyük filolar için greedy başlangıçlı en-iyi-iyileştirme yerel araması
             (ekle / çıkar / takas hamleleri, vektörize).
"""
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

ALPHA       = 1.4
EXACT_MAX_N = 12
_SUBSET_CACHE: Dict[int, np.ndarray] = {}


def _subsets(n: int) -> np.ndarray:
    if n not in _SUBSET_CACHE:
        idx = np.arange(1 << n)[:, None]
        _SUBSET_CACHE[n] = ((idx >> np.arange(n)) & 1).astype(np.float64)
    return _SUBSET_CACHE[n]


def _arrays(servers):
    cap  = np.array([s.max_requests_per_sec for s in servers], dtype=float)
    idle = np.array([s.idle_power_watts for s in servers], dtype=float)
    dyn  = np.array([s.tdp_watts - s.idle_power_watts for s in servers], dtype=float)
    return cap, idle, dyn


def objective(mask: np.ndarray, cap, idle, dyn, exp_rps: float, wake_w: np.ndarray) -> np.ndarray:
    """mask: (m, n) 0/1. Dönüş: her alt küme için amaç değeri (W). Kısıt dışı → +inf."""
    c = mask @ cap
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(c > 0, np.minimum(1.0, exp_rps / c), 1.0)
    return mask @ idle + (mask @ dyn) * rho ** ALPHA + mask @ wake_w


def select_energy(available: Sequence, required_rps: float, expected_rps: float,
                  powered_ids: Set[str], lam: float, target_util: float,
                  min_active: int, solver: str = "auto") -> List:
    """Enerji + geçiş maliyetini en aza indiren uygun alt kümeyi döndürür."""
    avail = list(available)
    n = len(avail)
    if n <= min_active:
        return avail
    cap, idle, dyn = _arrays(avail)
    on = np.array([s.server_id in powered_ids for s in avail], dtype=bool)
    wake_w = lam * idle * (~on)
    need = required_rps / target_util
    exp_rps = max(0.0, expected_rps)

    if solver == "exact" or (solver == "auto" and n <= EXACT_MAX_N):
        M = _subsets(n)
        f = objective(M, cap, idle, dyn, exp_rps, wake_w)
        feasible = (M @ cap >= need - 1e-9) & (M.sum(axis=1) >= min_active)
        if not feasible.any():
            return avail                      # kapasite yetmiyor → hepsini aç
        best = int(np.argmin(np.where(feasible, f, np.inf)))
        return [avail[i] for i in np.flatnonzero(M[best])]
    # Çok başlangıçlı yerel arama: (1) açık sunuculardan, (2) boş kümeden başla; iyisini seç
    best_x, best_f = None, np.inf
    for start in (on, np.zeros(n, dtype=bool)):
        x = _local_search(cap, idle, dyn, exp_rps, wake_w, need, min_active, start)
        fx = float(objective(x[None], cap, idle, dyn, exp_rps, wake_w)[0])
        if fx < best_f:
            best_x, best_f = x, fx
    return [avail[i] for i in np.flatnonzero(best_x)]


def _local_search(cap, idle, dyn, exp_rps, wake_w, need, min_active, on) -> np.ndarray:
    n = len(cap)
    if cap.sum() < need:
        return np.ones(n)
    # Başlangıç: açık sunucular + verimlilik sırasıyla (rps / tam yük gücü) eksik kapasite
    x = on.astype(float).copy()
    order = np.argsort(-(cap / (idle + dyn)))
    for i in order:
        if x @ cap >= need and x.sum() >= min_active:
            break
        x[i] = 1.0

    def f_of(X):
        return objective(X, cap, idle, dyn, exp_rps, wake_w)

    def feasible(X):
        return (X @ cap >= need - 1e-9) & (X.sum(axis=1) >= min_active)

    cur = float(f_of(x[None])[0])
    eye = np.eye(n)
    for _ in range(4 * n):
        cands = []
        flips = np.abs(x[None] - eye)                       # tek bit çevir: ekle / çıkar
        cands.append(flips)
        ins, outs = np.flatnonzero(x == 0), np.flatnonzero(x == 1)
        if len(ins) and len(outs):                          # takas: birini çıkar, birini ekle
            I, O = np.meshgrid(ins, outs, indexing="ij")
            sw = np.repeat(x[None], I.size, axis=0)
            sw[np.arange(I.size), I.ravel()] = 1.0
            sw[np.arange(I.size), O.ravel()] = 0.0
            cands.append(sw)
        C = np.vstack(cands)
        fv = np.where(feasible(C), f_of(C), np.inf)
        j = int(np.argmin(fv))
        if fv[j] < cur - 1e-9:
            x, cur = C[j].copy(), float(fv[j])
        else:
            break
    return x
