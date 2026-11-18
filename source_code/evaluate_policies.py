"""
evaluate_policies.py — Yer-gerçeği (ground-truth) talep ile adil politika karşılaştırması
==========================================================================================

train.py içindeki run_autoscaler_evaluation() döngüseldir: aktif sunucunun CPU'su
modelin kendi tahminine eşitlenir ve sunucu uyutulduğunda talep kalan sunuculara
yeniden dağılmaz. Bu dosya, ölçekleme politikalarını aşağıdaki kurallarla değerlendirir:

  * Filo talebi D(t) (istek/sn) TrafficGenerator'dan üretilir ve TÜM politikalar için
    AYNIDIR (aynı seed → aynı talep, aynı arıza takvimi).
  * Talep, hizmet veren sunuculara kapasiteleriyle orantılı dağıtılır:
        ρ(t) = D(t) / Σ max_requests_per_sec   (hizmet veren sunucular)
    ρ > 1 → istekler düşer; ρ > SLA_UTIL → SLA ihlali sayılır.
  * Uyandırılan sunucu BOOT_TICKS tick sonra hizmete girer (açılış gecikmesi);
    açılırken boşta güç çeker. Tahminin değeri tam olarak bu gecikmeden gelir.
  * Sunucu arızaları önceden üretilir; arızalı sunucu hizmet veremez ve seçilemez.
  * Küçültme (scale-in) tüm politikalarda aynı stabilizasyon penceresine tabidir;
    yeni sunucular açılana kadar eskiler kapatılmaz (make-before-break).

Politikalar = <talep kaynağı>_<küme seçici>  (+ all_on)
-------------------------------------------------------
  Talep kaynağı:
    observed : t anında gözlenen talep (reaktif)
    naive    : son değer + doğrulamadan kalibre edilmiş q90 güvenlik payı
               (tahminin katkısını "sadece pay eklemek"ten ayırmak için)
    rf, lstm : model tahmini max(x[t+1..t+H]) + q90 güvenlik payı
    oracle   : kusursuz gelecek bilgisi (ulaşılabilir üst sınır)
  Küme seçici:
    greedy   : Kubernetes HPA benzeri; rps/W verimlilik sırasıyla kapasite ≥ talep/0.70
    opt      : projenin AutoscalerEngine._find_optimal_combo() kombinasyon optimizasyonu
  observed_greedy = HPA benzeri reaktif BASELINE; all_on = hepsi açık.

Not (talep modeli): Orijinal üreteç her tick'te deseni yeniden rastgele seçtiği için
yük neredeyse beyaz gürültüdür (lag-1 otokorelasyon ≈ 0.28). Gerçek trafikte rejimler
bir süre devam eder; bu yüzden burada desen ortalama REGIME_MEAN_TICKS tick boyunca
korunur ve EMA ile hafif bir atalet eklenir. Bu varsayım raporda açıkça belirtilmelidir.

Kullanım:
    python evaluate_policies.py                 # varsayılan: 30 test seed'i × 7 gün
    python evaluate_policies.py --seeds 3 --days 3 --no-lstm   # hızlı deneme
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.autoscaler import AutoscalerEngine, TARGET_UTIL_PCT
from core.server_profiles import create_server_fleet
from core.traffic_generator import TrafficGenerator, WORKLOAD_CLASSES
from models.models import RandomForestRegressor, LSTMModel

# ── Deney parametreleri ──────────────────────────────────────────────────────
DT_SECONDS          = 300.0     # 5 dakikalık tick
DEMAND_SCALE        = 0.60      # yük=1.0 iken talep = filo kapasitesinin %60'ı
REGIME_MEAN_TICKS   = 12        # desen ortalama 1 saat sürer
EMA_ALPHA           = 0.4       # talep ataleti
BOOT_TICKS          = 1         # uyandırılan sunucu 1 tick sonra hizmete girer
SCALE_IN_STAB_TICKS = 3         # küçültme için istenen kümenin 3 tick boyunca küçük kalması
TARGET_UTIL         = TARGET_UTIL_PCT / 100.0   # 0.70 — projedeki hedefle aynı
SLA_UTIL            = 0.85      # bu kullanımın üstü SLA ihlali (gecikme patlar)
FAIL_PROB_PER_TICK  = 1.5e-4    # sunucu başına tick başına arıza olasılığı
FAIL_MIN_TICKS      = 6
FAIL_MAX_TICKS      = 24
MIN_ACTIVE          = 2
N_LAGS              = 12
HORIZON             = 1 + BOOT_TICKS   # t+1 … t+1+BOOT talebini karşılamak gerekir
WL_INDEX            = {w: i for i, w in enumerate(WORKLOAD_CLASSES)}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Talep izi (trace) üretimi
# ══════════════════════════════════════════════════════════════════════════════

def generate_trace(seed: int, days: float, n_servers: int) -> Dict:
    """Normalize yük x(t)∈[0,1], iş yükü etiketi ve arıza takvimi üretir."""
    gen = TrafficGenerator(seed=seed)
    rng = np.random.default_rng(seed + 7919)
    n   = int(days * 86400 / DT_SECONDS)

    load = np.zeros(n)
    wl   = np.zeros(n, dtype=int)
    pattern, ema = None, None
    for k in range(n):
        t = k * DT_SECONDS
        if pattern is None or rng.random() < 1.0 / REGIME_MEAN_TICKS:
            pattern = gen.select_pattern_for_tick(t)
        raw = gen.get_load_for_pattern(pattern, t)
        ema = raw if ema is None else (1 - EMA_ALPHA) * ema + EMA_ALPHA * raw
        load[k] = ema
        wl[k]   = WL_INDEX.get(gen.select_workload_for_tick(t).value, 0)

    # Arıza takvimi: failed[k, i] = True → sunucu i, tick k'da arızalı
    failed = np.zeros((n, n_servers), dtype=bool)
    for i in range(n_servers):
        k = 0
        while k < n:
            if rng.random() < FAIL_PROB_PER_TICK:
                dur = int(rng.integers(FAIL_MIN_TICKS, FAIL_MAX_TICKS + 1))
                failed[k:k + dur, i] = True
                k += dur
            else:
                k += 1
    return {"load": load, "wl": wl, "failed": failed, "n": n}


# ══════════════════════════════════════════════════════════════════════════════
# 2. Tahminciler (filo seviyesinde talep)
# ══════════════════════════════════════════════════════════════════════════════

def _time_feats(k: int) -> List[float]:
    t   = k * DT_SECONDS
    hod = (t % 86400) / 86400.0
    dow = ((t // 86400) % 7) / 7.0
    return [np.sin(2 * np.pi * hod), np.cos(2 * np.pi * hod),
            np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow)]


def _wl_onehot(w: int) -> List[float]:
    v = [0.0] * len(WORKLOAD_CLASSES)
    v[w] = 1.0
    return v


def _tab_row(load: np.ndarray, wl: np.ndarray, k: int) -> List[float]:
    """t anında bilinen tablo özellikleri: x[t..t-N+1], zaman, iş yükü."""
    return load[k - N_LAGS + 1:k + 1][::-1].tolist() + _time_feats(k) + _wl_onehot(wl[k])


def build_tabular(load: np.ndarray, wl: np.ndarray):
    """t anında bilinenlerle (x[t-N+1..t], zaman, iş yükü) → hedef max(x[t+1..t+H])."""
    X, y, idx = [], [], []
    for k in range(N_LAGS - 1, len(load) - HORIZON):
        lags = load[k - N_LAGS + 1:k + 1][::-1].tolist()
        X.append(lags + _time_feats(k) + _wl_onehot(wl[k]))
        y.append(load[k + 1:k + 1 + HORIZON].max())
        idx.append(k)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), idx


def build_sequences(load: np.ndarray, wl: np.ndarray, seq_len: int):
    X, y = [], []
    for k in range(seq_len - 1, len(load) - HORIZON):
        seq = [[load[j]] + _time_feats(j) + _wl_onehot(wl[j])
               for j in range(k - seq_len + 1, k + 1)]
        X.append(seq)
        y.append(load[k + 1:k + 1 + HORIZON].max())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class Forecaster:
    """Eğitilmiş model + doğrulama kalıntılarından üretilen belirsizlik payı."""

    def __init__(self, name: str, model, kind: str, seq_len: int = 0):
        self.name, self.model, self.kind, self.seq_len = name, model, kind, seq_len
        self.margin = 0.0
        self.val_metrics: Dict = {}

    def predict_at(self, load_hist: np.ndarray, wl_hist: np.ndarray, k: int) -> float:
        if self.kind == "naive":
            return float(load_hist[k])
        if self.kind in ("rf", "tab"):
            if k < N_LAGS - 1:
                return float(load_hist[k])
            x = np.array([_tab_row(load_hist, wl_hist, k)], dtype=np.float32)
            return float(self.model.predict(x)[0])
        if k < self.seq_len - 1:
            return float(load_hist[k])
        seq = np.array([[[load_hist[j]] + _time_feats(j) + _wl_onehot(wl_hist[j])
                         for j in range(k - self.seq_len + 1, k + 1)]], dtype=np.float32)
        return float(np.ravel(self.model.predict(seq))[0])

    def margin_at(self, q: float) -> float:
        """Doğrulama kalıntılarının q kantili (≥0) — pay taraması için."""
        return max(0.0, float(np.quantile(self.val_resid, q)))

    def set_level(self, q: float):
        """İşletme noktasını seç (sabit pay: doğrulama kalıntılarının q kantili)."""
        self.level = q
        self.margin = self.margin_at(q)

    def margin_now(self, load_hist: np.ndarray, wl_hist: np.ndarray, k: int) -> float:
        """t anındaki güvenlik payı. Sabit-pay tahmincilerde sabittir."""
        return self.margin

    def calibrate(self, yv: np.ndarray, pv: np.ndarray, q: float = 0.90):
        resid = yv - pv                      # pozitif → eksik tahmin (tehlikeli taraf)
        self.val_resid = np.asarray(resid, dtype=float)
        self.margin = max(0.0, float(np.quantile(resid, q)))
        ss_res = float(np.sum((yv - pv) ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2)) or 1.0
        self.val_metrics = {"mae": float(np.mean(np.abs(resid))),
                            "r2": 1.0 - ss_res / ss_tot,
                            "margin_q90": self.margin}


class LinearAR:
    """Ridge regresyon nokta tahmincisi (numpy ile hızlı tahmin).
    G3 teşhisi: rejim değişimi olmayan tick'lerde yük ≈ AR(2); ağaçlar doğrusal
    ekstrapolasyonu iyi temsil edemez. Katsayılar tamamen veriden öğrenilir."""

    def __init__(self, alpha: float = 1e-3):
        from sklearn.linear_model import Ridge
        self._m = Ridge(alpha=alpha)

    def fit(self, X, y):
        self._m.fit(X, y)
        self.coef_, self.intercept_ = self._m.coef_.astype(float), float(self._m.intercept_)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


def fit_ar2(load: np.ndarray):
    """x[t] ≈ a·x[t-1] + b·x[t-2] katsayılarını eğitim verisinden kestirir (EKK)."""
    A = np.column_stack([load[1:-1], load[:-2]])
    a, b = np.linalg.lstsq(A, load[2:], rcond=None)[0]
    return float(a), float(b)


def _state_feats(load: np.ndarray, k: int, ar2) -> List[float]:
    """Belirsizlik payı için durum özellikleri (yalnızca t ve öncesi):
    seviye, son iki değişim, AR(2) 'sürpriz'i (rejim değişimi göstergesi)."""
    a, b = ar2
    x0, x1, x2 = load[k], load[k - 1], load[k - 2]
    x3 = load[k - 3]
    surprise  = x0 - (a * x1 + b * x2)
    surprise1 = x1 - (a * x2 + b * x3)
    return [x0, x0 - x1, x1 - x2, surprise, abs(surprise), surprise1, abs(surprise1)]


class ConditionalMarginForecaster(Forecaster):
    """Nokta tahminci (LinearAR) + duruma koşullu kantil payı.
    Her τ için, nokta tahmincinin kalıntısının τ-kantilini HistGradientBoosting
    (loss='quantile') ile duruma (seviye, değişim, sürpriz, saat, iş yükü) bağlı
    olarak tahmin eder. set_level(τ) işletme noktasını seçer."""

    def __init__(self, name: str, base: Forecaster, ar2, taus: List[float]):
        super().__init__(name, base.model, "tab")
        self.base, self.ar2, self.taus = base, ar2, list(taus)
        self.qmodels: Dict[float, object] = {}
        self.coverage: Dict[float, float] = {}
        self.level = 0.90

    def _feat(self, load, wl, k) -> List[float]:
        return _state_feats(load, k, self.ar2) + _time_feats(k) + _wl_onehot(wl[k])

    def fit_margins(self, load, wl, idx_tr, resid_tr, idx_cal, resid_cal, idx_va, resid_va):
        """τ başına kantil modeli eğitir; ardından ayrı bir kalibrasyon diliminde
        konformal kaydırma (CQR) uygular ve kapsamayı dokunulmamış doğrulamada ölçer."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        F_tr  = np.array([self._feat(load, wl, k) for k in idx_tr])
        F_cal = np.array([self._feat(load, wl, k) for k in idx_cal])
        F_va  = np.array([self._feat(load, wl, k) for k in idx_va])
        self.shift: Dict[float, float] = {}
        self.coverage_raw: Dict[float, float] = {}
        for tau in self.taus:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=tau, **HGB_PARAMS)
            m.fit(F_tr, resid_tr)
            self.qmodels[tau] = m
            # CQR: kalibrasyon dilimindeki aşımın τ kantili kadar kaydır
            n_cal = len(resid_cal)
            lvl = min(1.0, np.ceil((n_cal + 1) * tau) / n_cal)
            self.shift[tau] = float(np.quantile(resid_cal - m.predict(F_cal), lvl))
            pred_va = m.predict(F_va)
            self.coverage_raw[tau] = float(np.mean(resid_va <= pred_va))
            self.coverage[tau] = float(np.mean(resid_va <= pred_va + self.shift[tau]))
        self.val_resid = resid_va

    def set_level(self, q: float):
        tau = min(self.taus, key=lambda t: abs(t - q))
        assert abs(tau - q) < 1e-9, f"τ={q} kantil ızgarasında yok: {self.taus}"
        self.level = tau
        self.margin = float("nan")      # sabit pay yok; margin_now kullanılır

    def _precompute(self, load_hist, wl_hist):
        """Tüm tick'ler için τ başına payı toplu hesaplar. Satır k yalnızca x[≤k]
        kullandığından (bkz. _state_feats) bu bir gelecek sızıntısı DEĞİLDİR; sadece hız."""
        n = len(load_hist)
        F = np.array([self._feat(load_hist, wl_hist, k) for k in range(3, n)])
        self._cache = {}
        for tau, m in self.qmodels.items():
            arr = np.full(n, max(0.0, float(np.quantile(self.val_resid, tau))))
            arr[3:] = np.maximum(0.0, m.predict(F) + self.shift[tau])
            self._cache[tau] = arr
        self._cache_src = (load_hist, wl_hist)

    def margin_now(self, load_hist, wl_hist, k) -> float:
        src = getattr(self, "_cache_src", (None, None))
        if src[0] is not load_hist or src[1] is not wl_hist:
            self._precompute(load_hist, wl_hist)
        return float(self._cache[self.level][k])


HGB_PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                  random_state=0)
COND_TAUS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]


def train_forecasters(train_seed: int, train_days: float, use_lstm: bool,
                      n_servers: int, use_rf: bool = True) -> List[Forecaster]:
    """Sentetik eğitim izi üretip tahmincileri eğitir (bkz. train_forecasters_on)."""
    tr = generate_trace(train_seed, train_days, n_servers)
    return train_forecasters_on(tr["load"], tr["wl"], use_lstm, use_rf)


def trace_from_series(load: np.ndarray, wl: Optional[np.ndarray], n_servers: int,
                      failure_seed: int) -> Dict:
    """Dış bir talep dizisinden (ör. Azure) simülasyon izi kurar. Arıza takvimi
    generate_trace ile aynı süreçle, verilen seed'den üretilir."""
    load = np.asarray(load, dtype=float)
    n = len(load)
    wl = np.zeros(n, dtype=int) if wl is None else np.asarray(wl, dtype=int)
    rng = np.random.default_rng(failure_seed + 7919)
    failed = np.zeros((n, n_servers), dtype=bool)
    for i in range(n_servers):
        k = 0
        while k < n:
            if rng.random() < FAIL_PROB_PER_TICK:
                dur = int(rng.integers(FAIL_MIN_TICKS, FAIL_MAX_TICKS + 1))
                failed[k:k + dur, i] = True
                k += dur
            else:
                k += 1
    return {"load": load, "wl": wl, "failed": failed, "n": n}


def train_forecasters_on(load: np.ndarray, wl: np.ndarray, use_lstm: bool,
                         use_rf: bool = True, verbose: bool = True) -> List[Forecaster]:
    """Verilen eğitim dizisinde (zaman sıralı %0-80 eğitim / %80-100 doğrulama)
    tüm tahmincileri eğitir ve kalibre eder."""
    out: List[Forecaster] = []
    _print = print if verbose else (lambda *a, **k: None)

    # Naive (son değer) — referans tahminci
    X, y, _ = build_tabular(load, wl)
    cut = int(len(X) * 0.8)

    f_naive = Forecaster("naive", None, "naive")
    f_naive.calibrate(y[cut:], X[cut:, 0])
    out.append(f_naive)
    if use_rf:
        t0 = time.time()
        rf = RandomForestRegressor(n_estimators=40, max_depth=10, max_features=8, rng_seed=1)
        rf.fit(X[:cut], y[:cut])
        f_rf = Forecaster("rf", rf, "rf")
        f_rf.calibrate(y[cut:], rf.predict(X[cut:]))
        _print(f"  RF   eğitildi ({time.time()-t0:.0f}s) | val MAE={f_rf.val_metrics['mae']:.4f} "
              f"R²={f_rf.val_metrics['r2']:.3f} | naive MAE={f_naive.val_metrics['mae']:.4f} "
              f"| pay(q90)={f_rf.margin:.4f}")
        out.append(f_rf)

    # G3-1: doğrusal AR / ridge nokta tahminci
    t0 = time.time()
    lin = LinearAR().fit(X[:cut], y[:cut])
    f_lin = Forecaster("lin", lin, "tab")
    f_lin.calibrate(y[cut:], lin.predict(X[cut:]))
    out.append(f_lin)
    _print(f"  LIN  eğitildi ({time.time()-t0:.1f}s) | val MAE={f_lin.val_metrics['mae']:.4f} "
          f"R²={f_lin.val_metrics['r2']:.3f} | pay(q90)={f_lin.margin:.4f}")

    # G3-2: LIN + duruma koşullu kantil payı
    t0 = time.time()
    _, _, idx = build_tabular(load, wl)
    idx = np.array(idx)
    ar2 = fit_ar2(load[:idx[cut]])              # yalnızca eğitim bölümü
    resid = y - lin.predict(X)
    f_q = ConditionalMarginForecaster("linq", f_lin, ar2, COND_TAUS)
    c70 = int(len(X) * 0.7)                    # kantil eğitimi %0-70, CQR %70-80, doğrulama %80-100
    f_q.fit_margins(load, wl, idx[:c70], resid[:c70], idx[c70:cut], resid[c70:cut],
                    idx[cut:], resid[cut:])
    f_q.val_metrics = dict(f_lin.val_metrics, ar2=list(ar2),
                           coverage={str(t): c for t, c in f_q.coverage.items()},
                           coverage_raw={str(t): c for t, c in f_q.coverage_raw.items()},
                           cqr_shift={str(t): v for t, v in f_q.shift.items()})
    f_q.set_level(0.90)
    out.append(f_q)
    cov = " ".join(f"{t:g}:{f_q.coverage_raw[t]:.3f}→{c:.3f}" for t, c in f_q.coverage.items())
    _print(f"  LINQ eğitildi ({time.time()-t0:.0f}s) | AR(2)=({ar2[0]:.3f},{ar2[1]:.3f}) "
          f"| val kapsama τ→oran: {cov}")

    for f in out:
        if f.kind != "tab" or f.name == "lin":
            f.set_level(0.90)

    if use_lstm:
        seq_len = 12
        Xs, ys = build_sequences(load, wl, seq_len)
        cut_s = int(len(Xs) * 0.8)
        t0 = time.time()
        lstm = LSTMModel(input_size=Xs.shape[2], hidden_size=24, output_size=1, rng_seed=1)
        lstm._adam_lr = 0.003
        # numpy BPTT yavaş: eğitim kümesinden alt örnekleme
        rng = np.random.default_rng(0)
        sub = rng.choice(cut_s, size=min(3000, cut_s), replace=False)
        lstm.fit(Xs[sub], ys[sub], epochs=8, batch_size=32, verbose=False)
        f_lstm = Forecaster("lstm", lstm, "lstm", seq_len)
        f_lstm.calibrate(ys[cut_s:], np.ravel(lstm.predict(Xs[cut_s:])))
        _print(f"  LSTM eğitildi ({time.time()-t0:.0f}s) | val MAE={f_lstm.val_metrics['mae']:.4f} "
              f"R²={f_lstm.val_metrics['r2']:.3f} | pay(q90)={f_lstm.margin:.4f}")
        out.append(f_lstm)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. Sunucu kümesi seçimi
# ══════════════════════════════════════════════════════════════════════════════

def select_greedy(available, required_rps: float) -> List:
    """HPA benzeri: verimlilik (rps/W) sırasıyla, kapasite ≥ talep/hedef olana dek ekle."""
    need = required_rps / TARGET_UTIL
    ranked = sorted(available, key=lambda s: s.max_requests_per_sec / s.tdp_watts,
                    reverse=True)
    chosen, cap = [], 0.0
    for s in ranked:
        if cap >= need and len(chosen) >= MIN_ACTIVE:
            break
        chosen.append(s)
        cap += s.max_requests_per_sec
    return chosen


def select_optimal(available, required_rps: float, dominant_wl: Optional[str]) -> List:
    """Projenin AutoscalerEngine._find_optimal_combo() fonksiyonunu kullanır."""
    if len(available) <= MIN_ACTIVE:
        return list(available)
    eng = AutoscalerEngine(fleet=list(available), min_active_servers=MIN_ACTIVE)
    return eng._find_optimal_combo(required_rps, predictions=None,
                                   dominant_workload=dominant_wl)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Simülasyon
# ══════════════════════════════════════════════════════════════════════════════

def _wpct(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Ağırlıklı yüzdelik (her tick'in gecikmesi, o tick'te hizmet verilen istek sayısıyla)."""
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cw, q / 100.0 * cw[-1])])


def simulate(policy: str, trace: Dict, forecaster: Optional[Forecaster] = None,
             record: bool = False) -> Dict:
    fleet = create_server_fleet()
    ids   = [s.server_id for s in fleet]
    cap_total = sum(s.max_requests_per_sec for s in fleet)
    load, wl, failed, n = trace["load"], trace["wl"], trace["failed"], trace["n"]
    demand = load * DEMAND_SCALE * cap_total

    powered: Dict[str, int] = {sid: 0 for sid in ids}   # sid → kalan açılış tick'i
    recent_targets: deque = deque(maxlen=SCALE_IN_STAB_TICKS)
    m = {"rho": [], "dropped": 0.0, "demand": 0.0, "served": 0.0, "energy_kwh": 0.0, "cost_usd": 0.0,
         "active": [], "lat": [], "lat_w": [], "sla_ticks": 0, "wakeups": 0, "sleeps": 0,
         "drop_ticks": 0}

    for k in range(n):
        # Arızalar: arızalı sunucu güç çekmez, hizmet vermez, kümeden düşer
        for i, sid in enumerate(ids):
            if failed[k, i] and sid in powered:
                del powered[sid]

        serving = [s for s in fleet if s.server_id in powered and powered[s.server_id] == 0]
        booting = [s for s in fleet if s.server_id in powered and powered[s.server_id] > 0]
        cap = sum(s.max_requests_per_sec for s in serving)
        d   = demand[k]
        rho = d / cap if cap > 0 else float("inf")

        served = min(d, cap)
        m["dropped"] += d - served
        m["served"]  += served
        m["demand"]  += d
        m["rho"].append(min(rho, 5.0))
        if rho > SLA_UTIL:
            m["sla_ticks"] += 1
        if rho > 1.0:
            m["drop_ticks"] += 1

        util_pct = min(rho, 1.0) * 100.0
        watts = 0.0
        for s in serving:
            s.current_cpu_pct, s.current_temp_c = util_pct, 40.0
            watts += s.compute_power_draw()
        for s in booting:
            watts += s.idle_power_watts
        m["energy_kwh"] += watts * DT_SECONDS / 3600.0 / 1000.0
        m["cost_usd"]   += sum(s.cost_per_hour_usd for s in serving + booting) * DT_SECONDS / 3600.0
        m["active"].append(len(serving) + len(booting))
        if serving and served > 0:
            # Hizmet verilen isteklerin gecikmesi (aşırı yükte ρ=1 kuyruk faktörü);
            # düşen istekler ayrıca dropped_req_pct ile raporlanır.
            lat = (sum(s.latency_base_ms * s.max_requests_per_sec for s in serving) / cap
                   / max(0.02, 1.0 - min(rho, 1.0) * 0.95))
            m["lat"].append(lat)
            m["lat_w"].append(served)

        # Açılış sayaçlarını ilerlet
        for sid in list(powered):
            if powered[sid] > 0:
                powered[sid] -= 1

        # ── Karar (t anında bilinenlerle, t+1 için) ──────────────────────
        avail = [s for i, s in enumerate(fleet) if not failed[k, i]]   # yalnızca t anı bilgisi
        dom_wl = WORKLOAD_CLASSES[wl[k]]
        if policy == "all_on":
            target = avail
        else:
            source, selector = policy.rsplit("_", 1)      # ör. "rf_greedy"
            if source == "observed":
                req = d
            elif source == "oracle":
                fut = demand[k + 1:k + 1 + HORIZON]
                req = float(fut.max()) if len(fut) else d
            else:
                x_hat = (forecaster.predict_at(load, wl, k)
                         + forecaster.margin_now(load, wl, k))
                req = max(x_hat, 0.0) * DEMAND_SCALE * cap_total
            target = (select_greedy(avail, req) if selector == "greedy"
                      else select_optimal(avail, req, dom_wl))

        target_ids = {s.server_id for s in target}
        cur_ids    = set(powered)
        to_add     = target_ids - cur_ids
        to_remove  = cur_ids - target_ids

        for sid in to_add:
            powered[sid] = BOOT_TICKS
            m["wakeups"] += 1

        # Küçültme (HPA tarzı): son SCALE_IN_STAB_TICKS hedefin BİRLEŞİMİNDE olan
        # sunucu kapatılmaz; ayrıca hedefteki yeni sunucular açılana dek kapatma yok
        # (make-before-break). all_on dahil tüm politikalara aynı kural uygulanır.
        recent_targets.append(target_ids)
        keep = set().union(*recent_targets)
        still_booting = any(powered.get(sid, 0) > 0 for sid in target_ids)
        if to_remove and not to_add and not still_booting:
            for sid in to_remove - keep:
                del powered[sid]
                m["sleeps"] += 1

    lat, lw = np.array(m["lat"]), np.array(m["lat_w"])
    extra = {"rho_series": list(m["rho"]), "active_series": list(m["active"])} if record else {}
    hours = n * DT_SECONDS / 3600.0
    return {
        "avg_active_servers": float(np.mean(m["active"])),
        "energy_kwh":         m["energy_kwh"],
        "avg_power_w":        m["energy_kwh"] * 1000.0 / hours,
        "cost_usd":           m["cost_usd"],
        "sla_violation_pct":  100.0 * m["sla_ticks"] / n,
        "overload_tick_pct":  100.0 * m["drop_ticks"] / n,
        "dropped_req_pct":    100.0 * m["dropped"] / max(m["demand"], 1.0),
        "avg_util_pct":       100.0 * float(np.mean(np.clip(m["rho"], 0, 1))),
        "p95_latency_ms":     _wpct(lat, lw, 95),    # istek ağırlıklı
        "p99_latency_ms":     _wpct(lat, lw, 99),
        "demand_total":       m["demand"],
        "served_total":       m["served"],
        "dropped_total":      m["dropped"],
        "wakeups":            m["wakeups"],
        "sleeps":             m["sleeps"],
        **extra,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Ana akış
# ══════════════════════════════════════════════════════════════════════════════

KEYS = [("avg_active_servers", "Ort. aktif sunucu", "{:.2f}"),
        ("avg_power_w",        "Ort. güç (W)",      "{:.0f}"),
        ("energy_kwh",         "Enerji (kWh)",      "{:.1f}"),
        ("cost_usd",           "Maliyet ($)",       "{:.1f}"),
        ("avg_util_pct",       "Ort. kullanım (%)", "{:.1f}"),
        ("sla_violation_pct",  "SLA ihlali (% tick)", "{:.2f}"),
        ("overload_tick_pct",  "Aşırı yük (% tick)", "{:.2f}"),
        ("dropped_req_pct",    "Düşen istek (%)",   "{:.3f}"),
        ("p95_latency_ms",     "p95 gecikme (ms)",  "{:.2f}"),
        ("p99_latency_ms",     "p99 gecikme (ms)",  "{:.1f}"),
        ("wakeups",            "Uyandırma sayısı",  "{:.0f}")]


REF_REACTIVE = "observed_greedy"   # BKO alt referansı (HPA benzeri reaktif)
REF_ORACLE   = "oracle_greedy"     # BKO üst referansı (kusursuz tahmin, aynı seçici)
N_BOOT       = 2000


def _ci(samples: np.ndarray) -> List[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def summarize(results: Dict[str, List[Dict]], policies: List[str], seed: int = 0) -> Dict:
    """Seed ortalaması + %95 bootstrap GA. Bootstrap, seed'leri yeniden örnekler ve
    tüm politikalar için AYNI örneklemi kullanır (eşleştirilmiş karşılaştırma).

    BKO = (SLA_reaktif − SLA_politika) / (SLA_reaktif − SLA_oracle)
          1 → oracle kadar iyi, 0 → reaktif kadar, <0 → reaktiften kötü.
    """
    rng = np.random.default_rng(seed)
    n_seeds = len(results[policies[0]])
    arr = {p: {k: np.array([r[k] for r in results[p]]) for k, _, _ in KEYS} for p in policies}
    boot_idx = rng.integers(0, n_seeds, size=(N_BOOT, n_seeds))

    def mean_ci(x):
        return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "ci95": _ci(x[boot_idx].mean(axis=1))}

    sla_r = arr[REF_REACTIVE]["sla_violation_pct"]
    sla_o = arr[REF_ORACLE]["sla_violation_pct"]
    e_ref = arr[REF_REACTIVE]["energy_kwh"]

    out = {}
    for p in policies:
        out[p] = {k: mean_ci(arr[p][k]) for k, _, _ in KEYS}
        sla_p = arr[p]["sla_violation_pct"]

        def bko(idx_rows):
            r = sla_r[idx_rows].mean(axis=-1); o = sla_o[idx_rows].mean(axis=-1)
            q = sla_p[idx_rows].mean(axis=-1)
            return (r - q) / np.maximum(r - o, 1e-9)

        def de(idx_rows):
            return 100.0 * (arr[p]["energy_kwh"][idx_rows].mean(axis=-1)
                            / e_ref[idx_rows].mean(axis=-1) - 1.0)

        full = np.arange(n_seeds)
        out[p]["gap_closure"]      = {"mean": float(bko(full)), "ci95": _ci(bko(boot_idx))}
        out[p]["delta_energy_pct"] = {"mean": float(de(full)),  "ci95": _ci(de(boot_idx))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30, help="test seed sayısı")
    ap.add_argument("--days", type=float, default=7.0, help="her testin gün sayısı")
    ap.add_argument("--train-days", type=float, default=30.0)
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--out", type=str, default="logs/policy_comparison.json")
    args = ap.parse_args()

    n_srv = len(create_server_fleet())
    print("═" * 72)
    print("  Tahminciler eğitiliyor (eğitim seed=1000, test seed'lerinden bağımsız)")
    print("═" * 72)
    forecasters = train_forecasters(1000, args.train_days, not args.no_lstm, n_srv)
    sources = ["observed"] + [f.name for f in forecasters] + ["oracle"]
    fmap = {}
    policies = ["all_on"]
    for sel in ["greedy", "opt"]:
        for src in sources:
            pol = f"{src}_{sel}"
            policies.append(pol)
            fmap[pol] = next((f for f in forecasters if f.name == src), None)
    results: Dict[str, List[Dict]] = {p: [] for p in policies}
    test_seeds = [2001 + i for i in range(args.seeds)]

    print("═" * 72)
    print(f"  Politikalar: {len(test_seeds)} seed × {args.days:g} gün")
    print("═" * 72)
    for sd in test_seeds:
        tr = generate_trace(sd, args.days, n_srv)
        print(f"  seed={sd} | arıza olayı={int(np.sum(np.diff(tr['failed'].astype(int), axis=0) == 1))}")
        for p in policies:
            results[p].append(simulate(p, tr, fmap.get(p)))

    summary = summarize(results, policies)
    # Tablo (politikalar satır)
    short = [("avg_active_servers","Aktif"),("avg_power_w","Güç W"),("cost_usd","Maliyet$"),
             ("sla_violation_pct","SLA%"),("dropped_req_pct","Düşen%"),
             ("p99_latency_ms","p99 ms"),("wakeups","Uyandır")]
    print("\n" + "Politika".ljust(16) + "".join(l.rjust(10) for _, l in short) + "  ΔEnerji")
    print("-" * (16 + 10 * len(short) + 9))
    base = summary["observed_greedy"]["energy_kwh"]["mean"]
    for p in policies:
        row = p.ljust(16)
        for key, _ in short:
            v = summary[p][key]["mean"]
            row += (f"{v:.2f}" if key in ("avg_active_servers","sla_violation_pct")
                    else f"{v:.3f}" if key == "dropped_req_pct" else f"{v:.0f}").rjust(10)
        e = summary[p]["energy_kwh"]["mean"]
        row += f"{100*(e-base)/base:+8.1f}%"
        print(row)
    print("(ΔEnerji: observed_greedy = HPA benzeri reaktif baseline'a göre)")
    print("\nBoşluk Kapatma Oranı (BKO) ve ΔEnerji — %95 bootstrap güven aralığı:")
    for p in policies:
        g = summary[p]["gap_closure"]; e = summary[p]["delta_energy_pct"]
        print(f"  {p:<16} BKO={100*g['mean']:6.1f}% [{100*g['ci95'][0]:6.1f}, {100*g['ci95'][1]:6.1f}]"
              f" | ΔEnerji={e['mean']:+6.1f}% [{e['ci95'][0]:+6.1f}, {e['ci95'][1]:+6.1f}]")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"schema_version": 1,
                   "description": "evaluate_policies.py sonuçları; metrikler seed ortalaması, "
                                  "ci95 = seed üzerinde %95 bootstrap güven aralığı",
                   "config": {"seeds": test_seeds, "days": args.days,
                              "boot_ticks": BOOT_TICKS, "target_util": TARGET_UTIL,
                              "sla_util": SLA_UTIL, "demand_scale": DEMAND_SCALE,
                              "regime_mean_ticks": REGIME_MEAN_TICKS,
                              "scale_in_stab_ticks": SCALE_IN_STAB_TICKS,
                              "train_seed": 1000, "train_days": args.train_days,
                              "reference_reactive": REF_REACTIVE, "reference_oracle": REF_ORACLE,
                              "bootstrap_samples": N_BOOT},
                   "forecasters": {f.name: f.val_metrics for f in forecasters},
                   "summary": summary, "per_seed": results}, fh, indent=2, ensure_ascii=False)
    print(f"\nSonuçlar → {args.out}")


if __name__ == "__main__":
    main()
