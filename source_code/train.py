import numpy as np
import pandas as pd
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from core.server_profiles import create_server_fleet
from core.metrics_collector import MetricsCollector
from core.traffic_generator import (TrafficGenerator, TrafficPattern,
                                     WorkloadType, WORKLOAD_CLASSES,
                                     WORKLOAD_SERVER_AFFINITY)
from collections import deque, Counter

WL_TO_INT = {wl: i for i, wl in enumerate(WORKLOAD_CLASSES)}
INT_TO_WL = {i: wl for wl, i in WL_TO_INT.items()}
from core.autoscaler import AutoscalerEngine
from core.llm_workload_advisor import LLMWorkloadAdvisor
from models.models import (
    RandomForestRegressor, LSTMModel, GraphSAGEModel,
    make_sequences, train_test_split_time
)

# ──────────────────────────────────────────────────────────
# Yapılandırma
# ──────────────────────────────────────────────────────────
CFG = {
    # Veri
    "total_days":        30,
    "tick_seconds":      300,
    # RF
    "rf_n_estimators":   50,
    "rf_max_depth":      10,
    "rf_lag_steps":      12,
    "rf_weekly_lags":    [2016, 2017, 2018],
    # LSTM  ← v3: hidden küçültüldü, epoch azaltıldı
    "lstm_seq_len":      24,
    "lstm_hidden":       64,   # 96 → 64  (PyTorch daha verimli kullanır)
    "lstm_epochs":       20,   # 25 → 20
    "lstm_batch":        512,  # 256 → 512 (PyTorch gerçek batch paralelliği)
    # GraphSAGE
    "gsage_snapshots":   500,
    "gsage_hidden":      (48, 24),
    "gsage_epochs":      40,
    # Autoscaler test
    "autoscaler_hours":  24,
    # LLM advisor
    "llm_interval_ticks": 6,   # 6 tick = 30 dakika
    "llm_model": "gpt-4.1-mini",
    "llm_endpoint": "https://api.openai.com/v1/responses",
}


def print_banner(text: str):
    w = 68
    print(f"\n{'═'*w}\n  {text}\n{'═'*w}")


def build_llm_advisor(args) -> Optional[LLMWorkloadAdvisor]:
    if args is None or not getattr(args, "use_llm", False):
        return None

    advisor = LLMWorkloadAdvisor(
        api_key=getattr(args, "llm_api_key", None),
        model=getattr(args, "llm_model", CFG["llm_model"]),
        endpoint=getattr(args, "llm_endpoint", CFG["llm_endpoint"]),
        timeout_sec=getattr(args, "llm_timeout", 20.0),
    )

    if advisor.online:
        print(f"  ✓ LLM advisor aktif | model={advisor.model}")
    else:
        print("  [UYARI] --use-llm açık ama OPENAI_API_KEY yok; heuristic fallback kullanılacak.")
    return advisor


# ──────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, server_id: str,
                     lag_steps: int = 12,
                     weekly_lag_offsets: Optional[List[int]] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    if weekly_lag_offsets is None:
        weekly_lag_offsets = [2016]

    WEEK_BACK  = 3
    WEEK_FWD   = 6

    sdf = df[df["server_id"] == server_id].sort_values("timestamp").reset_index(drop=True)
    # Tüm offset'lerin gerektirdiği en büyük min_idx
    max_week_offset = max(weekly_lag_offsets)
    min_idx = max(lag_steps, max_week_offset + WEEK_BACK + 1)
    if len(sdf) < min_idx + WEEK_FWD + 2:
        return np.array([]), np.array([])

    metric_cols = ["cpu_pct", "memory_pct", "requests_norm",
                   "latency_ms", "power_watts", "temp_c"]
    time_cols   = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                   "wom_sin",  "wom_cos",  "is_weekday"]

    means = sdf[metric_cols].mean()
    stds  = sdf[metric_cols].std().replace(0, 1.0)
    norm  = (sdf[metric_cols] - means) / stds
    cpu_raw = sdf["cpu_pct"].values

    week_win = 288
    features, targets = [], []

    for i in range(min_idx, len(sdf) - 1):
        feat = []

        for lag in range(lag_steps, 0, -1):
            feat.extend(norm.iloc[i - lag].values)

        # Her haftalık offset için ayrı pencere + özet ekle
        for week_offset in weekly_lag_offsets:
            anchor = i - week_offset
            w_start = anchor - WEEK_BACK
            w_end   = anchor + WEEK_FWD + 1

            if w_start >= 0 and w_end <= len(sdf):
                week_window_data = norm.iloc[w_start:w_end].values
            elif w_start < 0:
                week_window_data = np.zeros((WEEK_BACK + WEEK_FWD + 1, len(metric_cols)))
            else:
                available = norm.iloc[max(0, w_start):len(sdf)].values
                pad = WEEK_BACK + WEEK_FWD + 1 - len(available)
                week_window_data = np.vstack([available, np.zeros((pad, len(metric_cols)))])

            feat.extend(week_window_data.flatten().tolist())

            summ_start = anchor - week_win
            summ_end   = anchor + 1
            if summ_start >= 0:
                seg = cpu_raw[summ_start:summ_end]
                feat.extend([float(np.mean(seg)) / 100.0,
                             float(np.std(seg))  / 100.0,
                             float(np.max(seg))  / 100.0])
            else:
                feat.extend([float(np.mean(cpu_raw[:max(1,i)])) / 100.0, 0.0,
                             float(np.max(cpu_raw[:max(1,i)]))  / 100.0])

        for tc in time_cols:
            feat.append(float(sdf[tc].iloc[i]) if tc in sdf.columns else 0.0)

        features.append(feat)
        targets.append(float(sdf["cpu_pct"].iloc[i + 1]))

    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32)


def prepare_lstm_sequences(df: pd.DataFrame, server_ids: List[str],
                            seq_len: int = 24,
                            weekly_lag_offsets: Optional[List[int]] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paylaşımlı LSTM giriş dizileri — VEKTÖRİZE edilmiş versiyon.

    v2'deki iç içe Python döngüsü (her adım için ayrı slice ve concatenate)
    yerine her sunucu için tüm işlemler toplu NumPy operasyonlarıyla yapılır.

    Çıkış formatı aynı: (N, seq_len, features)
    """
    if weekly_lag_offsets is None:
        weekly_lag_offsets = [2016]

    WEEK_BACK     = 3
    WEEK_FWD      = 6
    WEEK_WIN_SIZE = WEEK_BACK + WEEK_FWD + 1   # 10
    week_offset   = weekly_lag_offsets[0]
    week_sum_win  = 288

    metric_cols = ["cpu_pct", "memory_pct", "requests_norm",
                   "latency_ms", "power_watts", "temp_c"]
    time_cols   = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                   "wom_sin",  "wom_cos",  "is_weekday"]
    n_srv       = len(server_ids)
    n_metrics   = len(metric_cols)
    n_time      = len(time_cols)

    all_X, all_y = [], []

    for srv_idx, srv_id in enumerate(server_ids):
        sdf = df[df["server_id"] == srv_id].sort_values("timestamp").reset_index(drop=True)
        min_idx = max(seq_len, week_offset + WEEK_BACK + 1)
        if len(sdf) < min_idx + WEEK_FWD + 2:
            continue

        T = len(sdf)

        # ── Normalize ──────────────────────────────────────────────
        means = sdf[metric_cols].mean()
        stds  = sdf[metric_cols].std().replace(0, 1.0)
        norm  = ((sdf[metric_cols] - means) / stds).values.astype(np.float32)  # (T, 6)

        has_time = all(c in sdf.columns for c in time_cols)
        time_arr = sdf[time_cols].values.astype(np.float32) if has_time \
                   else np.zeros((T, n_time), dtype=np.float32)               # (T, 7)

        cpu_raw = sdf["cpu_pct"].values.astype(np.float32)

        # Geçerli zaman damgası aralığı
        valid_start = min_idx
        valid_end   = T - 1          # i+1 < T olmalı
        n_valid     = valid_end - valid_start

        if n_valid <= 0:
            continue

        indices = np.arange(valid_start, valid_end)  # (n_valid,)

        # ── A) Ana dizi  (n_valid, seq_len, 13) ───────────────────
        # Her i için norm[i-seq_len:i] ve time_arr[i-seq_len:i]
        # Sliding window → stride_tricks ile sıfır kopyalama
        seq_idx = indices[:, None] + np.arange(-seq_len, 0)[None, :]  # (n_valid, seq_len)
        seq_metrics = norm[seq_idx]       # (n_valid, seq_len, 6)
        seq_time    = time_arr[seq_idx]   # (n_valid, seq_len, 7)
        seq_base    = np.concatenate([seq_metrics, seq_time], axis=2)  # (n_valid, seq_len, 13)

        # ── B) Haftalık pencere dizi  (n_valid, seq_len, 60) ──────
        # Her i, her j (seq adımı) için: anchor = (i - seq_len + j) - week_offset
        # Tüm (i, j) çiftleri için anchor matris olarak hesaplanır.
        step_offsets = np.arange(-seq_len, 0)                           # (seq_len,)
        t_j_mat      = indices[:, None] + step_offsets[None, :]         # (n_valid, seq_len)
        anchor_mat   = t_j_mat - week_offset                            # (n_valid, seq_len)

        # Pencere başlangıç ve bitiş indeksleri
        ws_mat = anchor_mat - WEEK_BACK   # (n_valid, seq_len)
        we_mat = anchor_mat + WEEK_FWD    # dahil, yani [ws, we] → we+1 hariç

        # Sınır içi maske
        valid_mask = (ws_mat >= 0) & (we_mat < T)

        # Tüm (i,j) çiftleri için pencere dizini matrisini oluştur
        # win_idx: (n_valid, seq_len, WEEK_WIN_SIZE)
        win_range  = np.arange(WEEK_WIN_SIZE)                              # (10,)
        win_idx    = ws_mat[:, :, None] + win_range[None, None, :]         # (n_valid, seq_len, 10)

        # Sınır dışı indeksleri 0'a klamp (sıfır ile doldurulacak zaten)
        win_idx_clamped = np.clip(win_idx, 0, T - 1)

        # norm'dan toplu al → (n_valid, seq_len, 10, 6)
        week_raw = norm[win_idx_clamped]

        # Geçersiz konumları sıfırla
        week_raw[~valid_mask] = 0.0

        # (n_valid, seq_len, 60)
        week_seq = week_raw.reshape(n_valid, seq_len, WEEK_WIN_SIZE * n_metrics)

        # ── C) Haftalık özet  (n_valid, seq_len, 3) ───────────────
        # Her i için anchor_last = (i-1) - week_offset
        # Özet pencere: [anchor_last - week_sum_win, anchor_last]
        anchor_last = indices - 1 - week_offset       # (n_valid,)
        ss_arr      = anchor_last - week_sum_win       # (n_valid,)
        se_arr      = anchor_last + 1                  # (n_valid,)

        weekly_ctx = np.zeros((n_valid, 3), dtype=np.float32)
        for idx_i, (ss, se, anchor) in enumerate(zip(ss_arr, se_arr, anchor_last)):
            if ss >= 0 and se <= T:
                seg = cpu_raw[ss:se]
                weekly_ctx[idx_i] = [np.mean(seg)/100.0,
                                     np.std(seg)/100.0,
                                     np.max(seg)/100.0]
            else:
                i_abs = indices[idx_i]
                seg = cpu_raw[:max(1, i_abs)]
                weekly_ctx[idx_i] = [np.mean(seg)/100.0, 0.0, np.max(seg)/100.0]

        # (n_valid, seq_len, 3) — her adıma yay
        weekly_ctx_seq = np.broadcast_to(
            weekly_ctx[:, None, :], (n_valid, seq_len, 3)
        ).copy()

        # ── D) One-hot sunucu kimliği  (n_valid, seq_len, n_srv) ──
        one_hot = np.zeros((n_valid, seq_len, n_srv), dtype=np.float32)
        one_hot[:, :, srv_idx] = 1.0

        # ── Birleştir: (n_valid, seq_len, 13+60+3+n_srv) ──────────
        X_srv = np.concatenate([seq_base, week_seq, weekly_ctx_seq, one_hot], axis=2)
        y_srv = cpu_raw[indices + 1]   # bir sonraki adımın cpu'su

        all_X.append(X_srv)
        all_y.append(y_srv)

    if not all_X:
        return np.array([]), np.array([])

    return (np.concatenate(all_X, axis=0).astype(np.float32),
            np.concatenate(all_y, axis=0).astype(np.float32))


def prepare_graphsage_snapshots(df: pd.DataFrame, n_snapshots: int = 500
                                 ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """GraphSAGE snapshot'ları — değişiklik yok."""
    base_cols  = ["cpu_pct", "memory_pct", "requests_norm",
                  "latency_ms", "power_watts", "temp_c"]
    server_ids = sorted(df["server_id"].unique())
    timestamps = sorted(df["timestamp"].unique())
    tick_sec   = float(timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 300.0
    week_ticks = int(7 * 86400 / tick_sec)

    step        = max(1, (len(timestamps) - week_ticks) // n_snapshots)
    selected_ts = timestamps[week_ticks::step][:n_snapshots]

    X_snapshots, y_snapshots = [], []
    for i, ts in enumerate(selected_ts[:-1]):
        ts_next    = selected_ts[i + 1]
        pivot_curr = df[df["timestamp"] == ts].set_index("server_id")[base_cols]
        pivot_next = df[df["timestamp"] == ts_next].set_index("server_id")[["cpu_pct"]]
        common     = [s for s in server_ids if s in pivot_curr.index and s in pivot_next.index]
        if len(common) < 2:
            continue

        X_now = pivot_curr.loc[common].values.astype(float)

        ts_week_ago_start = ts - 7 * 86400 - 86400
        ts_week_ago_end   = ts - 7 * 86400
        mask_week         = (df["timestamp"] >= ts_week_ago_start) & \
                            (df["timestamp"] <= ts_week_ago_end)
        week_df           = df[mask_week]
        weekly_avg_cpu    = (week_df.groupby("server_id")["cpu_pct"].mean()
                             if len(week_df) > 0 else pd.Series(dtype=float))
        weekly_feat       = np.array([
            weekly_avg_cpu.get(s, float(X_now[j, 0])) / 100.0
            for j, s in enumerate(common)
        ]).reshape(-1, 1)

        X_aug = np.concatenate([X_now, weekly_feat], axis=1)
        X_aug = (X_aug - X_aug.mean(0)) / (X_aug.std(0) + 1e-8)

        y = pivot_next.loc[common]["cpu_pct"].values.astype(float)
        X_snapshots.append(X_aug)
        y_snapshots.append(y)

    return X_snapshots, y_snapshots


# ──────────────────────────────────────────────────────────
# Değerlendirme
# ──────────────────────────────────────────────────────────

def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2   = float(1.0 - np.sum((y_true - y_pred) ** 2) /
                 (np.sum((y_true - y_true.mean()) ** 2) + 1e-10))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-5))) * 100)
    print(f"  {name:<22} | MAE={mae:.3f} | RMSE={rmse:.3f} | R²={r2:.4f} | MAPE={mape:.1f}%")
    return {"model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "R2": round(r2, 4), "MAPE_pct": round(mape, 2)}


# ──────────────────────────────────────────────────────────
# Autoscaler için canlı özellik çıkarma yardımcıları
# ──────────────────────────────────────────────────────────

def _time_feats_live(t: float):
    """Autoscaler döngüsünde anlık zaman özellikleri (7 boyut, prepare_features ile aynı sıra)."""
    hour = (t % 86400) / 3600.0
    dow  = int(t // 86400) % 7
    wom  = (int(t // 86400) // 7) % 4
    return [
        np.sin(2 * np.pi * hour / 24),   # hour_sin
        np.cos(2 * np.pi * hour / 24),   # hour_cos
        np.sin(2 * np.pi * dow  /  7),   # dow_sin
        np.cos(2 * np.pi * dow  /  7),   # dow_cos
        np.sin(2 * np.pi * wom  /  4),   # wom_sin
        np.cos(2 * np.pi * wom  /  4),   # wom_cos
        float(dow < 5),                  # is_weekday
    ]


def _build_rf_feat_live(buf: list, t: float, lag_steps: int = 12,
                        weekly_lag_offsets: Optional[List[int]] = None,
                        wl_onehot: Optional[list] = None) -> np.ndarray:
    """
    Rolling buffer'dan RF özellik vektörü oluştur.

    Eğitimdeki RF feature şemasıyla birebir uyumlu olacak şekilde
    haftalık blokları (offset başına 60 + 3) sıfır dolgu ile üretir.
    """
    if weekly_lag_offsets is None:
        weekly_lag_offsets = [2016, 2017, 2018]

    n_metrics   = 6
    weekly_dim_per_offset = (3 + 6 + 1) * n_metrics + 3  # 63
    feat: list  = []
    for lag in range(lag_steps, 0, -1):
        if lag <= len(buf):
            feat.extend(buf[-lag])
        else:
            feat.extend([0.0] * n_metrics)
    # Eğitimde her haftalık offset için ayrı blok var (rf_weekly_lags).
    for _ in weekly_lag_offsets:
        feat.extend([0.0] * weekly_dim_per_offset)
    feat.extend(_time_feats_live(t))
    return np.array(feat, dtype=np.float32).reshape(1, -1)


def _build_lstm_seq_live(buf: list, t: float,
                          seq_len: int, n_servers: int, srv_idx: int,
                          wl_onehot: Optional[list] = None) -> np.ndarray:
    """
    Rolling buffer'dan LSTM dizisi oluştur — şekil (1, seq_len, feat_dim).
    feat_dim = 13 (metrics+time) + 63 (haftalık) + n_servers
    prepare_lstm_sequences ile aynı imza.
    """
    n_metrics  = 6
    n_time     = 7
    weekly_dim = 60 + 3
    feat_dim   = n_metrics + n_time + weekly_dim + n_servers

    seq = np.zeros((seq_len, feat_dim), dtype=np.float32)
    tf  = _time_feats_live(t)

    for j in range(seq_len):
        lag = seq_len - j
        if lag <= len(buf):
            metrics = buf[-lag]
        else:
            metrics = [0.0] * n_metrics
        base = metrics + tf
        seq[j, :13]                       = base
        seq[j, 13 + weekly_dim + srv_idx] = 1.0

    return seq[np.newaxis, :, :]


def run_autoscaler_evaluation(fleet_factory,
                               rf_model_per_srv: dict,
                               rf_wl_per_srv: dict,
                               shared_lstm,
                               total_secs: int = 3600 * 24,
                               rf_lag_steps: int = 12,
                               lstm_seq_len: int = 24,
                               llm_advisor: Optional[LLMWorkloadAdvisor] = None,
                               llm_interval_ticks: int = 6):
    """Gerçekçi autoscaler testi — 24 saatlik bir gün boyunca.

    rf_model_per_srv : {server_id → RandomForestRegressor}  (Phase 2'den)
    shared_lstm      : LSTMModel                             (Phase 3'ten)
    Her tick'te eğitilmiş modeller gerçek predict() çağrısıyla tahmin üretir.
    """
    print_banner("Autoscaler Simülasyonu — 24 Saatlik Konsolidasyon Testi")

    autoscaler_rf   = AutoscalerEngine(fleet=fleet_factory(), min_active_servers=2)
    autoscaler_lstm = AutoscalerEngine(fleet=fleet_factory(), min_active_servers=2)

    tgen    = TrafficGenerator(seed=99)
    dt      = 300.0
    t       = 0.0
    n_ticks = int(total_secs / dt)

    fleet_rf   = autoscaler_rf.fleet
    fleet_lstm = autoscaler_lstm.fleet

    server_ids  = [s.server_id for s in fleet_rf]
    n_servers   = len(server_ids)
    srv_idx_map = {sid: i for i, sid in enumerate(server_ids)}

    # Per-server rolling buffers (normalize ham değerler yaklaşık z-score için)
    # Buffer: [cpu_norm, mem_norm, req_norm, lat_norm, pwr_norm, tmp_norm]
    from collections import deque
    METRICS_APPROX_MEAN = np.array([50.0, 40.0, 0.5,  5.0, 150.0, 60.0])
    METRICS_APPROX_STD  = np.array([25.0, 20.0, 0.3,  8.0,  80.0, 15.0])

    def _normalize(srv):
        raw = np.array([srv.current_cpu_pct, srv.current_memory_pct,
                        srv.current_requests / max(srv.max_requests_per_sec, 1),
                        srv.compute_latency(), srv.compute_power_draw(),
                        srv.current_temp_c], dtype=np.float32)
        return ((raw - METRICS_APPROX_MEAN) / METRICS_APPROX_STD).tolist()

    buf_rf   = {s.server_id: deque(maxlen=max(rf_lag_steps, lstm_seq_len) + 5)
                for s in fleet_rf}
    buf_lstm = {s.server_id: deque(maxlen=max(rf_lag_steps, lstm_seq_len) + 5)
                for s in fleet_lstm}

    rf_active_log, lstm_active_log = [], []
    rf_power_log,  lstm_power_log  = [], []
    last_llm_wl: Optional[str] = None

    for tick in range(n_ticks):
        pattern   = tgen.select_pattern_for_tick(t)
        base_load = tgen.get_load_for_pattern(pattern, t)
        noise     = np.random.randn() * 3

        # ── Anlık workload one-hot ─────────────────────────────────────────
        # Eğitimde en güçlü feature — deploy'da sıfır bırakmak tutarsızlık yaratır.
        cur_wl      = tgen.select_workload_for_tick(t)
        cur_wl_idx  = WL_TO_INT.get(cur_wl.value, 2)
        cur_wl_oh   = [0.0] * len(WORKLOAD_CLASSES)
        cur_wl_oh[cur_wl_idx] = 1.0

        # ── LLM dominant workload danışmanı (opsiyonel) ─────────────────────
        if llm_advisor is not None and tick % max(1, llm_interval_ticks) == 0:
            active_now = [s for s in fleet_rf if s.is_active]
            ctx = {
                "hour_of_day": round((t % 86400) / 3600.0, 2),
                "pattern": pattern.value,
                "base_load": round(float(base_load), 4),
                "workload_hint": cur_wl.value,
                "active_servers": len(active_now),
                "avg_cpu_active": round(float(np.mean([s.current_cpu_pct for s in active_now])) if active_now else 0.0, 3),
                "avg_mem_active": round(float(np.mean([s.current_memory_pct for s in active_now])) if active_now else 0.0, 3),
                "avg_latency_ms": round(float(np.mean([s.compute_latency() for s in active_now])) if active_now else 0.0, 3),
            }
            llm_pick = llm_advisor.suggest_workload(ctx)
            if llm_pick.get("workload"):
                last_llm_wl = llm_pick["workload"]
                conf_txt = f", conf={llm_pick.get('confidence')}" if llm_pick.get("confidence") else ""
                print(f"  t={t/3600:.1f}h | [LLM] dominant_workload={last_llm_wl}{conf_txt}")

        # ── RF tahminleri ─────────────────────────────────────────────
        rf_preds: dict = {}
        rf_wl_votes: list = []
        for srv in fleet_rf:
            if not srv.is_active:
                continue
            buf  = list(buf_rf[srv.server_id])
            rf_m = rf_model_per_srv.get(srv.server_id)
            rf_wl = rf_wl_per_srv.get(srv.server_id) if rf_wl_per_srv else None
            if rf_m is not None and len(buf) >= rf_lag_steps:
                feat    = _build_rf_feat_live(buf, t,
                                              lag_steps=rf_lag_steps,
                                              weekly_lag_offsets=CFG["rf_weekly_lags"],
                                              wl_onehot=cur_wl_oh)
                raw_cpu = float(np.clip(rf_m.predict(feat)[0], 0.0, 100.0))
                if rf_wl is not None:
                    rf_wl_votes.append(int(rf_wl.predict(feat)[0]))
            else:
                raw_cpu = float(np.clip(base_load * 100 + np.random.randn() * 5, 0.0, 100.0))
            rf_preds[srv.server_id] = raw_cpu
        rf_dominant_wl = (INT_TO_WL[Counter(rf_wl_votes).most_common(1)[0][0]]
                          if rf_wl_votes else cur_wl.value)

        # ── LSTM tahminleri ───────────────────────────────────────────
        lstm_preds: dict = {}
        lstm_wl_votes: list = []
        for srv in fleet_lstm:
            if not srv.is_active:
                continue
            buf  = list(buf_lstm[srv.server_id])
            sidx = srv_idx_map[srv.server_id]
            if shared_lstm is not None and len(buf) >= lstm_seq_len:
                seq = _build_lstm_seq_live(buf, t, lstm_seq_len, n_servers, sidx,
                                           wl_onehot=cur_wl_oh)
                if hasattr(shared_lstm, "predict_cpu"):
                    raw_cpu = float(np.clip(shared_lstm.predict_cpu(seq)[0], 0.0, 100.0))
                    if hasattr(shared_lstm, "predict_workload"):
                        lstm_wl_votes.append(int(shared_lstm.predict_workload(seq)[0]))
                else:
                    # Tek çıktılı LSTMModel ile geriye dönük uyumluluk.
                    raw_cpu = float(np.clip(shared_lstm.predict(seq)[0], 0.0, 100.0))
            else:
                raw_cpu = float(np.clip(base_load * 100 + np.random.randn() * 2.5, 0.0, 100.0))
            lstm_preds[srv.server_id] = raw_cpu
        lstm_dominant_wl = (INT_TO_WL[Counter(lstm_wl_votes).most_common(1)[0][0]]
                            if lstm_wl_votes else cur_wl.value)

        # ── Filo durumunu güncelle ve buffer'a ekle ───────────────────
        for fleet_obj, preds, autoscaler, buffers in [
            (fleet_rf,   rf_preds,   autoscaler_rf,   buf_rf),
            (fleet_lstm, lstm_preds, autoscaler_lstm, buf_lstm),
        ]:
            for srv in fleet_obj:
                if not srv.is_active:
                    continue
                load = preds.get(srv.server_id, 50.0)
                srv.current_cpu_pct  = float(np.clip(load + noise, 0, 100))
                srv.current_requests = srv.max_requests_per_sec * (srv.current_cpu_pct / 100)
                srv.update_thermal(dt)
                buffers[srv.server_id].append(_normalize(srv))
            model_dom_wl = rf_dominant_wl if autoscaler is autoscaler_rf else lstm_dominant_wl
            dom_wl = last_llm_wl if last_llm_wl is not None else model_dom_wl
            decision = autoscaler.decide(preds, t, dominant_workload=dom_wl)
            if decision.action != "hold":
                print(f"  t={t/3600:.1f}h | [{decision.action.upper()}] {decision.reason[:75]}")

        rf_active_log.append(sum(1 for s in fleet_rf   if s.is_active))
        lstm_active_log.append(sum(1 for s in fleet_lstm if s.is_active))
        rf_power_log.append(sum(s.compute_power_draw() for s in fleet_rf   if s.is_active))
        lstm_power_log.append(sum(s.compute_power_draw() for s in fleet_lstm if s.is_active))
        t += dt

    rf_rep   = autoscaler_rf.report()
    lstm_rep = autoscaler_lstm.report()
    baseline_power = sum(s.tdp_watts for s in fleet_rf)

    print(f"\n  {'Metrik':<38} {'RF':>14} {'LSTM':>14}")
    print(f"  {'-'*68}")
    rows = [
        ("Ort. Aktif Sunucu",          f"{np.mean(rf_active_log):.1f}",   f"{np.mean(lstm_active_log):.1f}"),
        ("Ort. Güç (W)",               f"{np.mean(rf_power_log):.0f}",    f"{np.mean(lstm_power_log):.0f}"),
        ("Taban Güç (tüm aktif, W)",   f"{baseline_power}",               f"{baseline_power}"),
        ("Tasarruf Oranı (%)",
         f"{(1-np.mean(rf_power_log)/baseline_power)*100:.1f}%",
         f"{(1-np.mean(lstm_power_log)/baseline_power)*100:.1f}%"),
        ("Konsolidasyon Sayısı",       rf_rep['consolidations'],           lstm_rep['consolidations']),
        ("Scale-out Sayısı",           rf_rep['scale_outs'],               lstm_rep['scale_outs']),
        ("Toplam Tasarruf ($/saat)",   f"{rf_rep['total_cost_saved_per_hour_usd']:.4f}",
                                       f"{lstm_rep['total_cost_saved_per_hour_usd']:.4f}"),
        ("Son Aktif Sunucu",           rf_rep['current_active_servers'],   lstm_rep['current_active_servers']),
        ("Son Uyku Sunucu",            rf_rep['current_idle_servers'],     lstm_rep['current_idle_servers']),
    ]
    for label, rv, lv in rows:
        print(f"  {label:<38} {str(rv):>14} {str(lv):>14}")

    if llm_advisor is not None:
        rep = llm_advisor.report()
        print(f"\n  LLM Advisor | online={rep['online']} | calls={rep['calls_total']} | "
              f"success={rep['calls_success']} | failed={rep['calls_failed']}")
        if rep.get("last_error"):
            print(f"  LLM son hata: {rep['last_error']}")

    return {
        "rf":   {"active_avg": float(np.mean(rf_active_log)),
                 "power_avg":  float(np.mean(rf_power_log)),  **rf_rep},
        "lstm": {"active_avg": float(np.mean(lstm_active_log)),
                 "power_avg":  float(np.mean(lstm_power_log)), **lstm_rep},
    }


# ──────────────────────────────────────────────────────────
# Ana Pipeline
# ──────────────────────────────────────────────────────────

def main(args=None):
    for d in ["data", "models", "logs"]:
        Path(d).mkdir(exist_ok=True)

    all_results: Dict = {}
    llm_advisor = build_llm_advisor(args)

    # ══════════════════════════════════════════
    # PHASE 1: 30 günlük veri toplama
    # ══════════════════════════════════════════
    print_banner("Phase 1 — 30 Günlük Metrik Toplama")

    total_sec = CFG["total_days"] * 86400
    dt        = float(CFG["tick_seconds"])

    collector = MetricsCollector(data_dir="data", seed=42)
    rows = collector.collect(
        total_seconds=total_sec,
        dt=dt,
        verbose=True,
        chunk_size=100_000
    )
    collector.build_wide_format()

    df = pd.read_csv("data/metrics_snapshot.csv")
    server_ids = sorted(df["server_id"].unique())
    print(f"\n  Sunucu sayısı : {len(server_ids)}")
    print(f"  Zaman damgaları: {df['timestamp'].nunique():,}")
    print(f"  Toplam satır  : {len(df):,}")
    print(f"  CPU aralığı   : {df['cpu_pct'].min():.1f}% – {df['cpu_pct'].max():.1f}%")
    print(f"  Güç aralığı   : {df['power_watts'].min():.0f}W – {df['power_watts'].max():.0f}W")
    print(f"  Hafta sayısı  : {df['week_of_sim'].nunique() if 'week_of_sim' in df.columns else 'N/A'}")

    # ══════════════════════════════════════════
    # PHASE 2: Random Forest
    # ══════════════════════════════════════════
    print_banner("Phase 2 — Random Forest (Haftalık Gecikmeli Özellikler)")

    rf_results_all   = []
    rf_model_per_srv = {}
    rf_wl_per_srv    = {}

    for srv_id in server_ids:
        X, y = prepare_features(
            df, srv_id,
            lag_steps=CFG["rf_lag_steps"],
            weekly_lag_offsets=CFG["rf_weekly_lags"]
        )
        if len(X) < 100:
            print(f"  [{srv_id}] Yetersiz veri ({len(X)} örnek), atlandı.")
            continue

        X_tr, X_te, y_tr, y_te = train_test_split_time(X, y, test_ratio=0.2)
        print(f"\n  [{srv_id}]  özellik={X.shape[1]}  train={len(X_tr)}  test={len(X_te)}")

        rf = RandomForestRegressor(
            n_estimators=CFG["rf_n_estimators"],
            max_depth=CFG["rf_max_depth"],
            rng_seed=42
        )
        t0 = time.time()
        rf.fit(X_tr, y_tr)
        print(f"    Eğitim süresi: {time.time()-t0:.1f}s")

        result = evaluate_model("RandomForest", y_te, rf.predict(X_te))
        result["server_id"] = srv_id
        rf_results_all.append(result)
        rf_model_per_srv[srv_id] = rf

    rf_avg_mae = float(np.mean([r["MAE"] for r in rf_results_all]))
    rf_avg_r2  = float(np.mean([r["R2"]  for r in rf_results_all]))
    print(f"\n  ── RF Genel ── MAE={rf_avg_mae:.3f} | R²={rf_avg_r2:.4f}")
    all_results["random_forest"] = {"avg_mae": rf_avg_mae, "avg_r2": rf_avg_r2}

    with open("models/rf_models.pkl", "wb") as f:
        pickle.dump(rf_model_per_srv, f)
    print("  ✓ RF modelleri kaydedildi → models/rf_models.pkl")

    # ══════════════════════════════════════════
    # PHASE 3: Paylaşımlı LSTM
    # ══════════════════════════════════════════
    print_banner("Phase 3 — Paylaşımlı LSTM (PyTorch, Haftalık Bağlam)")

    seq_len = CFG["lstm_seq_len"]
    print(f"  Diziler oluşturuluyor…  (seq_len={seq_len}, lag={CFG['rf_weekly_lags'][0]})")

    t0 = time.time()
    X_all, y_all = prepare_lstm_sequences(
        df, server_ids,
        seq_len=seq_len,
        weekly_lag_offsets=[CFG["rf_weekly_lags"][0]]
    )
    print(f"  Dizi hazırlama süresi: {time.time()-t0:.1f}s")

    shared_lstm = None   # Phase 5'te her zaman erişilebilir olsun
    if len(X_all) == 0:
        print("  [UYARI] LSTM için yeterli veri yok — Phase 3 atlandı.")
        lstm_results_all = []
        all_results["lstm"] = {"avg_mae": -1.0, "avg_r2": -1.0}
    else:
        X_tr, X_te, y_tr, y_te = train_test_split_time(X_all, y_all, test_ratio=0.2)
        input_size = X_all.shape[2]
        print(f"  Toplam örnek : {len(X_all):,}")
        print(f"  Giriş boyutu : seq={seq_len} × feat={input_size}")
        print(f"  Train={len(X_tr):,}  Test={len(X_te):,}")

        shared_lstm = LSTMModel(
            input_size=input_size,
            hidden_size=CFG["lstm_hidden"],
            rng_seed=42
        )
        t0 = time.time()
        history = shared_lstm.fit(
            X_tr, y_tr,
            epochs=CFG["lstm_epochs"],
            batch_size=CFG["lstm_batch"],
            verbose=True,
            X_val=X_te, y_val=y_te
        )
        print(f"  LSTM eğitim süresi: {time.time()-t0:.1f}s")

        lstm_result = evaluate_model("Shared-LSTM", y_te, shared_lstm.predict(X_te))
        lstm_result["server_id"] = "shared"
        lstm_results_all = [lstm_result]
        all_results["lstm"] = {"avg_mae": lstm_result["MAE"], "avg_r2": lstm_result["R2"]}

        with open("models/lstm_shared.pkl", "wb") as f:
            pickle.dump(shared_lstm, f)
        print("  ✓ Shared LSTM kaydedildi → models/lstm_shared.pkl")

    # ══════════════════════════════════════════
    # PHASE 4: GraphSAGE
    # ══════════════════════════════════════════
    print_banner("Phase 4 — Inductive GraphSAGE (PyTorch, Haftalık Özellikli)")

    X_snaps, y_snaps = prepare_graphsage_snapshots(df, n_snapshots=CFG["gsage_snapshots"])
    split   = int(len(X_snaps) * 0.8)
    X_tr_s  = X_snaps[:split];  X_te_s = X_snaps[split:]
    y_tr_s  = y_snaps[:split];  y_te_s = y_snaps[split:]

    print(f"  Snapshot: {len(X_snaps)} (train={split}, test={len(X_te_s)})")
    print(f"  Node sayısı: {X_snaps[0].shape[0]},  Özellik: {X_snaps[0].shape[1]}")

    gsage = GraphSAGEModel(
        in_features=X_snaps[0].shape[1],
        hidden_sizes=CFG["gsage_hidden"],
        rng_seed=42
    )
    t0 = time.time()
    gsage.fit(X_tr_s, y_tr_s, epochs=CFG["gsage_epochs"], verbose=True)
    print(f"  Eğitim süresi: {time.time()-t0:.1f}s")

    gsage_preds, gsage_true = [], []
    for Xs, ys in zip(X_te_s, y_te_s):
        gsage_preds.extend(gsage.predict(Xs).tolist())
        gsage_true.extend(ys.tolist())

    gsage_result = evaluate_model("GraphSAGE", np.array(gsage_true), np.array(gsage_preds))
    all_results["graphsage"] = {"avg_mae": gsage_result["MAE"], "avg_r2": gsage_result["R2"]}

    with open("models/graphsage.pkl", "wb") as f:
        pickle.dump(gsage, f)
    print("  ✓ GraphSAGE kaydedildi → models/graphsage.pkl")

    # ══════════════════════════════════════════
    # PHASE 5: Autoscaler (24 saat)
    # ══════════════════════════════════════════
    scaler_results = run_autoscaler_evaluation(
        fleet_factory=create_server_fleet,
        rf_model_per_srv=rf_model_per_srv,
        rf_wl_per_srv=rf_wl_per_srv,
        shared_lstm=shared_lstm,
        total_secs=3600 * CFG["autoscaler_hours"],
        rf_lag_steps=CFG["rf_lag_steps"],
        lstm_seq_len=CFG["lstm_seq_len"],
        llm_advisor=llm_advisor,
        llm_interval_ticks=getattr(args, "llm_interval_ticks", CFG["llm_interval_ticks"]),
    )
    all_results["autoscaler"] = scaler_results

    # ══════════════════════════════════════════
    # SONUÇ RAPORU
    # ══════════════════════════════════════════
    print_banner("Sonuç Özeti")

    print(f"\n  {'Model':<22} {'Ort. MAE':>10} {'Ort. R²':>10}")
    print(f"  {'-'*44}")
    for name, res in all_results.items():
        if "avg_mae" in res and res["avg_mae"] >= 0:
            print(f"  {name:<22} {res['avg_mae']:>10.3f} {res['avg_r2']:>10.4f}")

    print(f"\n  Autoscaler (24 saatlik test):")
    for mname, rep in scaler_results.items():
        print(f"    [{mname.upper()}] Aktif={rep['active_avg']:.1f} | "
              f"Güç={rep['power_avg']:.0f}W | "
              f"Konsolide={rep['consolidations']} | "
              f"Tasarruf=${rep['total_cost_saved_per_hour_usd']:.4f}/saat")

    with open("logs/training_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\n  ✓ Tüm sonuçlar → logs/training_results.json")

    print(f"\n{'═'*68}\n  Eğitim tamamlandı.\n{'═'*68}\n")
    return all_results


def eval_only(args=None):
    """
    Kaydedilmiş modelleri yükleyip sadece autoscaler testini çalıştırır.
    Eğitim gerekmez — Phase 2-4 atlanır.

    Kullanım:
        python train.py --eval-only
    """
    import sys
    for d in ["data", "models", "logs"]:
        Path(d).mkdir(exist_ok=True)

    print_banner("Eval-Only Modu — Kayıtlı Modeller Yükleniyor")
    llm_advisor = build_llm_advisor(args)

    # ── RF modelleri ───────────────────────────────────────────
    rf_path = Path("models/rf_models.pkl")
    if not rf_path.exists():
        print("  [HATA] models/rf_models.pkl bulunamadı. Önce tam eğitimi çalıştırın.")
        sys.exit(1)
    with open(rf_path, "rb") as f:
        rf_model_per_srv = pickle.load(f)
    print(f"  ✓ RF modelleri yüklendi  ({len(rf_model_per_srv)} sunucu)")
    rf_wl_per_srv = {}

    # ── Shared LSTM ────────────────────────────────────────────
    lstm_path = Path("models/lstm_shared.pkl")
    shared_lstm = None
    if lstm_path.exists():
        with open(lstm_path, "rb") as f:
            shared_lstm = pickle.load(f)
        print("  ✓ Shared LSTM yüklendi")
    else:
        print("  [UYARI] models/lstm_shared.pkl bulunamadı — LSTM tahminleri heuristic kullanır.")

    # ── GraphSAGE (isteğe bağlı yeniden değerlendirme) ─────────
    gsage_path = Path("models/graphsage.pkl")
    if gsage_path.exists():
        with open(gsage_path, "rb") as f:
            gsage = pickle.load(f)
        # Kaydedilmiş snapshot varsa kısa değerlendirme yap
        snapshot = Path("data/metrics_snapshot.csv")
        if snapshot.exists():
            print("  GraphSAGE snapshot testi yapılıyor…")
            df_gs = pd.read_csv(snapshot)
            X_snaps, y_snaps = prepare_graphsage_snapshots(df_gs, n_snapshots=100)
            if len(X_snaps) >= 10:
                split = int(len(X_snaps) * 0.8)
                gsage_preds, gsage_true = [], []
                for Xs, ys in zip(X_snaps[split:], y_snaps[split:]):
                    gsage_preds.extend(gsage.predict(Xs).tolist())
                    gsage_true.extend(ys.tolist())
                evaluate_model("GraphSAGE (saved)", np.array(gsage_true), np.array(gsage_preds))
    else:
        print("  [UYARI] models/graphsage.pkl bulunamadı — GraphSAGE atlandı.")

    # ── Autoscaler testi ───────────────────────────────────────
    scaler_results = run_autoscaler_evaluation(
        fleet_factory=create_server_fleet,
        rf_model_per_srv=rf_model_per_srv,
        rf_wl_per_srv=rf_wl_per_srv,
        shared_lstm=shared_lstm,
        total_secs=3600 * CFG["autoscaler_hours"],
        rf_lag_steps=CFG["rf_lag_steps"],
        lstm_seq_len=CFG["lstm_seq_len"],
        llm_advisor=llm_advisor,
        llm_interval_ticks=getattr(args, "llm_interval_ticks", CFG["llm_interval_ticks"]),
    )

    print_banner("Eval-Only Sonuçları")
    for mname, rep in scaler_results.items():
        print(f"  [{mname.upper()}] Aktif={rep['active_avg']:.1f} | "
              f"Güç={rep['power_avg']:.0f}W | "
              f"Konsolide={rep['consolidations']} | "
              f"Tasarruf=${rep['total_cost_saved_per_hour_usd']:.4f}/saat")

    with open("logs/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(scaler_results, f, indent=2, ensure_ascii=False)
    print("\n  ✓ Eval sonuçları → logs/eval_results.json")
    return scaler_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CPU Autoscaler Pipeline")
    parser.add_argument("--eval-only", action="store_true",
                        help="Eğitim yapmadan kayıtlı modelleri yükle ve autoscaler'ı test et")
    parser.add_argument("--use-llm", action="store_true",
                        help="Dominant workload seçimini LLM danışmanı ile güçlendir")
    parser.add_argument("--llm-model", type=str, default=CFG["llm_model"],
                        help="Responses API model adı (örn: gpt-4.1-mini)")
    parser.add_argument("--llm-endpoint", type=str, default=CFG["llm_endpoint"],
                        help="LLM endpoint URL")
    parser.add_argument("--llm-timeout", type=float, default=20.0,
                        help="LLM API timeout (saniye)")
    parser.add_argument("--llm-interval-ticks", type=int, default=CFG["llm_interval_ticks"],
                        help="Kaç tick'te bir LLM çağrısı yapılacağı")
    parser.add_argument("--llm-api-key", type=str, default=None,
                        help="OPENAI_API_KEY yerine doğrudan API key vermek için")
    args = parser.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        main(args)
