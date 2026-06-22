import numpy as np
import csv
import os
import time
from pathlib import Path
from typing import List, Dict

from core.server_profiles import ServerProfile, create_server_fleet
from core.traffic_generator import TrafficGenerator, TrafficPattern, WorkloadType


class MetricsCollector:

    SNAPSHOT_COLS = [
        # Zaman
        "timestamp",
        "day_of_sim",           # kaçıncı gün (0-indexed)
        "hour_of_day",          # 0.0–23.99
        "day_of_week",          # 0=Pzt … 6=Paz
        "is_weekday",           # 1/0
        "week_of_sim",          # kaçıncı hafta
        # Zaman kodlamaları (sinüzoidal)
        "hour_sin", "hour_cos",
        "dow_sin",  "dow_cos",  # day-of-week
        "wom_sin",  "wom_cos",  # week-of-month (4 haftalık periyot)
        # Sunucu kimliği
        "server_id", "server_name", "arch", "generation",
        "cpu_cores", "tdp_watts", "cost_per_hour_usd",
        # Anlık metrikler
        "cpu_pct", "memory_pct", "requests_norm",
        "latency_ms", "power_watts", "temp_c",
        "effective_clock_ghz", "throttle_events",
        "total_energy_kwh", "total_cost_usd",
        # Trafik bağlamı
        "traffic_pattern", "workload_type",
        "request_origin", "latency_sensitivity", "bandwidth_pressure",
        "is_active",
    ]

    def __init__(self, data_dir: str = "data", seed: int = 42):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.fleet: List[ServerProfile] = create_server_fleet()
        self.traffic_gen = TrafficGenerator(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.server_pattern_offset = {
            srv.server_id: float(self.rng.uniform(0, 3600 * 4))
            for srv in self.fleet
        }

    # ─────────────────────────────────────────
    # Zaman özelliği hesapları
    # ─────────────────────────────────────────

    @staticmethod
    def _time_features(t: float) -> Dict:
        """t saniye cinsinden simülasyon zamanı → zengin zaman özellikleri"""
        day_of_sim  = int(t // 86400)
        hour_of_day = (t % 86400) / 3600.0
        day_of_week = day_of_sim % 7          # 0=Pzt
        week_of_sim = day_of_sim // 7

        # Sinüzoidal kodlamalar
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
        dow_sin  = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos  = np.cos(2 * np.pi * day_of_week / 7)
        # Aylık hafta: 4 haftalık periyot
        wom      = week_of_sim % 4
        wom_sin  = np.sin(2 * np.pi * wom / 4)
        wom_cos  = np.cos(2 * np.pi * wom / 4)

        return {
            "day_of_sim":  day_of_sim,
            "hour_of_day": round(hour_of_day, 4),
            "day_of_week": day_of_week,
            "is_weekday":  int(day_of_week < 5),
            "week_of_sim": week_of_sim,
            "hour_sin":    round(float(hour_sin), 6),
            "hour_cos":    round(float(hour_cos), 6),
            "dow_sin":     round(float(dow_sin),  6),
            "dow_cos":     round(float(dow_cos),  6),
            "wom_sin":     round(float(wom_sin),  6),
            "wom_cos":     round(float(wom_cos),  6),
        }

    # ─────────────────────────────────────────
    # Trafik uygulaması
    # ─────────────────────────────────────────

    def _apply_traffic_to_server(self, server: ServerProfile,
                                  load: float, event, dt: float):
        # 1. Donanım verimliliği: güçlü sunucu aynı yükü daha az CPU'yla karşılar
        eff = server.compute_efficiency_score()

        # 2. Workload affinity bonusu:
        #    - Sunucu bu iş yükü için optimize edilmişse CPU daha da düşer
        #    - Uyumsuz iş yükündeyse hafif penaltı alır
        wl_value = event.workload_type.value if hasattr(event, "workload_type") else "business"
        aff_bonus = server.affinity_bonus(wl_value)   # 0.90 – 1.30

        # 3. Streaming iş yükünde CPU düşük ama memory yüksek olmalı
        #    Gaming iş yükünde CPU tutarlı yüksek
        combined_eff = eff * aff_bonus

        cpu_noise  = self.rng.standard_normal() * 2.5
        target_cpu = (load * 100.0 * event.cpu_intensity) / max(combined_eff, 0.3)
        alpha = 0.3
        server.current_cpu_pct = float(np.clip(
            alpha * target_cpu + (1 - alpha) * server.current_cpu_pct + cpu_noise,
            0.0, 100.0
        ))
        # Memory: streaming affinity'si olan sunucular yüksek memory baskısını
        # daha iyi absorbe eder (512GB vs 32GB farkı)
        mem_scale = max(np.sqrt(eff), 0.5)
        # Streaming'de memory baskısı CPU'dan bağımsız olarak yükselir
        bw_factor = getattr(event, "bandwidth_pressure", 0.3)
        mem_noise = self.rng.standard_normal() * 1.5
        mem_pressure_scaled = (event.memory_pressure * 80 + bw_factor * 15) / mem_scale
        server.current_memory_pct = float(np.clip(
            mem_pressure_scaled + server.current_cpu_pct * 0.12 + mem_noise,
            5.0, 95.0
        ))
        req_noise = self.rng.standard_normal() * 0.02
        server.current_requests = float(np.clip(
            load * server.max_requests_per_sec
            + req_noise * server.max_requests_per_sec,
            0.0, server.max_requests_per_sec * 1.05
        ))
        server.update_thermal(dt)
        server.update_cost_energy(dt)

    def _cascading_failure_check(self, t: float):
        if self.rng.random() < 0.0003:   # 30 günde ~13 arıza
            active = [s for s in self.fleet if s.is_active]
            if len(active) > 3:
                victim = self.rng.choice(active)
                victim.is_active = False
                day = int(t // 86400)
                print(f"  [FAULT] {victim.server_id} offline  (gün {day}, t={t:.0f}s)")

    # ─────────────────────────────────────────
    # Ana toplama döngüsü
    # ─────────────────────────────────────────

    def collect(self, total_seconds: int = 86400 * 30, dt: float = 60.0,
                verbose: bool = True, chunk_size: int = 50_000) -> int:
        """
        total_seconds : simülasyon süresi (varsayılan 30 gün)
        dt            : tick aralığı saniye (varsayılan 60 s = 1 dakika)
        chunk_size    : belleğe alınmadan doğrudan diske yazılan satır sayısı
        """
        snapshot_path = self.data_dir / "metrics_snapshot.csv"
        rows_written  = 0
        total_ticks   = int(total_seconds / dt)
        total_days    = total_seconds / 86400

        if verbose:
            print(f"\n{'='*64}")
            print(f"  CPU Autoscaler — Veri Toplama  (v2 / 30-gün)")
            print(f"  Filo     : {len(self.fleet)} sunucu")
            print(f"  Süre     : {total_seconds:,}s  ({total_days:.0f} gün)")
            print(f"  Tick     : {dt}s")
            print(f"  Tick sayısı: {total_ticks:,}")
            print(f"  Tahmini satır: ~{total_ticks * len(self.fleet):,}")
            print(f"{'='*64}\n")

        buffer: List[Dict] = []

        with open(snapshot_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SNAPSHOT_COLS)
            writer.writeheader()

            t = 0.0
            tick_count = 0
            last_hour_print = -1

            while t < total_seconds:


                pattern   = self.traffic_gen.select_pattern_for_tick(t)
                base_load = self.traffic_gen.get_load_for_pattern(pattern, t)
                tf        = self._time_features(t)

                self._cascading_failure_check(t)

                for server in self.fleet:
                    if not server.is_active:
                        continue

                    t_offset = t + self.server_pattern_offset[server.server_id]
                    workload = self.traffic_gen.select_workload_type(t)
                    event = self.traffic_gen.generate_event(
                        server.server_id, t_offset, pattern, base_load,
                        per_server_noise=0.06, workload_type=workload
                    )
                    self._apply_traffic_to_server(server, event.request_rate, event, dt)

                    row = {
                        "timestamp":          round(t, 1),
                        **tf,
                        "server_id":          server.server_id,
                        "server_name":        server.name,
                        "arch":               server.arch,
                        "generation":         server.generation,
                        "cpu_cores":          server.cpu_cores,
                        "tdp_watts":          server.tdp_watts,
                        "cost_per_hour_usd":  round(server.cost_per_hour_usd, 4),
                        "cpu_pct":            round(server.current_cpu_pct, 2),
                        "memory_pct":         round(server.current_memory_pct, 2),
                        "requests_norm":      round(server.current_requests / server.max_requests_per_sec, 4),
                        "latency_ms":         round(server.compute_latency(), 3),
                        "power_watts":        round(server.compute_power_draw(), 2),
                        "temp_c":             round(server.current_temp_c, 2),
                        "effective_clock_ghz":round(server.compute_effective_speed(), 3),
                        "throttle_events":    server.throttle_events,
                        "total_energy_kwh":   round(server.total_energy_kwh, 6),
                        "total_cost_usd":     round(server.total_cost_usd, 6),
                        "traffic_pattern":    pattern.value,
                        "workload_type":      event.workload_type.value,
                        "request_origin":     event.origin,
                        "latency_sensitivity": round(event.latency_sensitivity, 3),
                        "bandwidth_pressure":  round(event.bandwidth_pressure, 3),
                        "is_active":          int(server.is_active),
                    }
                    buffer.append(row)
                    rows_written += 1

                # Chunk dolunca diske yaz, belleği temizle
                if len(buffer) >= chunk_size:
                    writer.writerows(buffer)
                    buffer.clear()

                # Saatlik ilerleme yazdır
                if verbose:
                    cur_hour = int(t // 3600)
                    if cur_hour != last_hour_print and cur_hour % 12 == 0:
                        active_n = sum(1 for s in self.fleet if s.is_active)
                        avg_cpu  = np.mean([s.current_cpu_pct for s in self.fleet if s.is_active])
                        day      = tf["day_of_sim"]
                        dow_names = ["Pzt","Sal","Çar","Per","Cum","Cmt","Paz"]
                        dow_str  = dow_names[tf["day_of_week"]]
                        print(f"  Gün {day+1:2d} ({dow_str}) {tf['hour_of_day']:05.2f}h"
                              f" | pattern={pattern.value:<18}"
                              f" | active={active_n}/{len(self.fleet)}"
                              f" | avg_cpu={avg_cpu:.1f}%"
                              f" | rows={rows_written:,}")
                        last_hour_print = cur_hour

                t += dt
                tick_count += 1

            # Kalan tampon
            if buffer:
                writer.writerows(buffer)
                buffer.clear()

        if verbose:
            print(f"\n  ✓ Snapshot tamamlandı: {rows_written:,} satır → {snapshot_path}")

        return rows_written

    # ─────────────────────────────────────────
    # Wide-format dönüşümü
    # ─────────────────────────────────────────

    def build_wide_format(self) -> int:
        """
        Snapshot CSV → wide-format.
        30 günlük veri büyük olabileceğinden chunked okuma yapılır.
        """
        import pandas as pd

        snapshot_path = self.data_dir / "metrics_snapshot.csv"
        wide_path     = self.data_dir / "metrics_wide.csv"

        print("  Wide-format dönüşümü başlıyor (büyük dosya için birkaç dakika sürebilir)…")

        # Tüm veriyi oku (30 gün × 10 sunucu × 1440 tick = ~432k satır, yaklaşık 80MB)
        df = pd.read_csv(snapshot_path)

        pivot_cols = ["cpu_pct", "memory_pct", "requests_norm",
                      "latency_ms", "power_watts", "temp_c"]

        pivoted = []
        for metric in pivot_cols:
            piv = df.pivot_table(index="timestamp", columns="server_id",
                                  values=metric, aggfunc="mean")
            piv.columns = [f"{col}_{metric}" for col in piv.columns]
            pivoted.append(piv)

        wide_df = pd.concat(pivoted, axis=1).reset_index()

        # Zaman özelliklerini ekle (timestamp bazında ilk değer alınır)
        time_cols = ["day_of_sim", "hour_of_day", "day_of_week", "is_weekday",
                     "week_of_sim", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                     "wom_sin", "wom_cos"]
        time_df = df.groupby("timestamp")[time_cols].first()
        wide_df = wide_df.merge(time_df, on="timestamp", how="left")

        # Filo geneli istatistikler
        fleet_agg = df.groupby("timestamp").agg(
            fleet_avg_cpu      =("cpu_pct",     "mean"),
            fleet_max_cpu      =("cpu_pct",     "max"),
            fleet_min_cpu      =("cpu_pct",     "min"),
            fleet_total_power_w=("power_watts", "sum"),
            fleet_active_servers=("is_active",  "sum"),
            fleet_avg_latency_ms=("latency_ms", "mean"),
            fleet_avg_temp_c   =("temp_c",      "mean"),
        ).reset_index()
        wide_df = wide_df.merge(fleet_agg, on="timestamp", how="left")

        wide_df.to_csv(wide_path, index=False)
        print(f"  ✓ Wide-format: {len(wide_df):,} satır × {len(wide_df.columns)} sütun → {wide_path}")
        return len(wide_df)