import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
import time

@dataclass
class ServerProfile:
    """Realistic server hardware profile"""
    server_id: str
    name: str
    cpu_cores: int
    base_clock_ghz: float
    boost_clock_ghz: float
    tdp_watts: int                  # Thermal Design Power (max heat/power)
    idle_power_watts: float         # Power at 0% CPU
    cost_per_hour_usd: float        # Cloud/datacenter cost
    memory_gb: int
    max_requests_per_sec: int       # Throughput ceiling
    latency_base_ms: float          # Baseline latency at idle
    thermal_throttle_temp_c: float  # Throttles above this
    reliability_score: float        # 0-1, MTBF-based
    generation: str                 # e.g. "Intel Xeon Gen4"
    arch: str                       # e.g. "x86", "ARM"
    workload_affinity: tuple = ()
    current_cpu_pct: float = 0.0
    current_temp_c: float = 35.0
    current_memory_pct: float = 0.0
    current_requests: float = 0.0
    is_active: bool = True
    uptime_seconds: float = 0.0
    total_energy_kwh: float = 0.0
    total_cost_usd: float = 0.0
    throttle_events: int = 0

    def affinity_bonus(self, workload_type_value: str) -> float:
        if not self.workload_affinity:
            return 1.0
        if len(self.workload_affinity) > 0 and self.workload_affinity[0] == workload_type_value:
            return 1.30   # birincil affinity: %30 verimlilik bonusu
        if workload_type_value in self.workload_affinity:
            return 1.15   # ikincil affinity: %15 bonus
        return 0.90       # uyumsuz iş yükü: %10 penaltı

    def compute_efficiency_score(self) -> float:
        """
        Sunucunun ham işlem gücünü normalize eden skor (1.0 = referans).

        Formül: (cpu_cores × boost_clock_ghz × arch_factor) / referans_güç
        Referans: srv-budget-01 (14 çekirdek, 3.3 GHz, x86) = 1.0

        Yüksek skor → aynı istek yükü için daha düşük CPU yüzdesi demek.
        ARM mimarisi IPC avantajı nedeniyle +%15 katsayısı alır.
        """
        arch_factor = 1.15 if self.arch == "ARM" else 1.0
        raw_power   = self.cpu_cores * self.boost_clock_ghz * arch_factor
        reference   = 14 * 3.3 * 1.0   # srv-budget-01 referans noktası
        return raw_power / reference

    def compute_power_draw(self) -> float:
        """Realistic power draw based on CPU utilization (not linear!)"""
        # Power scales roughly as idle + (TDP-idle) * util^1.4
        load_factor = (self.current_cpu_pct / 100.0) ** 1.4
        power = self.idle_power_watts + (self.tdp_watts - self.idle_power_watts) * load_factor
        # Thermal throttling penalty
        if self.current_temp_c > self.thermal_throttle_temp_c:
            power *= 1.08  # extra draw when throttling
        return min(power, self.tdp_watts * 1.1)

    def compute_effective_speed(self) -> float:
        """Actual clock speed considering thermal throttling"""
        if self.current_temp_c > self.thermal_throttle_temp_c:
            throttle_ratio = min(1.0, (self.current_temp_c - self.thermal_throttle_temp_c) / 20.0)
            effective = self.boost_clock_ghz - throttle_ratio * (self.boost_clock_ghz - self.base_clock_ghz)
            return max(effective, self.base_clock_ghz * 0.7)
        return self.boost_clock_ghz if self.current_cpu_pct > 60 else self.base_clock_ghz

    def compute_latency(self) -> float:
        """Response latency increases with load and thermal state"""
        base = self.latency_base_ms
        # Queueing theory: latency spikes near 100% utilization
        util = self.current_cpu_pct / 100.0
        queue_factor = 1.0 / max(0.01, 1.0 - util * 0.95)
        thermal_factor = 1.0 + max(0.0, (self.current_temp_c - self.thermal_throttle_temp_c) / 30.0)
        return base * queue_factor * thermal_factor

    def update_thermal(self, dt_seconds: float = 1.0):
        """Thermal dynamics: heat up under load, cool when idle"""
        ambient_temp = 25.0
        power = self.compute_power_draw()
        # Thermal resistance model
        target_temp = ambient_temp + (power / self.tdp_watts) * (self.thermal_throttle_temp_c - ambient_temp + 20)
        tau = 60.0  # thermal time constant in seconds
        alpha = 1.0 - np.exp(-dt_seconds / tau)
        self.current_temp_c = self.current_temp_c + alpha * (target_temp - self.current_temp_c)
        if self.current_temp_c > self.thermal_throttle_temp_c:
            self.throttle_events += 1

    def update_cost_energy(self, dt_seconds: float = 1.0):
        power_w = self.compute_power_draw()
        hours = dt_seconds / 3600.0
        self.total_energy_kwh += power_w * hours / 1000.0
        self.total_cost_usd += self.cost_per_hour_usd * hours
        self.uptime_seconds += dt_seconds


def create_server_fleet() -> list[ServerProfile]:
    """
    Create a diverse, realistic fleet of servers.
    Mix of budget, mid-range, high-performance, and ARM servers.
    """
    servers = [
        # === HIGH-PERFORMANCE XEON SERVERS ===
        # ─── HIGH-PERFORMANCE XEON: iş + gaming latency ───────────────────────
        ServerProfile(
            server_id="srv-xeon-01",
            name="Intel Xeon Gold 6354 (18C)",
            cpu_cores=18, base_clock_ghz=3.0, boost_clock_ghz=4.0,
            tdp_watts=205, idle_power_watts=65, cost_per_hour_usd=0.48,
            memory_gb=128, max_requests_per_sec=12000,
            latency_base_ms=1.2, thermal_throttle_temp_c=95,
            reliability_score=0.9995, generation="Intel Xeon Gen3", arch="x86",
            workload_affinity=("business", "gaming"),
        ),
        ServerProfile(
            server_id="srv-xeon-02",
            name="Intel Xeon Platinum 8380 (40C)",
            cpu_cores=40, base_clock_ghz=2.3, boost_clock_ghz=3.4,
            tdp_watts=270, idle_power_watts=85, cost_per_hour_usd=0.92,
            memory_gb=256, max_requests_per_sec=20000,
            latency_base_ms=1.8, thermal_throttle_temp_c=90,
            reliability_score=0.9998, generation="Intel Xeon Gen3", arch="x86",
            workload_affinity=("business", "data_batch"),   # yüksek çekirdek sayısı
        ),
        # ─── MID-RANGE AMD EPYC ────────────────────────────────────────────────
        ServerProfile(
            server_id="srv-epyc-01",
            name="AMD EPYC 7543 (32C)",
            cpu_cores=32, base_clock_ghz=2.8, boost_clock_ghz=3.7,
            tdp_watts=225, idle_power_watts=55, cost_per_hour_usd=0.62,
            memory_gb=128, max_requests_per_sec=15000,
            latency_base_ms=1.4, thermal_throttle_temp_c=95,
            reliability_score=0.9993, generation="AMD EPYC Milan", arch="x86",
            workload_affinity=("data_batch", "business"),
        ),
        ServerProfile(
            server_id="srv-epyc-02",
            name="AMD EPYC 9354 (32C) Gen4",
            cpu_cores=32, base_clock_ghz=3.25, boost_clock_ghz=3.8,
            tdp_watts=280, idle_power_watts=60, cost_per_hour_usd=0.78,
            memory_gb=256, max_requests_per_sec=18000,
            latency_base_ms=1.1, thermal_throttle_temp_c=95,
            reliability_score=0.9996, generation="AMD EPYC Genoa", arch="x86",
            workload_affinity=("gaming", "business"),  # yüksek boost clock
        ),
        # ─── ARM / GRAVITON: enerji verimli, streaming dostu ──────────────────
        ServerProfile(
            server_id="srv-arm-01",
            name="AWS Graviton3 (64C)",
            cpu_cores=64, base_clock_ghz=2.6, boost_clock_ghz=2.6,
            tdp_watts=150, idle_power_watts=35, cost_per_hour_usd=0.41,
            memory_gb=128, max_requests_per_sec=14000,
            latency_base_ms=1.6, thermal_throttle_temp_c=100,
            reliability_score=0.9997, generation="ARM Neoverse V1", arch="ARM",
            workload_affinity=("streaming", "cdn_static"),  # enerji verimli, yüksek bw
        ),
        ServerProfile(
            server_id="srv-arm-02",
            name="Ampere Altra Max (128C)",
            cpu_cores=128, base_clock_ghz=3.0, boost_clock_ghz=3.0,
            tdp_watts=250, idle_power_watts=45, cost_per_hour_usd=0.55,
            memory_gb=512, max_requests_per_sec=22000,
            latency_base_ms=2.0, thermal_throttle_temp_c=105,
            reliability_score=0.9994, generation="ARM Neoverse N1", arch="ARM",
            workload_affinity=("streaming", "ml_train", "data_batch"),   # 512GB RAM
        ),
        # ─── BUDGET: sadece batch'e uygun ─────────────────────────────────────
        ServerProfile(
            server_id="srv-budget-01",
            name="Intel Xeon E5-2680v4 (14C) Legacy",
            cpu_cores=14, base_clock_ghz=2.4, boost_clock_ghz=3.3,
            tdp_watts=120, idle_power_watts=45, cost_per_hour_usd=0.18,
            memory_gb=64, max_requests_per_sec=5000,
            latency_base_ms=3.5, thermal_throttle_temp_c=85,
            reliability_score=0.9980, generation="Intel Broadwell", arch="x86",
            workload_affinity=("data_batch",),   # yüksek latency, sadece data_batch tolere eder
        ),
        ServerProfile(
            server_id="srv-budget-02",
            name="AMD EPYC 7282 (16C) Entry",
            cpu_cores=16, base_clock_ghz=2.8, boost_clock_ghz=3.2,
            tdp_watts=120, idle_power_watts=40, cost_per_hour_usd=0.22,
            memory_gb=64, max_requests_per_sec=6000,
            latency_base_ms=2.8, thermal_throttle_temp_c=90,
            reliability_score=0.9985, generation="AMD EPYC Rome", arch="x86",
            workload_affinity=("data_batch", "cdn_static"),
        ),
        # ─── EDGE: çok düşük latency → gaming birincil ────────────────────────
        ServerProfile(
            server_id="srv-edge-01",
            name="Intel Core i9-13900K Edge Node",
            cpu_cores=24, base_clock_ghz=3.0, boost_clock_ghz=5.8,
            tdp_watts=253, idle_power_watts=70, cost_per_hour_usd=0.35,
            memory_gb=32, max_requests_per_sec=8000,
            latency_base_ms=0.8, thermal_throttle_temp_c=100,
            reliability_score=0.9975, generation="Intel Raptor Lake", arch="x86",
            workload_affinity=("gaming", "api_gateway"),    # 0.8ms latency → oyun için ideal
        ),
        ServerProfile(
            server_id="srv-edge-02",
            name="AMD Ryzen 9 7950X Edge Node",
            cpu_cores=16, base_clock_ghz=4.5, boost_clock_ghz=5.7,
            tdp_watts=170, idle_power_watts=55, cost_per_hour_usd=0.28,
            memory_gb=32, max_requests_per_sec=7500,
            latency_base_ms=0.9, thermal_throttle_temp_c=95,
            reliability_score=0.9978, generation="AMD Zen4", arch="x86",
            workload_affinity=("gaming", "api_gateway"),    # yüksek boost clock + düşük latency
        ),
    ]
    return servers