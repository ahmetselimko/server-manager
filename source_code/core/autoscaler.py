from core.traffic_generator import WORKLOAD_SERVER_AFFINITY, WorkloadType
"""
Autoscaler Engine
- Receives CPU forecasts from ML models
- Finds the OPTIMAL server combination: minimum servers, minimum cost
- Goal: pack load onto fewest active servers → highest utilization → least waste
- Considers: power efficiency, cost, thermal headroom, reliability

Optimization logic:
  _find_optimal_combo() tries every subset of servers (2^N, N=10 → 1024 combos)
  and returns the smallest, cheapest set that can safely carry the current load.
  "Safely" means no server exceeds SAFETY_CAP_PCT CPU.
  Fewest servers → highest utilization per server → closest to %100 target.
"""

import numpy as np
from itertools import combinations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from core.server_profiles import ServerProfile


# ── Utilization targets ───────────────────────────────────────────────────────
TARGET_UTIL_PCT = 70.0   # aim for ~80% avg CPU on active servers
SAFETY_CAP_PCT  = 80.0   # hard ceiling — never let any server exceed this


@dataclass
class ScalingDecision:
    timestamp: float
    action: str                         # "consolidate" | "scale_out" | "rebalance" | "hold"
    servers_to_deactivate: List[str]
    servers_to_activate: List[str]
    reason: str
    predicted_fleet_cpu_pct: float
    current_fleet_cpu_pct: float
    estimated_power_saved_w: float
    estimated_cost_saved_per_hour: float
    projected_avg_util_pct: float = 0.0
    projected_max_util_pct: float = 0.0
    active_server_count: int = 0


class AutoscalerEngine:
    """
    Proactive autoscaler powered by ML predictions.

    Strategy:
    1. Get per-server CPU forecasts for next horizon
    2. Scale current RPS by predicted load ratio
    3. _find_optimal_combo(): enumerate all server subsets, pick the
       SMALLEST (fewest servers) set whose per-server CPU stays under
       SAFETY_CAP_PCT. Ties broken by lowest cost+power score.
    4. Transition current fleet to that optimal set.
    5. Result: active servers run near TARGET_UTIL_PCT (~80%),
       idle servers sleep → minimum cost, maximum utilization.
    """

    def __init__(
        self,
        fleet: List[ServerProfile],
        cpu_consolidate_threshold: float = 30.0,
        cpu_scaleout_threshold: float = 75.0,
        cpu_headroom: float = 13.0,
        min_active_servers: int = 2,
        cooldown_ticks: int = 5,
        cost_weight: float = 0.5,
        power_weight: float = 0.3,
        latency_weight: float = 0.2,
        warmup_ticks: int = 24,
    ):
        self.fleet                 = fleet
        self.consolidate_threshold = cpu_consolidate_threshold
        self.scaleout_threshold    = cpu_scaleout_threshold
        self.cpu_headroom          = cpu_headroom
        self.min_active_servers    = min_active_servers
        self.cooldown_ticks        = cooldown_ticks
        self.cost_weight           = cost_weight
        self.power_weight          = power_weight
        self.latency_weight        = latency_weight
        # Warmup: ilk N tick konsolidasyon/rebalance engellenir
        self.warmup_ticks          = warmup_ticks
        self._tick_count           = 0

        self._cooldown_remaining   = 0
        self._decisions: List[ScalingDecision] = []
        self._server_map: Dict[str, ServerProfile] = {s.server_id: s for s in fleet}
        self._efficiency_ranking   = self._rank_servers()
        self._util_samples: List[float] = []

        # Kümülatif maliyet takibi — karar snapshot'ı değil her tick'te hesaplanır
        self._cumulative_cost_saved_usd: float = 0.0
        self._last_timestamp: Optional[float]  = None

    # ─────────────────────────────────────────────────────────────────────────
    # CORE: Optimal combination finder
    # ─────────────────────────────────────────────────────────────────────────

    def _find_optimal_combo(self, total_rps: float,
                              predictions: Optional[Dict[str, float]] = None,
                              ignore_target: bool = False,
                              dominant_workload: Optional[str] = None) -> List[ServerProfile]:
        """
        Try every subset of the fleet (2^10 = 1024 max).
        Return the subset that:
          1. Has >= min_active_servers members
          2. Total capacity covers total_rps * 1.15 (15% headroom)
          3. No single server exceeds SAFETY_CAP_PCT CPU
          4. avg CPU <= TARGET_UTIL_PCT  (skip if ignore_target=True, e.g. forced scale-out)
          5. Among valid subsets: fewest servers first then lowest cost+power score

        predictions: ML tahminleri varsa per-server CPU hesabında kullanılır.
        ignore_target: True ise TARGET_UTIL_PCT üst sınırı atlanır (scale-out zorlaması).
        """
        if total_rps <= 0:
            return self._cheapest_n(self.min_active_servers)

        best_combo: Optional[List[ServerProfile]] = None
        best_size  = len(self.fleet) + 1
        best_score = float('inf')

        for r in range(self.min_active_servers, len(self.fleet) + 1):
            # Once we found a valid combo of size r, no need to look at r+1
            if best_combo is not None and r > best_size:
                break

            for combo in combinations(self.fleet, r):
                total_cap = sum(s.max_requests_per_sec for s in combo)
                if total_cap == 0:
                    continue

                # Capacity check (15% safety margin)
                if total_cap < total_rps * 1.15:
                    continue

                # Per-server CPU projection
                # Predictions varsa ML tahminini kullan → avg != max çıkar
                if predictions:
                    proj_cpus = [float(np.clip(
                                    predictions.get(s.server_id,
                                                    total_rps / total_cap * 100.0),
                                    0.0, 100.0))
                                 for s in combo]
                else:
                    proj_cpus = [total_rps / total_cap * 100.0 for _ in combo]

                max_proj_cpu = max(proj_cpus)
                avg_proj_cpu = sum(proj_cpus) / len(proj_cpus)

                # Hard ceiling: hiçbir sunucu SAFETY_CAP_PCT'yi geçemez
                if max_proj_cpu > SAFETY_CAP_PCT:
                    continue

                # Hedef bant: ortalama TARGET_UTIL_PCT altında kalmalı
                # (scale-out zorlamasında bu kural atlanır)
                if not ignore_target and avg_proj_cpu > TARGET_UTIL_PCT:
                    continue

                # Combo score: weighted cost + power + latency + affinity (lower = better)
                cost     = sum(s.cost_per_hour_usd for s in combo)
                power    = sum(s.tdp_watts for s in combo) / 1000.0
                lat_norm = max(s.latency_base_ms for s in combo) / 10.0
                eff_bonus = 1.0 / max(
                    sum(s.compute_efficiency_score() for s in combo) / len(combo), 0.1)

                # Workload affinity: dominant iş yüküne uygun combo tercih edilir
                # İki katman:
                # 1. server_profiles.workload_affinity string eşleşmesi (0.90–1.30)
                # 2. WORKLOAD_SERVER_AFFINITY donanım tercihleri (latency/clock/cores)
                if dominant_workload:
                    aff_scores = []
                    wl_pref = {}
                    for wl in WorkloadType:
                        if wl.value == dominant_workload:
                            wl_pref = WORKLOAD_SERVER_AFFINITY.get(wl, {})
                            break
                    for s in combo:
                        score = s.affinity_bonus(dominant_workload)   # 0.90–1.30
                        # Donanım tercihi bonusları
                        if wl_pref.get("prefer_low_latency") and s.latency_base_ms <= 1.2:
                            score *= 1.10
                        if wl_pref.get("prefer_high_clock") and s.boost_clock_ghz >= 4.0:
                            score *= 1.08
                        if wl_pref.get("prefer_many_cores") and s.cpu_cores >= 32:
                            score *= 1.06
                        if wl_pref.get("preferred_arch") and s.arch == wl_pref["preferred_arch"]:
                            score *= 1.12
                        aff_scores.append(score)
                    avg_aff  = sum(aff_scores) / len(aff_scores)
                    aff_term = 1.0 / max(avg_aff, 0.1)
                else:
                    aff_term = 1.0

                score = (self.cost_weight    * cost
                       + self.power_weight   * power
                       + self.latency_weight * lat_norm
                       + 0.08               * eff_bonus
                       + 0.12               * aff_term)

                # Update best: fewer servers first, then cheaper
                if r < best_size or (r == best_size and score < best_score):
                    best_size  = r
                    best_score = score
                    best_combo = list(combo)

        # Fallback: all servers (load too high for any subset)
        return best_combo if best_combo is not None else list(self.fleet)

    def _cheapest_n(self, n: int) -> List[ServerProfile]:
        """Return n cheapest servers (used when load ≈ 0)."""
        return sorted(self.fleet, key=lambda s: s.cost_per_hour_usd)[:n]

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _rank_servers(self) -> List[str]:
        """Fallback efficiency ranking."""
        scores = {}
        for s in self.fleet:
            throughput_score  = s.max_requests_per_sec / max(s.cost_per_hour_usd, 0.01)
            power_efficiency  = s.max_requests_per_sec / max(s.tdp_watts, 1)
            reliability_bonus = s.reliability_score * 100
            scores[s.server_id] = (throughput_score * 0.5
                                   + power_efficiency * 0.3
                                   + reliability_bonus * 0.2)
        return sorted(scores, key=scores.get, reverse=True)

    def _active_servers(self) -> List[ServerProfile]:
        return [s for s in self.fleet if s.is_active]

    def _total_load_rps(self) -> float:
        return sum(s.current_requests for s in self._active_servers())

    def _fleet_avg_cpu(self) -> float:
        active = self._active_servers()
        if not active:
            return 0.0
        return float(np.mean([s.current_cpu_pct for s in active]))

    def _fleet_total_power(self) -> float:
        return sum(s.compute_power_draw() for s in self._active_servers())

    def _projected_utilization(
        self, combo: List[ServerProfile], total_rps: float,
        predictions: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float]:
        """
        Returns (avg_cpu_pct, max_cpu_pct) for a given combo + load.
        Predictions varsa per-server ML tahminini kullanır → avg != max.
        """
        total_cap = sum(s.max_requests_per_sec for s in combo)
        if total_cap == 0:
            return 0.0, 0.0
        if predictions:
            utils = [float(np.clip(predictions.get(s.server_id,
                           total_rps / total_cap * 100.0), 0.0, 100.0))
                     for s in combo]
        else:
            utils = [total_rps / total_cap * 100.0 for _ in combo]
        return float(np.mean(utils)), float(np.max(utils))

    # ─────────────────────────────────────────────────────────────────────────
    # Main decision method
    # ─────────────────────────────────────────────────────────────────────────

    def decide(
        self,
        predictions: Dict[str, float],   # server_id → predicted CPU %
        timestamp: float,
        dominant_workload: Optional[str] = None,
    ) -> ScalingDecision:
        """
        Core autoscaling decision using combinatorial optimization.

        Steps:
          1. Estimate future RPS from predictions
          2. Find optimal server combo via _find_optimal_combo()
          3. Diff vs. current active set
          4. Wake/sleep servers accordingly
          5. Return ScalingDecision
        """
        # ── Tick sayacı ve kümülatif maliyet takibi ──────────────────────────
        self._tick_count += 1
        in_warmup = self._tick_count <= self.warmup_ticks

        # Her tick'te uyuyan sunucuların maliyet tasarrufunu kümülatif topla.
        # Karar snapshot'ı değil, gerçek zaman farkı kullanılır.
        if self._last_timestamp is not None:
            dt_hours = (timestamp - self._last_timestamp) / 3600.0
            idle_cost_rate = sum(s.cost_per_hour_usd
                                 for s in self.fleet if not s.is_active)
            self._cumulative_cost_saved_usd += idle_cost_rate * dt_hours
        self._last_timestamp = timestamp

        # ── Cooldown guard ────────────────────────────────────────────────
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            avg_util = self._fleet_avg_cpu()
            self._util_samples.append(avg_util)
            return ScalingDecision(
                timestamp=timestamp, action="hold",
                servers_to_deactivate=[], servers_to_activate=[],
                reason=f"Cooldown ({self._cooldown_remaining} ticks remaining)",
                predicted_fleet_cpu_pct=avg_util,
                current_fleet_cpu_pct=avg_util,
                estimated_power_saved_w=0.0,
                estimated_cost_saved_per_hour=0.0,
                projected_avg_util_pct=avg_util,
                active_server_count=len(self._active_servers()),
            )

        # ── Current state ─────────────────────────────────────────────────
        currently_active     = self._active_servers()
        current_active_ids: Set[str] = {s.server_id for s in currently_active}
        fleet_avg_cpu        = self._fleet_avg_cpu()
        total_rps            = self._total_load_rps()
        current_max_cpu      = max((s.current_cpu_pct for s in currently_active), default=0.0)

        # ── Scale RPS by predicted load ratio ────────────────────────────
        if predictions and fleet_avg_cpu > 1.0:
            predicted_avg = float(np.mean(list(predictions.values())))
            scale_factor  = predicted_avg / max(fleet_avg_cpu, 1.0)
            predicted_rps = total_rps * scale_factor
        else:
            predicted_avg = fleet_avg_cpu
            predicted_rps = total_rps

        # ── Forced scale-out: anlık CPU eşiği aşıldıysa model ne derse desin ──
        # Uyuyan sunucular varken herhangi bir aktif sunucu scaleout_threshold'u
        # geçtiyse, TARGET_UTIL_PCT sınırını görmezden gelerek daha büyük combo ara.
        has_sleeping = any(not s.is_active for s in self.fleet)
        force_scaleout = (current_max_cpu > self.scaleout_threshold and has_sleeping)

        # ── Find optimal combo ────────────────────────────────────────────
        optimal_combo = self._find_optimal_combo(
            predicted_rps, predictions,
            ignore_target=force_scaleout,
            dominant_workload=dominant_workload,
        )
        optimal_ids: Set[str] = {s.server_id for s in optimal_combo}

        # ── Diff ──────────────────────────────────────────────────────────
        ids_to_activate   = optimal_ids - current_active_ids
        ids_to_deactivate = current_active_ids - optimal_ids

        # ── Warmup guard: konsolidasyon/rebalance'ı ilk N tick'te engelle ────
        # Scale-out'a izin ver; sunucu uyutmayı engelle.
        if in_warmup and ids_to_deactivate:
            ids_to_deactivate = set()
            if not ids_to_activate:
                proj_avg, proj_max = self._projected_utilization(
                    list(currently_active), predicted_rps, predictions)
                self._util_samples.append(proj_avg)
                return ScalingDecision(
                    timestamp=timestamp, action="hold",
                    servers_to_deactivate=[], servers_to_activate=[],
                    reason=(f"Warmup ({self._tick_count}/{self.warmup_ticks}) "
                            f"— konsolidasyon ertelendi"),
                    predicted_fleet_cpu_pct=predicted_avg,
                    current_fleet_cpu_pct=fleet_avg_cpu,
                    estimated_power_saved_w=0.0,
                    estimated_cost_saved_per_hour=0.0,
                    projected_avg_util_pct=proj_avg,
                    projected_max_util_pct=proj_max,
                    active_server_count=len(currently_active),
                )

        # ── Project utilization ───────────────────────────────────────────
        proj_avg, proj_max = self._projected_utilization(
            optimal_combo, predicted_rps, predictions)
        self._util_samples.append(proj_avg)

        # ── No change ─────────────────────────────────────────────────────
        if not ids_to_activate and not ids_to_deactivate:
            return ScalingDecision(
                timestamp=timestamp, action="hold",
                servers_to_deactivate=[], servers_to_activate=[],
                reason=(f"Fleet optimal. CPU={fleet_avg_cpu:.1f}% "
                        f"| proj_util={proj_avg:.1f}% avg / {proj_max:.1f}% max "
                        f"| active={len(currently_active)} servers"),
                predicted_fleet_cpu_pct=predicted_avg,
                current_fleet_cpu_pct=fleet_avg_cpu,
                estimated_power_saved_w=0.0,
                estimated_cost_saved_per_hour=0.0,
                projected_avg_util_pct=proj_avg,
                projected_max_util_pct=proj_max,
                active_server_count=len(currently_active),
            )

        # ── Apply transitions ─────────────────────────────────────────────
        servers_waking   = []
        servers_sleeping = []
        power_saved      = 0.0
        cost_saved       = 0.0

        for srv in self.fleet:
            if srv.server_id in ids_to_deactivate:
                power_saved           += srv.compute_power_draw()
                cost_saved            += srv.cost_per_hour_usd
                srv.is_active          = False
                srv.current_cpu_pct    = 0.0
                servers_sleeping.append(srv.server_id)

            elif srv.server_id in ids_to_activate:
                srv.is_active          = True
                srv.current_cpu_pct    = 0.0
                srv.current_temp_c     = 35.0
                servers_waking.append(srv.server_id)

        # ── Action label ──────────────────────────────────────────────────
        if servers_sleeping and not servers_waking:
            action = "consolidate"
        elif servers_waking and not servers_sleeping:
            action = "scale_out"
        else:
            action = "rebalance"

        self._cooldown_remaining = self.cooldown_ticks

        reason = (
            f"Optimal combo → {len(optimal_combo)} servers active "
            f"| proj_util={proj_avg:.1f}% avg / {proj_max:.1f}% max "
            f"| pred_rps={predicted_rps:.0f} "
            f"| wake={servers_waking} sleep={servers_sleeping}"
        )

        decision = ScalingDecision(
            timestamp=timestamp,
            action=action,
            servers_to_deactivate=servers_sleeping,
            servers_to_activate=servers_waking,
            reason=reason,
            predicted_fleet_cpu_pct=predicted_avg,
            current_fleet_cpu_pct=fleet_avg_cpu,
            estimated_power_saved_w=power_saved,
            estimated_cost_saved_per_hour=cost_saved,
            projected_avg_util_pct=proj_avg,
            projected_max_util_pct=proj_max,
            active_server_count=len(optimal_combo),
        )
        self._decisions.append(decision)
        return decision

    # ─────────────────────────────────────────────────────────────────────────
    # Report
    # ─────────────────────────────────────────────────────────────────────────

    def report(self) -> Dict:
        """Summary of autoscaling efficiency"""
        consolidations = [d for d in self._decisions if d.action == "consolidate"]
        scaleouts      = [d for d in self._decisions if d.action == "scale_out"]
        rebalances     = [d for d in self._decisions if d.action == "rebalance"]

        total_power_saved = sum(
            d.estimated_power_saved_w for d in consolidations + rebalances
        )
        # Kümülatif maliyet: karar snapshot toplamı değil, her tick'te uyuyan
        # sunucuların gerçek zaman × maliyet/saat toplamı.
        cumulative_cost = self._cumulative_cost_saved_usd

        util_arr = np.array(self._util_samples) if self._util_samples else np.array([0.0])
        in_band  = float(np.mean(
            (util_arr >= TARGET_UTIL_PCT) & (util_arr <= SAFETY_CAP_PCT)) * 100)

        return {
            "total_decisions"              : len(self._decisions),
            "consolidations"               : len(consolidations),
            "scale_outs"                   : len(scaleouts),
            "rebalances"                   : len(rebalances),
            "total_power_saved_w"          : round(total_power_saved, 2),
            "total_cost_saved_per_hour_usd": round(cumulative_cost, 4),
            "current_active_servers"       : sum(1 for s in self.fleet if s.is_active),
            "current_idle_servers"         : sum(1 for s in self.fleet if not s.is_active),
            # Utilization quality
            "avg_active_util_pct"          : round(float(np.mean(util_arr)), 2),
            "max_active_util_pct"          : round(float(np.max(util_arr)), 2),
            "pct_ticks_in_target_band"     : round(in_band, 1),
            "target_util_band"             : f"{TARGET_UTIL_PCT:.0f}–{SAFETY_CAP_PCT:.0f}%",
        }