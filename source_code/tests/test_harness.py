"""
evaluate_policies.py için temel doğruluk testleri.
Çalıştırma (source_code klasöründen):  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_policies as ep                      # noqa: E402
from core.server_profiles import create_server_fleet  # noqa: E402

N_SRV = len(create_server_fleet())


@pytest.fixture(scope="module")
def trace():
    return ep.generate_trace(seed=4242, days=2, n_servers=N_SRV)


def test_trace_is_deterministic():
    a = ep.generate_trace(seed=7, days=1, n_servers=N_SRV)
    b = ep.generate_trace(seed=7, days=1, n_servers=N_SRV)
    assert np.array_equal(a["load"], b["load"])
    assert np.array_equal(a["wl"], b["wl"])
    assert np.array_equal(a["failed"], b["failed"])


def test_different_seeds_differ():
    a = ep.generate_trace(seed=7, days=1, n_servers=N_SRV)
    b = ep.generate_trace(seed=8, days=1, n_servers=N_SRV)
    assert not np.array_equal(a["load"], b["load"])


def test_load_in_unit_interval(trace):
    assert trace["load"].min() >= 0.0 and trace["load"].max() <= 1.0


@pytest.mark.parametrize("policy", ["all_on", "observed_greedy", "oracle_greedy",
                                    "observed_opt"])
def test_demand_conservation(trace, policy):
    """Hizmet verilen + düşen istek = toplam talep (talep kaybolmaz / yoktan var olmaz)."""
    r = ep.simulate(policy, trace)
    assert r["served_total"] + r["dropped_total"] == pytest.approx(r["demand_total"], rel=1e-9)
    assert r["dropped_total"] >= 0.0


@pytest.mark.parametrize("policy", ["observed_greedy", "oracle_greedy", "observed_opt"])
def test_simulation_is_deterministic(trace, policy):
    assert ep.simulate(policy, trace) == ep.simulate(policy, trace)


def test_oracle_not_worse_than_reactive(trace):
    """Kusursuz tahmin, aynı seçiciyle reaktiften daha fazla SLA ihlali üretmemeli."""
    for sel in ("greedy", "opt"):
        o = ep.simulate(f"oracle_{sel}", trace)
        r = ep.simulate(f"observed_{sel}", trace)
        assert o["sla_violation_pct"] <= r["sla_violation_pct"] + 1e-9


def test_all_on_uses_most_energy(trace):
    e_all = ep.simulate("all_on", trace)["energy_kwh"]
    for p in ("observed_greedy", "oracle_greedy", "observed_opt"):
        assert ep.simulate(p, trace)["energy_kwh"] <= e_all + 1e-9


def test_active_servers_within_fleet(trace):
    for p in ("all_on", "observed_greedy", "oracle_opt"):
        r = ep.simulate(p, trace)
        assert ep.MIN_ACTIVE - 1e-9 <= r["avg_active_servers"] <= N_SRV + 1e-9


def test_no_future_leakage_in_features():
    """Tablo özellikleri yalnızca t ve öncesini kullanmalı; hedef geleceğe bakmalı."""
    load = np.arange(40, dtype=float) / 40.0
    wl = np.zeros(40, dtype=int)
    X, y, idx = ep.build_tabular(load, wl)
    for row, target, k in zip(X, y, idx):
        assert row[0] == pytest.approx(load[k])                       # en yeni lag = x[t]
        assert target == pytest.approx(load[k + 1:k + 1 + ep.HORIZON].max())
        assert max(row[:ep.N_LAGS]) <= load[k] + 1e-6                 # artan seride gelecek yok


def test_gap_closure_endpoints():
    """BKO: reaktifin kendisi 0, oracle'ın kendisi 1 olmalı."""
    fake = {p: [dict({k: 1.0 for k, _, _ in ep.KEYS}, sla_violation_pct=s)
                for s in vals]
            for p, vals in {"observed_greedy": [2.0, 3.0], "oracle_greedy": [0.0, 0.1],
                            "x": [1.0, 1.5]}.items()}
    s = ep.summarize(fake, list(fake))
    assert s["observed_greedy"]["gap_closure"]["mean"] == pytest.approx(0.0)
    assert s["oracle_greedy"]["gap_closure"]["mean"] == pytest.approx(1.0)
    assert 0.0 < s["x"]["gap_closure"]["mean"] < 1.0


# ── Fable incelemesi sonrası eklenen testler ──────────────────────────────────

def _manual_trace(load):
    load = np.asarray(load, dtype=float)
    n = len(load)
    return {"load": load, "wl": np.zeros(n, dtype=int),
            "failed": np.zeros((n, N_SRV), dtype=bool), "n": n}


def test_boot_delay_is_enforced():
    """Reaktif politika talep sıçramasını t'de görür; yeni sunucu t+1'de henüz
    hizmet vermez (açılış), t+2'de verir → t ve t+1 aşırı yüklü, t+2 değil."""
    k0 = 20
    tr = _manual_trace([0.05] * k0 + [0.95] * 10)
    r = ep.simulate("observed_greedy", tr, record=True)
    rho = r["rho_series"]
    assert rho[k0] > 1.0 and rho[k0 + ep.BOOT_TICKS] > 1.0
    assert rho[k0 + ep.BOOT_TICKS + 1] <= ep.SLA_UTIL


def test_oracle_prewakes_before_step():
    k0 = 20
    tr = _manual_trace([0.05] * k0 + [0.95] * 10)
    r = ep.simulate("oracle_greedy", tr, record=True)
    assert max(r["rho_series"][k0:k0 + 5]) <= ep.SLA_UTIL


def test_scale_in_waits_for_stabilization():
    """Talep düşünce sunucular en az SCALE_IN_STAB_TICKS tick boyunca açık kalır."""
    k0 = 20
    tr = _manual_trace([0.9] * k0 + [0.05] * 15)
    r = ep.simulate("observed_greedy", tr, record=True)
    act = r["active_series"]
    high = act[k0 - 1]
    assert all(a == high for a in act[k0:k0 + ep.SCALE_IN_STAB_TICKS - 1])
    assert act[-1] < high


def test_energy_units_all_on_constant_load():
    x, n = 0.5, 24
    tr = _manual_trace([x] * n)
    r = ep.simulate("all_on", tr)
    fleet = create_server_fleet()
    cap = sum(s.max_requests_per_sec for s in fleet)
    rho = x * ep.DEMAND_SCALE * cap / cap
    watts = 0.0
    for s in fleet:
        s.current_cpu_pct, s.current_temp_c = rho * 100.0, 40.0
        watts += s.compute_power_draw()
    expected_kwh = watts * n * ep.DT_SECONDS / 3600.0 / 1000.0
    assert r["energy_kwh"] == pytest.approx(expected_kwh, rel=1e-9)


def test_failed_server_never_serves():
    tr = _manual_trace([0.3] * 30)
    tr["failed"][:, 0] = True                      # sunucu 0 hep arızalı
    r_all = ep.simulate("all_on", tr, record=True)
    assert max(r_all["active_series"]) <= N_SRV - 1


def test_forecast_policies_run(trace):
    f = ep.Forecaster("naive", None, "naive")
    f.margin = 0.05
    r = ep.simulate("naive_greedy", trace, f)
    assert r["served_total"] + r["dropped_total"] == pytest.approx(r["demand_total"], rel=1e-9)


def test_no_future_leakage_in_sequences():
    load = np.arange(40, dtype=float) / 40.0
    wl = np.zeros(40, dtype=int)
    X, y = ep.build_sequences(load, wl, seq_len=5)
    for i, (seq, target) in enumerate(zip(X, y)):
        k = i + 5 - 1
        assert seq[-1][0] == pytest.approx(load[k])                  # son adım = x[t]
        assert target == pytest.approx(load[k + 1:k + 1 + ep.HORIZON].max())


# ── G2: Pareto analizi yardımcıları ───────────────────────────────────────────

def test_pareto_interpolation():
    import pareto_margin as pm
    de  = np.array([4.0, 0.0, 8.0])           # sırasız giriş
    sla = np.array([1.0, 2.0, 0.0])
    assert pm._interp_sla_at_energy(de, sla, 2.0) == pytest.approx(1.5)
    assert np.isnan(pm._interp_sla_at_energy(de, sla, 9.0))      # aralık dışı → NaN
    assert pm._interp_energy_at_sla(de, sla, 0.5) == pytest.approx(6.0)
    assert np.isnan(pm._interp_energy_at_sla(de, sla, 3.0))


def test_paired_bootstrap_sign():
    import pareto_margin as pm
    rng = np.random.default_rng(0)
    a = np.linspace(1.0, 2.0, 20)
    r = pm._paired(a - 0.5, a, rng)            # her seed'de tam −0.5 fark
    assert r["mean"] == pytest.approx(-0.5)
    assert r["ci95"][0] == pytest.approx(-0.5) and r["ci95"][1] == pytest.approx(-0.5)


# ── G3: doğrusal AR + koşullu pay ─────────────────────────────────────────────

def test_state_feats_no_future_leakage():
    """_state_feats(k) yalnızca x[≤k] kullanmalı: geleceği değiştirmek sonucu değiştirmemeli."""
    rng = np.random.default_rng(0)
    load = rng.random(50)
    k = 20
    a = ep._state_feats(load, k, (1.5, -0.5))
    load2 = load.copy(); load2[k + 1:] = 9.9
    assert ep._state_feats(load2, k, (1.5, -0.5)) == a


def test_fit_ar2_recovers_coefficients():
    rng = np.random.default_rng(1)
    x = np.zeros(3000); x[:2] = 0.5
    for t in range(2, 3000):
        x[t] = 1.5 * x[t - 1] - 0.6 * x[t - 2] + rng.normal(0, 0.01)   # sabit terimsiz AR(2)
    a, b = ep.fit_ar2(x)
    assert a == pytest.approx(1.5, abs=0.05) and b == pytest.approx(-0.6, abs=0.05)


@pytest.fixture(scope="module")
def small_forecasters():
    return {f.name: f for f in ep.train_forecasters(1000, 6, False, N_SRV)}


def test_conditional_margin_cache_matches_direct(small_forecasters, trace):
    f = small_forecasters["linq"]
    f.set_level(0.9)
    load, wl = trace["load"], trace["wl"]
    for k in (3, 10, 100, len(load) - 1):
        feat = np.array([f._feat(load, wl, k)])
        direct = max(0.0, float(f.qmodels[0.9].predict(feat)[0]) + f.shift[0.9])
        assert f.margin_now(load, wl, k) == pytest.approx(direct, abs=1e-9)


def test_conditional_margin_monotone_in_tau(small_forecasters, trace):
    """Ortalama pay τ ile artmalı (kantil düzeni)."""
    f = small_forecasters["linq"]
    means = []
    for tau in (0.5, 0.8, 0.95, 0.99):
        f.set_level(tau)
        means.append(np.mean([f.margin_now(trace["load"], trace["wl"], k)
                              for k in range(len(trace["load"]))]))
    assert all(a <= b + 1e-9 for a, b in zip(means, means[1:]))


def test_new_forecaster_policies_conserve_demand(small_forecasters, trace):
    for name in ("lin", "linq"):
        f = small_forecasters[name]; f.set_level(0.9)
        r = ep.simulate(f"{name}_greedy", trace, f)
        assert r["served_total"] + r["dropped_total"] == pytest.approx(r["demand_total"], rel=1e-9)


def test_end_to_end_future_corruption(small_forecasters, trace):
    """Uçtan uca: t sonrası talep ve iş yükü rastgele bozulunca t'deki tahmin ve pay değişmemeli."""
    rng = np.random.default_rng(3)
    load, wl = trace["load"], trace["wl"]
    for name in ("naive", "rf", "lin", "linq"):
        f = small_forecasters[name]; f.set_level(0.9)
        for k in rng.integers(0, len(load) - 3, size=25):
            l2, w2 = load.copy(), wl.copy()
            l2[k + 1:] = rng.random(len(load) - k - 1)
            w2[k + 1:] = rng.integers(0, len(ep.WORKLOAD_CLASSES), len(load) - k - 1)
            assert f.predict_at(load, wl, k) == pytest.approx(f.predict_at(l2, w2, k), abs=1e-12)
            assert f.margin_now(load, wl, k) == pytest.approx(f.margin_now(l2, w2, k), abs=1e-12)


def test_conditional_margin_early_ticks(small_forecasters, trace):
    f = small_forecasters["linq"]; f.set_level(0.9)
    for k in range(3):
        assert f.margin_now(trace["load"], trace["wl"], k) == pytest.approx(f.margin_at(0.9))


def test_set_level_rejects_off_grid(small_forecasters):
    with pytest.raises(AssertionError):
        small_forecasters["linq"].set_level(0.93)


def test_coverage_close_to_target():
    """Deneydeki ayarla (30 günlük eğitim) CQR sonrası doğrulama kapsaması, üst kantillerde
    hedefin altına %3'ten fazla düşmemeli. (Kısa eğitimde kapsama bozulur — bilinen sınır.)"""
    fs = {f.name: f for f in ep.train_forecasters(1000, 30, False, N_SRV, use_rf=False)}
    f = fs["linq"]
    for tau in (0.8, 0.9, 0.95):
        assert f.coverage[tau] >= tau - 0.03


# ── G5/G6: gerçek iz (Azure) yardımcıları ─────────────────────────────────────

def test_trace_from_series_deterministic_and_shapes():
    x = np.linspace(0.1, 0.9, 500)
    a = ep.trace_from_series(x, None, N_SRV, failure_seed=7)
    b = ep.trace_from_series(x, None, N_SRV, failure_seed=7)
    assert a["n"] == 500 and a["failed"].shape == (500, N_SRV)
    assert np.array_equal(a["failed"], b["failed"]) and np.array_equal(a["load"], x)
    assert (a["wl"] == 0).all()
    r = ep.simulate("observed_greedy", a)
    assert r["served_total"] + r["dropped_total"] == pytest.approx(r["demand_total"], rel=1e-9)


def test_azure_fleets_disjoint_and_train_only_scaling():
    import azure_eval as az
    rng = np.random.default_rng(0)
    D = rng.random((300, 8640)).astype(np.float32) + 0.1
    D[:50, :1000] = 0.0                                # aktif olmayanlar aday olmamalı
    fleets, n_cand = az.build_fleets(D, 5, 10)
    allsubs = np.concatenate(fleets)
    assert len(np.unique(allsubs)) == len(allsubs)      # ayrık
    assert not np.isin(np.arange(50), allsubs).any()
    # ölçek yalnızca eğitim döneminden: test dönemine dev bir sıçrama ölçeği değiştirmemeli
    D2 = D.copy(); D2[fleets[0], -10:] = 1e6
    s1 = az.fleet_series(D, fleets[0])[3]; s2 = az.fleet_series(D2, fleets[0])[3]
    assert s1 == pytest.approx(s2)
    assert az.fleet_series(D2, fleets[0])[1].max() <= az.X_CLIP
