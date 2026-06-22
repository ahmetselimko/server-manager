import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class TrafficPattern(Enum):
    DAILY_BUSINESS  = "daily_business"
    NIGHT_BATCH     = "night_batch"
    FLASH_CROWD     = "flash_crowd"
    SEASONAL_WAVE   = "seasonal_wave"
    STEADY_STATE    = "steady_state"
    SAWTOOTH        = "sawtooth"
    RANDOM_WALK     = "random_walk"
    GRADUAL_RAMP    = "gradual_ramp"
    CASCADING_FAIL  = "cascading_fail"
    WEEKLY_PEAK     = "weekly_peak"
    MAINTENANCE     = "maintenance"
    CAMPAIGN_BURST  = "campaign_burst"


class WorkloadType(Enum):
    """
    İstek yükünün *içeriğini* tanımlar — pattern'den bağımsız.
    Her workload tipi farklı cpu_intensity / memory_pressure / origin
    ve sunucu tercih profili taşır.

    Zaman dağılımı (saat bazlı ağırlıklar):
      BUSINESS    → sabah/öğle iş saatleri   (08–18)
      GAMING      → akşam/gece               (18–02)
      STREAMING   → akşam/gece + hafta sonu  (17–01)
      SOCIAL      → tüm gün, akşam piki      (07–23)
      DATA_BATCH  → gece işlem penceresi     (00–06)
      ML_TRAIN    → gece/hafta sonu          (00–08, 20–24)
      CDN_STATIC  → gündüz + akşam erken     (06–22)
      API_GATEWAY → gündüz + öğleden sonra   (06–20)
    """
    BUSINESS    = "business"       # iş uyg., CRM, ERP, e-mail
    GAMING      = "gaming"         # oyun sunucuları, matchmaking, state sync
    STREAMING   = "streaming"      # video/müzik, yüksek throughput
    SOCIAL      = "social"         # sosyal medya, feed, bildirim
    DATA_BATCH  = "data_batch"     # ETL, rapor, log analiz
    ML_TRAIN    = "ml_train"       # model eğitim/inference — GPU benzeri CPU baskısı
    CDN_STATIC  = "cdn_static"     # statik içerik, cache hit → düşük CPU
    API_GATEWAY = "api_gateway"    # REST/gRPC proxy, auth, rate-limit


# ── Saat × gün tipine göre WorkloadType ağırlık tablosu ──────────────────────
# Her satır: (hour_start, hour_end, is_weekend, {WorkloadType: ağırlık})
# is_weekend: True=sadece hafta sonu, False=sadece hafta içi, None=her gün
_WORKLOAD_SCHEDULE: list = [
    # ── Gece (00–06): Batch ve ML baskın ──────────────────────────────────
    (0,  6,  None, {
        WorkloadType.DATA_BATCH:  0.45,
        WorkloadType.ML_TRAIN:    0.20,
        WorkloadType.GAMING:      0.12,   # gece oyuncuları
        WorkloadType.STREAMING:   0.10,
        WorkloadType.CDN_STATIC:  0.07,
        WorkloadType.API_GATEWAY: 0.04,
        WorkloadType.SOCIAL:      0.02,
        WorkloadType.BUSINESS:    0.00,
    }),
    # ── Sabah (06–09): İş trafiği açılıyor ────────────────────────────────
    (6,  9,  False, {
        WorkloadType.BUSINESS:    0.40,
        WorkloadType.API_GATEWAY: 0.20,
        WorkloadType.CDN_STATIC:  0.15,
        WorkloadType.SOCIAL:      0.12,
        WorkloadType.DATA_BATCH:  0.08,
        WorkloadType.STREAMING:   0.03,
        WorkloadType.GAMING:      0.02,
        WorkloadType.ML_TRAIN:    0.00,
    }),
    (6,  9,  True, {   # hafta sonu sabahı → sosyal + streaming ağır basar
        WorkloadType.SOCIAL:      0.30,
        WorkloadType.STREAMING:   0.25,
        WorkloadType.CDN_STATIC:  0.18,
        WorkloadType.GAMING:      0.12,
        WorkloadType.API_GATEWAY: 0.08,
        WorkloadType.DATA_BATCH:  0.05,
        WorkloadType.BUSINESS:    0.02,
        WorkloadType.ML_TRAIN:    0.00,
    }),
    # ── Gündüz (09–17): İş piki + sosyal ─────────────────────────────────
    (9,  17, False, {
        WorkloadType.BUSINESS:    0.38,
        WorkloadType.API_GATEWAY: 0.22,
        WorkloadType.CDN_STATIC:  0.16,
        WorkloadType.SOCIAL:      0.14,
        WorkloadType.STREAMING:   0.05,
        WorkloadType.DATA_BATCH:  0.03,
        WorkloadType.GAMING:      0.01,
        WorkloadType.ML_TRAIN:    0.01,
    }),
    (9,  17, True, {   # hafta sonu gündüzü → eğlence ağır basar
        WorkloadType.STREAMING:   0.28,
        WorkloadType.GAMING:      0.22,
        WorkloadType.SOCIAL:      0.20,
        WorkloadType.CDN_STATIC:  0.15,
        WorkloadType.API_GATEWAY: 0.08,
        WorkloadType.BUSINESS:    0.04,
        WorkloadType.DATA_BATCH:  0.02,
        WorkloadType.ML_TRAIN:    0.01,
    }),
    # ── Akşam (17–22): Oyun + dizi/film + sosyal zirve ───────────────────
    (17, 22, None, {
        WorkloadType.STREAMING:   0.30,
        WorkloadType.GAMING:      0.28,
        WorkloadType.SOCIAL:      0.20,
        WorkloadType.CDN_STATIC:  0.10,
        WorkloadType.API_GATEWAY: 0.07,
        WorkloadType.BUSINESS:    0.03,
        WorkloadType.DATA_BATCH:  0.01,
        WorkloadType.ML_TRAIN:    0.01,
    }),
    # ── Gece başı (22–00): Oyun + geç streaming ──────────────────────────
    (22, 24, None, {
        WorkloadType.GAMING:      0.35,
        WorkloadType.STREAMING:   0.28,
        WorkloadType.SOCIAL:      0.18,
        WorkloadType.CDN_STATIC:  0.08,
        WorkloadType.API_GATEWAY: 0.05,
        WorkloadType.DATA_BATCH:  0.04,
        WorkloadType.ML_TRAIN:    0.02,
        WorkloadType.BUSINESS:    0.00,
    }),
]

# ── Workload → sunucu tercihi eşlemesi ───────────────────────────────────────
# Her workload hangi sunucu özelliklerini tercih eder?
# Değer: (özellik, yön) — autoscaler ve metrics_collector bu ipucunu kullanır.
WORKLOAD_SERVER_AFFINITY: dict = {
    # Düşük latency kritik: edge sunucular (srv-edge-*)
    WorkloadType.GAMING:      {"preferred_arch": None,  "prefer_low_latency": True,
                                "prefer_high_clock": True,  "prefer_many_cores": False},
    # Yüksek throughput: ARM (Graviton/Altra) veya çok çekirdekli
    WorkloadType.STREAMING:   {"preferred_arch": "ARM", "prefer_low_latency": False,
                                "prefer_high_clock": False, "prefer_many_cores": True},
    # Dengeli: geniş bellek, orta latency
    WorkloadType.BUSINESS:    {"preferred_arch": None,  "prefer_low_latency": False,
                                "prefer_high_clock": False, "prefer_many_cores": False},
    # Yüksek çekirdek sayısı + bellek: EPYC veya ARM
    WorkloadType.DATA_BATCH:  {"preferred_arch": None,  "prefer_low_latency": False,
                                "prefer_high_clock": False, "prefer_many_cores": True},
    # CPU yoğun + yüksek bellek: Xeon Platinum veya EPYC Genoa
    WorkloadType.ML_TRAIN:    {"preferred_arch": None,  "prefer_low_latency": False,
                                "prefer_high_clock": True,  "prefer_many_cores": True},
    # Düşük CPU, yüksek network throughput: ARM enerji verimli
    WorkloadType.CDN_STATIC:  {"preferred_arch": "ARM", "prefer_low_latency": False,
                                "prefer_high_clock": False, "prefer_many_cores": True},
    # Orta latency, orta CPU: API gateway için Xeon veya EPYC
    WorkloadType.API_GATEWAY: {"preferred_arch": None,  "prefer_low_latency": True,
                                "prefer_high_clock": False, "prefer_many_cores": False},
    # Sosyal: ani spike'lar → düşük latency + burst kapasitesi
    WorkloadType.SOCIAL:      {"preferred_arch": None,  "prefer_low_latency": True,
                                "prefer_high_clock": True,  "prefer_many_cores": False},
}


# train.py ve models.py için integer etiket → string eşlemesi
# Sıra önemli: WL_TO_INT = {wl: i for i, wl in enumerate(WORKLOAD_CLASSES)}
WORKLOAD_CLASSES = [
    "gaming",
    "streaming",
    "business",
    "social",
    "data_batch",
    "ml_train",
    "cdn_static",
    "api_gateway",
]


@dataclass
class TrafficEvent:
    """A discrete traffic event sent to a server."""
    timestamp: float
    pattern: TrafficPattern
    workload_type: WorkloadType
    request_rate: float
    request_size_kb: float
    cpu_intensity: float
    memory_pressure: float
    burst_duration_sec: float
    origin: str
    # Workload kaynaklı sunucu karakteristikleri
    # metrics_collector ve autoscaler bu alanları kullanır
    latency_sensitivity: float = 0.5   # 0-1; gaming yüksek, batch düşük
    bandwidth_pressure: float  = 0.3   # 0-1; streaming yüksek, gaming düşük


class TrafficGenerator:
    """
    Gerçekçi, çok katmanlı trafik deseni üreteci.

    Zaman katmanları:
      - Günlük  : saat bazlı yük (sabah piki, öğle yavaşlaması, akşam piki,
                  gece taban yükü)
      - Haftalık: Pazartesi başlangıç yoğunluğu (hafta sonu multiplier yok)
      - Aylık   : bakım penceresi, kampanya günleri (otomatik yenilenir)
    """

    SECONDS_PER_DAY  = 86_400
    SECONDS_PER_WEEK = 604_800

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.active_pattern = TrafficPattern.DAILY_BUSINESS

        # steady_state gerçek dünyada baskın pattern; flash_crowd nadir.
        self.pattern_weights = {
            TrafficPattern.DAILY_BUSINESS: 0.28,
            TrafficPattern.STEADY_STATE:   0.22,
            TrafficPattern.NIGHT_BATCH:    0.10,
            TrafficPattern.SEASONAL_WAVE:  0.09,
            TrafficPattern.RANDOM_WALK:    0.08,
            TrafficPattern.SAWTOOTH:       0.07,
            TrafficPattern.WEEKLY_PEAK:    0.06,
            TrafficPattern.GRADUAL_RAMP:   0.04,
            TrafficPattern.CAMPAIGN_BURST: 0.03,
            TrafficPattern.MAINTENANCE:    0.02,
            TrafficPattern.FLASH_CROWD:    0.01,  # gerçekte çok nadir
        }
        # CASCADING_FAIL ağırlıklı havuzda yok; yalnızca select_pattern_for_tick
        # içindeki koşullu bloktan tetiklenir.

        self._random_walk_state: Optional[float] = None  # ilk çağrıda saat bazlı init

        # Flash crowd: bir sonraki olayın zamanı ve büyüklüğü
        self._next_flash_t: float = float('inf')
        self._flash_peak: float = 0.0
        self._flash_duration: float = 0.0

        # Cascading fail durumu
        self._cascade_start_t: Optional[float] = None
        self._cascade_depth: float = 0.0   # 0–1 arası çöküş derinliği

        # Kampanya günleri
        self._campaign_days: set = set()
        self._last_campaign_month: int = -1
        self._maintenance_day_of_month: int = 7
        self._refresh_campaign_days_if_needed(t=0.0)

    # ──────────────────────────────────────────
    # WorkloadType seçimi
    # ──────────────────────────────────────────

    def select_workload_type(self, t: float) -> WorkloadType:
        """
        Saati ve hafta içi/sonu durumunu kullanarak ağırlıklı WorkloadType seçer.
        _WORKLOAD_SCHEDULE tablosundaki ilk eşleşen satırı kullanır.
        """
        hour       = self._hour_of_day(t)
        is_weekend = not self._is_weekday(t)

        for h_start, h_end, wend_flag, weights in _WORKLOAD_SCHEDULE:
            if not (h_start <= hour < h_end):
                continue
            if wend_flag is not None and wend_flag != is_weekend:
                continue
            # Eşleşen satır bulundu
            wl_types = list(weights.keys())
            wl_probs = np.array([weights[w] for w in wl_types], dtype=float)
            wl_probs /= wl_probs.sum()
            return self.rng.choice(wl_types, p=wl_probs)

        # Varsayılan
        return WorkloadType.API_GATEWAY

    # ──────────────────────────────────────────
    # Workload bazlı event parametreleri
    # ──────────────────────────────────────────

    def _workload_event_params(self, wl: WorkloadType) -> dict:
        """
        Her WorkloadType için gerçekçi cpu_intensity, memory_pressure,
        request_size_kb ve origin döndürür.

        Mantık:
          GAMING      → düşük CPU/memory (state sync küçük), çok düşük latency
          STREAMING   → düşük CPU ama yüksek throughput (büyük paket)
          BUSINESS    → orta CPU + memory (DB sorguları, ORM)
          SOCIAL      → spike'lı CPU (feed hesaplama), orta memory
          DATA_BATCH  → yüksek CPU + yüksek memory (büyük dataset işleme)
          ML_TRAIN    → çok yüksek CPU + yüksek memory (matris çarpımı)
          CDN_STATIC  → çok düşük CPU (cache hit), büyük paket
          API_GATEWAY → orta CPU (auth+rate-limit), küçük paket
        """
        r = self.rng
        params = {
            WorkloadType.GAMING: dict(
                cpu_intensity        = r.uniform(0.15, 0.42),
                memory_pressure      = r.uniform(0.12, 0.35),
                request_size_kb      = r.uniform(0.5,  12.0),
                origin               = "gameserver",
                latency_sensitivity  = 0.92,   # frame drop → kullanıcı hisseder
                bandwidth_pressure   = 0.12,   # küçük state paketleri
            ),
            WorkloadType.STREAMING: dict(
                cpu_intensity        = r.uniform(0.10, 0.30),
                memory_pressure      = r.uniform(0.35, 0.65),
                request_size_kb      = r.uniform(200, 2000),
                origin               = "cdn",
                latency_sensitivity  = 0.20,   # 2-3s buffer var
                bandwidth_pressure   = 0.92,   # büyük video chunk
            ),
            WorkloadType.BUSINESS: dict(
                cpu_intensity        = r.uniform(0.35, 0.72),
                memory_pressure      = r.uniform(0.30, 0.62),
                request_size_kb      = r.uniform(5,   80),
                origin               = "api",
                latency_sensitivity  = 0.55,
                bandwidth_pressure   = 0.30,
            ),
            WorkloadType.SOCIAL: dict(
                cpu_intensity        = r.uniform(0.30, 0.78),
                memory_pressure      = r.uniform(0.28, 0.58),
                request_size_kb      = r.uniform(2,   40),
                origin               = "api",
                latency_sensitivity  = 0.50,
                bandwidth_pressure   = 0.40,
            ),
            WorkloadType.DATA_BATCH: dict(
                cpu_intensity        = r.uniform(0.70, 0.97),
                memory_pressure      = r.uniform(0.60, 0.92),
                request_size_kb      = r.uniform(100, 800),
                origin               = "batch",
                latency_sensitivity  = 0.04,   # latency önemsiz
                bandwidth_pressure   = 0.45,
            ),
            WorkloadType.ML_TRAIN: dict(
                cpu_intensity        = r.uniform(0.80, 0.99),
                memory_pressure      = r.uniform(0.65, 0.95),
                request_size_kb      = r.uniform(50,  500),
                origin               = "internal",
                latency_sensitivity  = 0.05,
                bandwidth_pressure   = 0.50,
            ),
            WorkloadType.CDN_STATIC: dict(
                cpu_intensity        = r.uniform(0.04, 0.18),
                memory_pressure      = r.uniform(0.10, 0.30),
                request_size_kb      = r.uniform(50,  1500),
                origin               = "cdn",
                latency_sensitivity  = 0.30,
                bandwidth_pressure   = 0.85,   # büyük statik dosyalar
            ),
            WorkloadType.API_GATEWAY: dict(
                cpu_intensity        = r.uniform(0.20, 0.55),
                memory_pressure      = r.uniform(0.18, 0.45),
                request_size_kb      = r.uniform(1,   30),
                origin               = "api",
                latency_sensitivity  = 0.65,   # API timeout kritik
                bandwidth_pressure   = 0.20,
            ),
        }
        return params[wl]

    # ──────────────────────────────────────────
    # Yardımcı zaman fonksiyonları
    # ──────────────────────────────────────────

    @staticmethod
    def _hour_of_day(t: float) -> float:
        return (t / 3600.0) % 24

    @staticmethod
    def _day_of_week(t: float) -> int:
        """0=Pazartesi … 6=Pazar"""
        return int(t / 86400.0) % 7

    @staticmethod
    def _day_of_month(t: float) -> int:
        """1-based, 30 günlük ay varsayımı."""
        return (int(t / 86400.0) % 30) + 1

    @staticmethod
    def _month_index(t: float) -> int:
        return int(t / 86400.0) // 30

    def _is_weekday(self, t: float) -> bool:
        return self._day_of_week(t) < 5

    def _day_multiplier(self, t: float) -> float:
        """
        Günlük haftalık çarpan — yalnızca Pazartesi için küçük bir başlangıç
        bonusu; hafta sonu keskin düşüş YOK (modern SaaS/B2C trafiği).
          Pazartesi  → 1.10
          Salı-Cuma  → 1.00
          Cumartesi  → 0.95
          Pazar      → 0.92
        """
        dow = self._day_of_week(t)
        multipliers = [1.10, 1.00, 1.00, 1.00, 1.00, 0.95, 0.92]
        return multipliers[dow]

    def _is_maintenance_window(self, t: float) -> bool:
        return (self._day_of_month(t) == self._maintenance_day_of_month
                and 2.0 <= self._hour_of_day(t) < 4.0)

    def _is_campaign_day(self, t: float) -> bool:
        return self._day_of_month(t) in self._campaign_days

    def _refresh_campaign_days_if_needed(self, t: float) -> None:
        """Her ay başında kampanya günlerini yenile."""
        mi = self._month_index(t)
        if mi == self._last_campaign_month:
            return
        self._last_campaign_month = mi
        days = [d for d in range(1, 31) if d != self._maintenance_day_of_month]
        chosen = self.rng.choice(days, size=2, replace=False)
        self._campaign_days = set(chosen.tolist())

    # ──────────────────────────────────────────
    # Pattern fonksiyonları
    # ──────────────────────────────────────────

    def _daily_business_load(self, t: float) -> float:
        """
        Günlük iş yükü — saat bazlı keskin geçişler.

        Saat segmentleri (iş günü):
          00–06 : gece taban  (~0.15)
          06–09 : sabah rampi (0.15 → 0.75)
          09–12 : sabah piki  (0.75–0.85)
          12–13 : öğle yavaşlaması (→ 0.60)
          13–18 : öğleden sonra piki (0.65–0.82)
          18–21 : akşam düşüşü (0.82 → 0.30)
          21–24 : gece taban  (~0.18)
        """
        hour = self._hour_of_day(t)
        dm   = self._day_multiplier(t)
        r    = self.rng.random()

        if hour < 6.0:
            base = 0.14 + 0.03 * r
        elif hour < 9.0:
            # Sabah rampi: yumuşak S-eğrisi
            p    = (hour - 6.0) / 3.0
            base = 0.15 + 0.62 * (3 * p**2 - 2 * p**3) + 0.03 * r
        elif hour < 12.0:
            base = 0.75 + 0.10 * np.sin((hour - 9.0) / 3.0 * np.pi) + 0.04 * r
        elif hour < 13.0:
            # Öğle yavaşlaması — hafif çukur, sert değil
            p    = (hour - 12.0) / 1.0
            base = 0.77 - 0.17 * np.sin(p * np.pi) + 0.03 * r
        elif hour < 18.0:
            base = 0.68 + 0.14 * np.sin((hour - 13.0) / 5.0 * np.pi) + 0.04 * r
        elif hour < 21.0:
            p    = (hour - 18.0) / 3.0
            base = 0.82 - 0.64 * (3 * p**2 - 2 * p**3) + 0.03 * r
        else:
            base = 0.17 + 0.04 * r

        return float(np.clip(base * dm, 0.0, 1.0))

    def _night_batch_load(self, t: float) -> float:
        """
        Gece batch işleri — iki dalgalı yapı:

          Birinci dalga (ana batch):
            00:00–00:30 : ön hazırlık ısınması (0.15 → 0.45)
            00:30–01:00 : hızlı tırmanma (0.45 → 0.78)
            01:00–04:30 : yüksek yük, hafif sinüs dalgalanması (0.76–0.90)
            04:30–05:30 : S-eğrisi soğuma (0.78 → 0.22)
            05:30–05:45 : cleanup spike — log flush, indeks yenileme (0.55–0.70)
            05:45–06:00 : cleanup sonrası yumuşak iniş (→ 0.12)

          İkinci dalga (sabah ön işleme — rapor üretimi, veri senkronu):
            06:30–07:00 : kısa ısınma (0.12 → 0.42)
            07:00–07:45 : ikinci doruk (0.40–0.52)
            07:45–08:15 : hızlı kapanış — mesai başlamadan önce tamamlanır (→ 0.10)

          Diğer saatler: düşük taban (~0.06–0.09)
        """
        hour = self._hour_of_day(t)
        r    = self.rng.random()

        # ── Birinci dalga ──────────────────────────────
        if hour < 0.5:
            # Ön hazırlık: hafif ısınma
            p    = hour / 0.5
            base = 0.15 + 0.30 * p
        elif hour < 1.0:
            # Hızlı tırmanma — S-eğrisi
            p    = (hour - 0.5) / 0.5
            base = 0.45 + 0.33 * (3 * p**2 - 2 * p**3)
        elif hour < 4.5:
            # Ana yüksek yük: sinüs dalgalanması (iş parçacığı iniş-çıkışı)
            progress = (hour - 1.0) / 3.5
            base     = 0.78 + 0.12 * np.sin(progress * np.pi)
            # Saat başı küçük dip — checkpoint/flush
            frac = hour - int(hour)
            if frac < 0.08:
                base -= 0.07 * (1.0 - frac / 0.08)
        elif hour < 5.5:
            # Soğuma: S-eğrisi (daha belirgin kapanış)
            p    = (hour - 4.5) / 1.0
            base = 0.78 - 0.56 * (3 * p**2 - 2 * p**3)
        elif hour < 5.75:
            # Cleanup spike — log flush, indeks yenileme, temp file temizliği
            spike_p = (hour - 5.5) / 0.25
            # Keskin zirve ortada
            base = 0.55 + 0.18 * np.sin(spike_p * np.pi)
        elif hour < 6.0:
            # Spike sonrası yumuşak iniş
            p    = (hour - 5.75) / 0.25
            base = 0.55 - 0.43 * (3 * p**2 - 2 * p**3)

        # ── İki dalga arası boşluk ─────────────────────
        elif hour < 6.5:
            base = 0.10 + 0.04 * r

        # ── İkinci dalga (sabah ön işleme) ────────────
        elif hour < 7.0:
            # Kısa ısınma
            p    = (hour - 6.5) / 0.5
            base = 0.10 + 0.35 * (3 * p**2 - 2 * p**3)
        elif hour < 7.75:
            # İkinci doruk — birinciden belirgin şekilde alçak
            progress = (hour - 7.0) / 0.75
            base     = 0.42 + 0.12 * np.sin(progress * np.pi)
        elif hour < 8.25:
            # Hızlı kapanış — mesai başlamadan önce tamamlanmalı
            p    = (hour - 7.75) / 0.5
            base = 0.42 - 0.32 * (3 * p**2 - 2 * p**3)

        # ── Gün içi taban ─────────────────────────────
        else:
            base = 0.06 + 0.03 * r

        return float(np.clip(base + 0.025 * self.rng.random(), 0.0, 1.0))

    def _flash_crowd_load(self, t: float) -> float:
        """
        Ani trafik patlaması — tek seferlik, büyüklüğü ve süresi rastgele.
        Olay schedule_flash_crowd() ile tetiklenir; aralarında taban yük döner.
        """
        if t < self._next_flash_t:
            # Olay öncesi: taban yük
            return float(0.15 + 0.05 * self.rng.random())

        elapsed = t - self._next_flash_t
        if elapsed > self._flash_duration:
            # Olay bitti, bir sonrakini planla
            gap = self.rng.uniform(3600, 14400)   # 1–4 saat sonra
            self._next_flash_t  = t + gap
            self._flash_peak    = float(self.rng.uniform(0.75, 0.98))
            self._flash_duration = float(self.rng.uniform(600, 3600))
            return float(0.15 + 0.05 * self.rng.random())

        # Olay devam ediyor: hızlı tırmanma, üstel düşüş
        rise_time = self._flash_duration * 0.08
        if elapsed < rise_time:
            load = 0.15 + (elapsed / rise_time) * (self._flash_peak - 0.15)
        else:
            decay = (elapsed - rise_time) / (self._flash_duration - rise_time)
            load  = self._flash_peak * np.exp(-3.5 * decay)
            load  = max(load, 0.15)
        return float(np.clip(load + 0.02 * self.rng.random(), 0.0, 1.0))

    def schedule_flash_crowd(self, first_t: float = 0.0) -> None:
        """Flash crowd için ilk olayı zamanla (opsiyonel dış çağrı)."""
        delay = self.rng.uniform(1800, 7200)
        self._next_flash_t   = first_t + delay
        self._flash_peak     = float(self.rng.uniform(0.75, 0.98))
        self._flash_duration = float(self.rng.uniform(600, 3600))

    def _seasonal_wave_load(self, t: float) -> float:
        """7 günlük haftalık periyot + 30 günlük aylık periyot."""
        weekly  = 0.5 + 0.3  * np.sin(2 * np.pi * t / self.SECONDS_PER_WEEK)
        monthly = 0.5 + 0.20 * np.sin(2 * np.pi * t / (self.SECONDS_PER_DAY * 30))
        return float(np.clip(0.20 + 0.40 * weekly + 0.20 * monthly, 0.0, 1.0))

    def _steady_state_load(self, t: float) -> float:
        """
        Sabit taban yükü — günlük ritme küçük oranda duyarlı.
        Gerçekte steady_state hiçbir zaman tamamen saate kör değildir.
        """
        hour  = self._hour_of_day(t)
        dm    = self._day_multiplier(t)
        # Hafif günlük salınım: gündüz biraz daha yüksek
        diurnal = 0.08 * np.sin((hour - 6.0) / 24.0 * 2 * np.pi)
        base    = np.clip(0.45 * dm + diurnal, 0.20, 0.85)
        noise   = 0.03 * self.rng.standard_normal()
        return float(np.clip(base + noise, 0.0, 1.0))

    def _sawtooth_load(self, t: float) -> float:
        """
        Sawtooth — retry/timeout döngüsü gibi kısa periyotlu tırmanma/düşme.
        Periyot sabittir (300 sn) ama bu pattern zaten sentetik/yapay bir yük
        simülasyonunu temsil etmektedir.
        """
        phase = (t % 300) / 300.0
        if phase < 0.30:
            return float(np.clip(0.15 + phase / 0.30 * 0.75, 0.0, 1.0))
        return float(max(0.15, 0.90 - (phase - 0.30) / 0.70 * 0.75))

    def _random_walk_load(self, t: float) -> float:
        """
        Rastgele yürüyüş — gün içi taban yük etrafında geziniyor.
        İlk çağrıda mevcut saate göre initialize edilir.
        """
        if self._random_walk_state is None:
            # Saat bazlı başlangıç değeri
            hour = self._hour_of_day(t)
            self._random_walk_state = float(
                np.clip(0.20 + 0.35 * np.sin((hour - 6.0) / 24.0 * 2 * np.pi), 0.10, 0.80)
            )

        # Taban yüke doğru hafif geri çekim (mean reversion)
        hour   = self._hour_of_day(t)
        target = float(np.clip(0.35 + 0.20 * np.sin((hour - 6.0) / 24.0 * 2 * np.pi), 0.15, 0.75))
        step   = self.rng.standard_normal() * 0.025
        pull   = 0.05 * (target - self._random_walk_state)
        self._random_walk_state = float(
            np.clip(self._random_walk_state + step + pull, 0.05, 0.95)
        )
        return self._random_walk_state

    def _gradual_ramp_load(self, t: float) -> float:
        """
        Kademeli tırmanma — her gün sıfırlanır (deployment sonrası ısınma gibi).
        Gün içinde 06:00'dan itibaren tırmanır, 20:00'de doruğa ulaşır.
        """
        hour     = self._hour_of_day(t)
        dm       = self._day_multiplier(t)
        ramp_start = 6.0
        ramp_end   = 20.0
        if hour < ramp_start:
            progress = 0.0
        elif hour >= ramp_end:
            progress = 1.0
        else:
            progress = (hour - ramp_start) / (ramp_end - ramp_start)
        base = 0.12 + progress * 0.72
        return float(np.clip(base * dm + 0.02 * self.rng.random(), 0.0, 1.0))

    def _cascading_fail_load(self, t: float) -> float:
        """
        Kademeli servis çöküşü — gerçekçi basamak yapısı ve sigmoid toparlanma:

          Faz 0 ( 0– 5 dk) : tetikleyici spike → hızlı tırmanma
          Faz 1 ( 5–20 dk) : basamaklı düşüş — her downstream servis ayrı
                              bir basamakta düşer (3–4 belirgin basamak)
          Faz 2 (20–60 dk) : çöküş tabanı — düşük, osilatif, kaotik
          Faz 3 (60–180 dk): sigmoid toparlanma (hız = cascade_depth'e göre)
          Sonrasında        : normal taban yüke dönüş
        """
        if self._cascade_start_t is None:
            self._cascade_start_t = t
            self._cascade_depth   = float(self.rng.uniform(0.55, 0.90))
            # Basamak sayısını ve konumlarını olay başında sabitle
            n_steps = self.rng.integers(3, 6)   # 3–5 basamak
            self._cascade_steps: list = sorted(
                self.rng.uniform(5.0, 19.0, size=n_steps).tolist()
            )
            self._cascade_step_drops: list = [
                float(self.rng.uniform(0.10, 0.22)) for _ in range(n_steps)
            ]

        elapsed_min = (t - self._cascade_start_t) / 60.0
        depth       = self._cascade_depth

        if elapsed_min < 5:
            # Faz 0: tetikleyici spike — hızlı, keskin tırmanma
            p    = elapsed_min / 5.0
            load = 0.45 + depth * 0.52 * (3 * p**2 - 2 * p**3)
            load += self.rng.uniform(-0.03, 0.05)

        elif elapsed_min < 20:
            # Faz 1: basamaklı düşüş
            # Tepe değerden başla, her basamak noktasında ani düşüş ekle
            peak = 0.45 + depth * 0.52
            # Temel eğim: basamaklar arası hafif lineer düşüş
            p    = (elapsed_min - 5.0) / 15.0
            load = peak - depth * 0.15 * p   # yavaş zemin eğimi

            # Belirlenen basamak noktalarında kümülatif ani düşüşler
            for step_t, drop in zip(self._cascade_steps, self._cascade_step_drops):
                if elapsed_min >= step_t:
                    # Basamak: ani düşüş + hafif "zıplama" yok — düz kesme
                    load -= drop
                elif elapsed_min > step_t - 0.5:
                    # Basamak öncesi küçük sarsıntı (servis timeout uyarıları)
                    load += self.rng.uniform(-0.06, 0.04)

            load += self.rng.uniform(-0.04, 0.04)

        elif elapsed_min < 60:
            # Faz 2: çöküş tabanı — düşük, osilatif, kaotik
            # Hafif osilatif davranış: kısmen yeniden gelen trafik → hata döngüsü
            osc  = 0.07 * np.sin(elapsed_min / 4.5 * np.pi)
            load = 0.10 + depth * 0.08 + osc + self.rng.uniform(-0.04, 0.08)

        elif elapsed_min < 180:
            # Faz 3: sigmoid toparlanma
            # Hızlı başlangıç, yavaş bitiş — servisler sırayla geri gelir
            p        = (elapsed_min - 60.0) / 120.0
            # Derin çöküşler daha yavaş toparlanır
            recovery_speed = 1.0 + (1.0 - depth) * 1.5
            sig      = 1.0 / (1.0 + np.exp(-recovery_speed * (p - 0.35) * 8))
            taban    = 0.10 + depth * 0.08
            hedef    = self._daily_business_load(t)
            load     = taban + (hedef - taban) * sig
            load     += self.rng.uniform(-0.03, 0.05)

        else:
            # Toparlanma tamamlandı — durumu sıfırla
            self._cascade_start_t = None
            self._cascade_depth   = 0.0
            load = self._daily_business_load(t)

        return float(np.clip(load, 0.0, 1.0))

    def _weekly_peak_load(self, t: float) -> float:
        """
        Pazartesi sabahı 08:00–13:00 geniş omuz → yüksek yük.
        Ay başı (1. gün) → benzer ama biraz daha dar spike.
        """
        hour = self._hour_of_day(t)
        dow  = self._day_of_week(t)
        dom  = self._day_of_month(t)
        dm   = self._day_multiplier(t)

        is_monday_morning = (dow == 0 and 7.5 <= hour < 13.0)
        is_month_start    = (dom == 1 and 8.0 <= hour < 12.0)

        if is_monday_morning:
            # Geniş omuz: 08:30 civarında zirve
            p    = (hour - 7.5) / 5.5
            peak = 0.78 + 0.15 * np.sin(p * np.pi)
            return float(np.clip(peak * dm + self.rng.random() * 0.04, 0.0, 1.0))

        if is_month_start:
            p    = (hour - 8.0) / 4.0
            peak = 0.82 + 0.12 * np.sin(p * np.pi)
            return float(np.clip(peak * dm + self.rng.random() * 0.04, 0.0, 1.0))

        return self._daily_business_load(t)

    def _maintenance_load(self, t: float) -> float:
        """
        Bakım penceresi (02:00–04:00):
          - Başlangıç ve bitiş: restart spike (bağlantı havuzu yenileme)
          - Ortası: çok düşük yük
        Pencere dışında çağrılırsa taban yük döner.
        """
        hour = self._hour_of_day(t)
        if not (2.0 <= hour < 4.0):
            return float(0.10 + 0.05 * self.rng.random())

        minutes_into = (hour - 2.0) * 60
        if minutes_into <= 6 or minutes_into >= 114:
            return float(np.clip(0.60 + self.rng.random() * 0.25, 0.0, 1.0))
        return float(np.clip(0.07 + self.rng.random() * 0.06, 0.0, 1.0))

    def _campaign_burst_load(self, t: float) -> float:
        """
        Kampanya/etkinlik günü: 10:00–22:00 yüksek yük.
        Günlük çarpan kullanılır; hafta sonu penaltısı yok.
        """
        hour = self._hour_of_day(t)
        dm   = self._day_multiplier(t)

        if 10.0 <= hour <= 22.0:
            burst = 0.78 + 0.16 * np.sin((hour - 10.0) / 12.0 * np.pi)
            return float(np.clip(burst * dm + self.rng.random() * 0.04, 0.0, 1.0))
        # Kampanya saatleri dışı: yükseltilmiş taban
        return float(np.clip(0.32 * dm + self.rng.random() * 0.04, 0.0, 1.0))

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def get_load_for_pattern(self, pattern: TrafficPattern, t: float) -> float:
        self._refresh_campaign_days_if_needed(t)
        dispatch = {
            TrafficPattern.DAILY_BUSINESS: self._daily_business_load,
            TrafficPattern.NIGHT_BATCH:    self._night_batch_load,
            TrafficPattern.FLASH_CROWD:    self._flash_crowd_load,
            TrafficPattern.SEASONAL_WAVE:  self._seasonal_wave_load,
            TrafficPattern.STEADY_STATE:   self._steady_state_load,
            TrafficPattern.SAWTOOTH:       self._sawtooth_load,
            TrafficPattern.RANDOM_WALK:    self._random_walk_load,
            TrafficPattern.GRADUAL_RAMP:   self._gradual_ramp_load,
            TrafficPattern.CASCADING_FAIL: self._cascading_fail_load,
            TrafficPattern.WEEKLY_PEAK:    self._weekly_peak_load,
            TrafficPattern.MAINTENANCE:    self._maintenance_load,
            TrafficPattern.CAMPAIGN_BURST: self._campaign_burst_load,
        }
        return float(np.clip(dispatch[pattern](t), 0.0, 1.0))

    def generate_event(self, server_id: str, t: float, pattern: TrafficPattern,
                       base_load: float, per_server_noise: float = 0.05,
                       workload_type: Optional[WorkloadType] = None) -> TrafficEvent:
        self._refresh_campaign_days_if_needed(t)
        noise = self.rng.standard_normal() * per_server_noise
        load  = float(np.clip(base_load + noise, 0.0, 1.0))

        # WorkloadType seçimi — dışarıdan verilmemişse zaman bazlı seç
        if workload_type is None:
            workload_type = self.select_workload_type(t)

        # ── Pattern bazlı overrides (bazı pattern'lar workload'u ezer) ──────
        # Maintenance ve cascading_fail kendi karakteristiklerini korur.
        if pattern == TrafficPattern.MAINTENANCE:
            origin          = "internal"
            size_kb         = self.rng.uniform(1, 12)
            cpu_intensity   = self.rng.uniform(0.05, 0.30)
            memory_pressure = self.rng.uniform(0.10, 0.38)
            workload_type   = WorkloadType.DATA_BATCH   # en yakın semantik
        elif pattern == TrafficPattern.CASCADING_FAIL:
            origin          = "api"
            size_kb         = self.rng.uniform(1, 40)
            cpu_intensity   = self.rng.uniform(0.05, 0.95)   # kaotik
            memory_pressure = self.rng.uniform(0.05, 0.95)
            # workload değişmez — hangi workload çöktüğü korunur
        elif pattern == TrafficPattern.NIGHT_BATCH:
            # Gece batch her zaman DATA_BATCH semantiği taşır
            workload_type   = WorkloadType.DATA_BATCH
            p = self._workload_event_params(workload_type)
            origin          = p["origin"]
            size_kb         = p["request_size_kb"]
            cpu_intensity   = p["cpu_intensity"]
            memory_pressure = p["memory_pressure"]
        else:
            # Normal akış: workload parametrelerini kullan
            p = self._workload_event_params(workload_type)
            origin          = p["origin"]
            size_kb         = p["request_size_kb"]
            cpu_intensity   = p["cpu_intensity"]
            memory_pressure = p["memory_pressure"]

            # Flash crowd ve campaign_burst için hafif yoğunluk artışı
            if pattern in (TrafficPattern.FLASH_CROWD, TrafficPattern.CAMPAIGN_BURST,
                           TrafficPattern.WEEKLY_PEAK):
                cpu_intensity   = float(np.clip(cpu_intensity   * 1.25, 0.0, 1.0))
                memory_pressure = float(np.clip(memory_pressure * 1.20, 0.0, 1.0))

        # Workload'a özgü latency/bandwidth değerlerini al
        _wp = self._workload_event_params(workload_type)

        return TrafficEvent(
            timestamp=t,
            pattern=pattern,
            workload_type=workload_type,
            request_rate=load,
            request_size_kb=float(size_kb),
            cpu_intensity=float(cpu_intensity),
            memory_pressure=float(memory_pressure),
            burst_duration_sec=self.rng.uniform(10, 120),
            origin=origin,
            latency_sensitivity=float(_wp["latency_sensitivity"]),
            bandwidth_pressure =float(_wp["bandwidth_pressure"]),
        )

    def select_workload_for_tick(self, t: float) -> WorkloadType:
        """
        metrics_collector ve train.py tarafından kullanılan alias.
        select_workload_type() ile aynı işlevi görür.
        """
        return self.select_workload_type(t)

    def select_pattern_for_tick(self, t: float) -> TrafficPattern:
        """
        Zaman bağlamlı pattern seçimi.

        Öncelik sırası:
          1. Bakım penceresi
          2. Pazartesi/ay-başı sabah piki
          3. Kampanya günü (gündüz saatleri)
          4. Gece batch penceresi
          5. Flash crowd (düşük olasılık, tek olay tabanlı)
          6. Cascading fail (gündüz, düşük olasılık)
          7. Ağırlıklı rastgele havuz (hafta sonu ayrımı YOK)
        """
        self._refresh_campaign_days_if_needed(t)
        hour = self._hour_of_day(t)
        dow  = self._day_of_week(t)
        dom  = self._day_of_month(t)

        # 1. Bakım penceresi
        if self._is_maintenance_window(t) and self.rng.random() < 0.88:
            return TrafficPattern.MAINTENANCE

        # 2. Pazartesi sabahı piki (geniş pencere: 07:30–13:00)
        if dow == 0 and 7.5 <= hour < 13.0 and self.rng.random() < 0.55:
            return TrafficPattern.WEEKLY_PEAK

        # 3. Ay başı piki
        if dom == 1 and 8.0 <= hour < 12.0 and self.rng.random() < 0.48:
            return TrafficPattern.WEEKLY_PEAK

        # 4. Kampanya günü (gündüz)
        if self._is_campaign_day(t) and 10.0 <= hour <= 22.0 and self.rng.random() < 0.58:
            return TrafficPattern.CAMPAIGN_BURST

        # 5. Gece batch (00:00–05:30)
        if hour < 5.5 and self.rng.random() < 0.72:
            return TrafficPattern.NIGHT_BATCH

        # 6. Flash crowd — olay tabanlı, düşük olasılık
        if self.rng.random() < 0.008:
            if self._next_flash_t == float('inf'):
                self.schedule_flash_crowd(t)
            return TrafficPattern.FLASH_CROWD

        # 7. Cascading fail — gündüz, çok nadir
        if 8.0 <= hour < 22.0 and self.rng.random() < 0.003:
            self._cascade_start_t = None   # yeni olay başlat
            return TrafficPattern.CASCADING_FAIL

        # 8. Ağırlıklı rastgele — tüm günler aynı havuz
        patterns = list(self.pattern_weights.keys())
        weights  = np.array([self.pattern_weights[p] for p in patterns])
        weights /= weights.sum()
        return self.rng.choice(patterns, p=weights)