"""Algothon 2026 — Alpha Engine: Real-World Data → Settlement Fair Values.

THE #1 EDGE IN THIS COMPETITION:
    Products settle based on observable real-world data.
    If we compute fair values from live data sources, every trade is +EV.

FREE APIs (no key needed):
    Open-Meteo:          temp + humidity → WX_SPOT, WX_SUM fair values
    EA Flood Monitoring:  Thames tidal   → TIDE_SPOT, TIDE_SWING fair values

DERIVED:
    LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT  (LHR from market)
    LON_FLY = 2*Put(6200) + Call(6200) - 2*Call(6600) + 3*Call(7000)

CONFIDENCE scales 0→1 as we approach settlement (12pm London).
Closer to settlement = better forecast = higher confidence = bigger positions.

Usage:
    alpha = AlphaEngine()
    alpha.refresh(market_mids)   # call every 60s
    fair, conf = alpha.get("WX_SPOT")
"""

from __future__ import annotations

import math
import time
import threading

import requests
import pandas as pd
from datetime import datetime, timedelta


# London coords for Open-Meteo
LONDON_LAT, LONDON_LON = 51.5074, -0.1278

# EA Flood Monitoring gauge ID: Westminster tidal level
THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"

# Don't hit external APIs more than every 2 minutes
_CACHE_TTL = 120


class AlphaEngine:
    """Fetches real-world data and computes settlement fair values."""

    def __init__(self):
        self.fair: dict[str, float] = {}
        self.confidence: dict[str, float] = {}
        self.last_update: float = 0
        self._lock = threading.Lock()

        # Raw data caches
        self._weather: pd.DataFrame | None = None
        self._tidal: pd.DataFrame | None = None
        self._weather_ts: float = 0
        self._tidal_ts: float = 0

    # ================================================================
    # PUBLIC API
    # ================================================================

    def refresh(self, market_mids: dict[str, float] | None = None) -> dict[str, float]:
        """Fetch latest data and recompute all fair values.

        Args:
            market_mids: current exchange mid prices (for LHR_COUNT etc.)
        Returns:
            dict of product -> fair_value
        """
        mids = market_mids or {}

        for label, fn in [("Weather", self._fetch_weather),
                          ("Tidal", self._fetch_tidal)]:
            try:
                fn()
            except Exception as e:
                print(f"[ALPHA] {label} error: {e}")

        self._compute(mids)
        self.last_update = time.time()
        return dict(self.fair)

    def get(self, product: str) -> tuple[float, float]:
        """Get (fair_value, confidence) for a product.

        Returns (NaN, 0.0) if no estimate available.
        """
        with self._lock:
            return (
                self.fair.get(product, float("nan")),
                self.confidence.get(product, 0.0),
            )

    # ================================================================
    # DATA FETCHING
    # ================================================================

    def _fetch_weather(self):
        """15-min weather observations + forecast from Open-Meteo (free)."""
        if time.time() - self._weather_ts < _CACHE_TTL:
            return

        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": LONDON_LAT,
                "longitude": LONDON_LON,
                "minutely_15": "temperature_2m,relative_humidity_2m",
                "past_minutely_15": 96,      # 24h back
                "forecast_minutely_15": 96,   # 24h forward
                "timezone": "Europe/London",
            },
            timeout=10,
        )
        r.raise_for_status()
        m = r.json()["minutely_15"]

        self._weather = pd.DataFrame({
            "time": pd.to_datetime(m["time"]),
            "temp_c": pd.to_numeric(m["temperature_2m"], errors="coerce"),
            "humidity": pd.to_numeric(m["relative_humidity_2m"], errors="coerce"),
        }).dropna()

        self._weather_ts = time.time()
        n = len(self._weather)
        rng = f"{self._weather['time'].iloc[0]} → {self._weather['time'].iloc[-1]}" if n else "empty"
        print(f"[ALPHA] Weather OK — {n} pts, {rng}")

    def _fetch_tidal(self):
        """Thames tidal readings from EA Flood Monitoring API (free)."""
        if time.time() - self._tidal_ts < _CACHE_TTL:
            return

        r = requests.get(
            f"https://environment.data.gov.uk/flood-monitoring/id/measures/"
            f"{THAMES_MEASURE}/readings",
            params={"_sorted": "", "_limit": 200},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return

        df = pd.DataFrame(items)[["dateTime", "value"]].rename(
            columns={"dateTime": "time", "value": "level"})
        df["time"] = (
            pd.to_datetime(df["time"], utc=True)
            .dt.tz_convert("Europe/London")
            .dt.tz_localize(None)
        )
        df["level"] = pd.to_numeric(df["level"], errors="coerce")
        self._tidal = df.dropna().sort_values("time").reset_index(drop=True)
        self._tidal_ts = time.time()

        latest = self._tidal.iloc[-1]
        print(f"[ALPHA] Tidal OK — {len(self._tidal)} pts, "
              f"latest={latest['level']:.3f} m @ {latest['time']}")

    # ================================================================
    # TIME HELPERS (session = 12pm → 12pm London)
    # ================================================================

    @staticmethod
    def _next_settlement() -> datetime:
        """Next 12:00 London time."""
        now = datetime.now()
        s = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if now >= s:
            s += timedelta(days=1)
        return s

    @staticmethod
    def _session_start() -> datetime:
        """Start of current 24h session (12pm yesterday)."""
        return AlphaEngine._next_settlement() - timedelta(hours=24)

    @staticmethod
    def _hours_left() -> float:
        """Hours until next settlement."""
        return max(0, (AlphaEngine._next_settlement() - datetime.now()).total_seconds() / 3600)

    # ================================================================
    # FAIR VALUE COMPUTATION
    # ================================================================

    def _compute(self, mids: dict[str, float]):
        fv: dict[str, float] = {}
        cf: dict[str, float] = {}
        hl = self._hours_left()

        # ── TIDE_SPOT = abs(level_mm) at settlement ─────────────────
        # EA gives actual readings (not forecasts). Latest reading is best estimate.
        # Tides change slowly (~2m over 6h), so confidence rises near settlement.
        if self._tidal is not None and not self._tidal.empty:
            try:
                lvl = float(self._tidal.iloc[-1]["level"])
                fv["TIDE_SPOT"] = abs(lvl * 1000)
                # Higher confidence closer to settlement
                cf["TIDE_SPOT"] = max(0.20, min(0.95, 1.0 - hl / 24))
            except Exception:
                pass

        # ── TIDE_SWING = sum(strangle payoffs on 15-min diffs) ──────
        # Cumulative over session. Observe partial + extrapolate rest.
        if self._tidal is not None and len(self._tidal) > 1:
            try:
                ss = self._session_start()
                td = self._tidal[self._tidal["time"] >= ss]
                if len(td) > 1:
                    cm = td["level"].values * 100  # metres → centimetres
                    diffs = [abs(cm[i] - cm[i - 1]) for i in range(1, len(cm))]
                    obs_payoff = sum(
                        max(0, 20 - d) + max(0, d - 25) for d in diffs
                    )
                    n_obs = len(diffs)
                    n_total = 96  # 24h / 15min
                    avg_per_interval = obs_payoff / n_obs if n_obs else 0
                    remaining = max(0, n_total - n_obs)
                    fv["TIDE_SWING"] = obs_payoff + avg_per_interval * remaining
                    cf["TIDE_SWING"] = max(0.10, n_obs / n_total)
            except Exception:
                pass

        # ── WX_SPOT = temp_F × humidity at settlement ───────────────
        # Open-Meteo gives FORECASTS → we can look up the 12pm value directly.
        if self._weather is not None and not self._weather.empty:
            try:
                st = self._next_settlement()
                df = self._weather
                idx = (df["time"] - st).abs().idxmin()
                row = df.loc[idx]
                temp_f = float(row["temp_c"]) * 9 / 5 + 32
                humidity = float(row["humidity"])
                fv["WX_SPOT"] = temp_f * humidity
                # Forecast accuracy improves closer to settlement
                cf["WX_SPOT"] = max(0.30, min(0.90, 1.0 - hl / 24))
            except Exception:
                pass

        # ── WX_SUM = sum(temp_F × humidity) / 100 over session ─────
        # Uses both observed and forecasted weather for full 24h window.
        if self._weather is not None and not self._weather.empty:
            try:
                ss = self._session_start()
                st = self._next_settlement()
                df = self._weather
                mask = (df["time"] >= ss) & (df["time"] <= st)
                session = df[mask]
                if not session.empty:
                    vals = (session["temp_c"] * 9 / 5 + 32) * session["humidity"]
                    fv["WX_SUM"] = vals.sum() / 100
                    cf["WX_SUM"] = max(0.15, min(0.85, len(session) / 96))
            except Exception:
                pass

        # ── LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT ──────────────
        # LHR_COUNT has no free data source → use market mid as proxy.
        tide = fv.get("TIDE_SPOT")
        wx = fv.get("WX_SPOT")
        lhr = mids.get("LHR_COUNT")
        if (tide is not None and wx is not None
                and lhr is not None and not math.isnan(lhr)):
            fv["LON_ETF"] = tide + wx + lhr
            cf["LON_ETF"] = min(
                cf.get("TIDE_SPOT", 0),
                cf.get("WX_SPOT", 0),
                0.5,  # LHR from market = moderate confidence
            )

        # ── LON_FLY = options structure on ETF ──────────────────────
        etf = fv.get("LON_ETF")
        if etf is not None:
            fv["LON_FLY"] = (
                2 * max(0, 6200 - etf)
                + max(0, etf - 6200)
                - 2 * max(0, etf - 6600)
                + 3 * max(0, etf - 7000)
            )
            cf["LON_FLY"] = cf.get("LON_ETF", 0) * 0.8

        # ── Publish ─────────────────────────────────────────────────
        with self._lock:
            self.fair = fv
            self.confidence = cf

        parts = [f"{k}={v:.0f}({cf.get(k, 0):.0%})"
                 for k, v in sorted(fv.items())]
        if parts:
            print(f"[ALPHA] {' | '.join(parts)}")
            print(f"[ALPHA] Settlement in {hl:.1f}h")

    # ================================================================
    # SUMMARY
    # ================================================================

    def summary(self) -> str:
        """Human-readable summary of all fair values."""
        with self._lock:
            if not self.fair:
                return "[ALPHA] No fair values computed yet"
            lines = ["[ALPHA] Fair Values:"]
            for k in sorted(self.fair):
                v = self.fair[k]
                c = self.confidence.get(k, 0)
                lines.append(f"  {k:<15} {v:>10.0f}  (conf={c:.0%})")
            return "\n".join(lines)
