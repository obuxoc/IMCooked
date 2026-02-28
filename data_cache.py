"""Algothon 2026 — Ultra-fast orderbook data caching layer.

This module sits between the raw SSE stream and your strategies.
Every SSE tick gets parsed into a flat snapshot dict, appended to a
bounded deque (zero-allocation once warm), and exposed via a dead-simple
API your teammates can use without reading a single line of exchange code.

USAGE (for teammates):
    snap = bot.latest["TIDE_SPOT"]      # latest snapshot (OrderBookSnapshot)
    snap.mid                            # mid price
    snap.spread                         # bid-ask spread
    snap.imbalance                      # order-book imbalance [-1, 1]
    snap.best_bid                       # best bid price

    df = bot.history("TIDE_SPOT", n=50) # last 50 ticks as pandas DataFrame
    mids = bot.mids                     # {"TIDE_SPOT": 3500.0, ...} all products

    # Arbitrage helpers
    etf_fair = bot.fair_etf             # TIDE_SPOT + WX_SPOT + LHR_COUNT mids
    arb_gap  = bot.etf_arb_gap         # LON_ETF mid - fair ETF value

    # Trade history
    df_trades = bot.trade_history(n=100)
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. SNAPSHOT — one flat row of everything you need per product per tick
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class OrderBookSnapshot:
    """One point-in-time snapshot of a single product's orderbook.

    All fields are plain floats/ints — teammates never touch Order objects.
    """

    # identity
    timestamp: float          # time.time() when we received the SSE event
    product: str

    # top-of-book
    best_bid: float           # highest bid price  (NaN if empty)
    best_ask: float           # lowest ask price    (NaN if empty)
    mid: float                # (best_bid + best_ask) / 2
    spread: float             # best_ask - best_bid
    micro_price: float        # volume-weighted mid = (bid_vol*ask + ask_vol*bid)/(bid_vol+ask_vol)

    # depth at best level
    best_bid_vol: int         # total volume at best bid
    best_ask_vol: int         # total volume at best ask
    best_bid_own: int         # our volume at best bid
    best_ask_own: int         # our volume at best ask

    # full-book aggregates
    total_bid_vol: int        # sum of volume across ALL bid levels
    total_ask_vol: int        # sum of volume across ALL ask levels
    total_own_bid_vol: int    # our total resting bid volume
    total_own_ask_vol: int    # our total resting ask volume
    bid_levels: int           # number of distinct bid price levels
    ask_levels: int           # number of distinct ask price levels

    # derived signals
    imbalance: float          # (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
    top_imbalance: float      # same but only at top-of-book
    weighted_mid: float       # VWAP-style mid using top 3 levels each side
    tick_size: float          # product tick size for rounding

    # top-3 depth (for richer strategies)
    bid_prices_3: list        # [best, 2nd, 3rd] bid prices
    bid_vols_3: list          # corresponding volumes
    ask_prices_3: list        # [best, 2nd, 3rd] ask prices
    ask_vols_3: list          # corresponding volumes


# Field names excluding list fields (for fast DataFrame conversion)
_SCALAR_FIELDS = [f.name for f in fields(OrderBookSnapshot)
                  if f.name not in ("bid_prices_3", "bid_vols_3",
                                    "ask_prices_3", "ask_vols_3")]


def snapshot_to_dict(s: OrderBookSnapshot) -> dict:
    """Convert snapshot to flat dict (scalar fields only) for DataFrame."""
    return {k: getattr(s, k) for k in _SCALAR_FIELDS}


# ---------------------------------------------------------------------------
# 2. EXTRACT — turn a raw OrderBook into a snapshot in < 5 µs
# ---------------------------------------------------------------------------

def extract_snapshot(orderbook) -> OrderBookSnapshot:
    """Convert a raw OrderBook (from bot_template) into a flat snapshot.

    This runs on every SSE tick so it must be FAST.
    No pandas, no allocation, just arithmetic.
    """
    ts = time.time()
    product = orderbook.product
    tick_size = orderbook.tick_size
    buys = orderbook.buy_orders    # already sorted best-first (highest price)
    sells = orderbook.sell_orders  # already sorted best-first (lowest price)

    nan = float("nan")

    # --- top of book ---
    best_bid = buys[0].price if buys else nan
    best_ask = sells[0].price if sells else nan
    best_bid_vol = buys[0].volume if buys else 0
    best_ask_vol = sells[0].volume if sells else 0
    best_bid_own = buys[0].own_volume if buys else 0
    best_ask_own = sells[0].own_volume if sells else 0

    mid = (best_bid + best_ask) / 2 if buys and sells else nan
    spread = (best_ask - best_bid) if buys and sells else nan

    # micro price (better fair value estimate than mid)
    if buys and sells and (best_bid_vol + best_ask_vol) > 0:
        micro_price = (best_bid_vol * best_ask + best_ask_vol * best_bid) / (best_bid_vol + best_ask_vol)
    else:
        micro_price = mid

    # --- full book aggregates ---
    total_bid_vol = sum(o.volume for o in buys)
    total_ask_vol = sum(o.volume for o in sells)
    total_own_bid_vol = sum(o.own_volume for o in buys)
    total_own_ask_vol = sum(o.own_volume for o in sells)
    bid_levels = len(buys)
    ask_levels = len(sells)

    # --- imbalance signals ---
    denom = total_bid_vol + total_ask_vol
    imbalance = (total_bid_vol - total_ask_vol) / denom if denom > 0 else 0.0
    top_denom = best_bid_vol + best_ask_vol
    top_imbalance = (best_bid_vol - best_ask_vol) / top_denom if top_denom > 0 else 0.0

    # --- weighted mid using top 3 levels ---
    def _wmid(bids, asks, depth=3):
        b = bids[:depth]
        a = asks[:depth]
        if not b or not a:
            return mid
        bv = sum(o.volume for o in b)
        av = sum(o.volume for o in a)
        if bv + av == 0:
            return mid
        bp = sum(o.price * o.volume for o in b) / bv if bv else b[0].price
        ap = sum(o.price * o.volume for o in a) / av if av else a[0].price
        return (bv * ap + av * bp) / (bv + av)

    weighted_mid = _wmid(buys, sells)

    # --- top-3 depth arrays ---
    def _top3(orders):
        prices = [o.price for o in orders[:3]]
        vols = [o.volume for o in orders[:3]]
        while len(prices) < 3:
            prices.append(nan)
            vols.append(0)
        return prices, vols

    bid_prices_3, bid_vols_3 = _top3(buys)
    ask_prices_3, ask_vols_3 = _top3(sells)

    return OrderBookSnapshot(
        timestamp=ts,
        product=product,
        best_bid=best_bid,
        best_ask=best_ask,
        mid=mid,
        spread=spread,
        micro_price=micro_price,
        best_bid_vol=best_bid_vol,
        best_ask_vol=best_ask_vol,
        best_bid_own=best_bid_own,
        best_ask_own=best_ask_own,
        total_bid_vol=total_bid_vol,
        total_ask_vol=total_ask_vol,
        total_own_bid_vol=total_own_bid_vol,
        total_own_ask_vol=total_own_ask_vol,
        bid_levels=bid_levels,
        ask_levels=ask_levels,
        imbalance=imbalance,
        top_imbalance=top_imbalance,
        weighted_mid=weighted_mid,
        tick_size=tick_size,
        bid_prices_3=bid_prices_3,
        bid_vols_3=bid_vols_3,
        ask_prices_3=ask_prices_3,
        ask_vols_3=ask_vols_3,
    )


# ---------------------------------------------------------------------------
# 3. CACHE — bounded, zero-copy ring buffer per product
# ---------------------------------------------------------------------------

_DEFAULT_MAXLEN = 5_000  # ~80 min at 1 update/sec = plenty for any signal


class DataCache:
    """Fast per-product orderbook + trade history cache.

    This is the single source of truth your strategies read from.
    The bot writes; strategies read.  Thread-safe for single-writer / multi-reader
    because deque.append is atomic in CPython.
    """

    def __init__(self, maxlen: int = _DEFAULT_MAXLEN):
        self._maxlen = maxlen
        # {product_symbol: deque[OrderBookSnapshot]}
        self._snapshots: dict[str, deque[OrderBookSnapshot]] = {}
        # latest snapshot per product (instant access, no deque indexing)
        self._latest: dict[str, OrderBookSnapshot] = {}
        # trade history (all products in one deque)
        self._trades: deque[dict] = deque(maxlen=maxlen)
        # cross-product mids cache (updated every tick)
        self._mids: dict[str, float] = {}

    # ---- WRITE (called by bot on every SSE tick) ----

    def update_orderbook(self, orderbook) -> OrderBookSnapshot:
        """Parse raw OrderBook, cache snapshot, return it."""
        snap = extract_snapshot(orderbook)
        sym = snap.product
        if sym not in self._snapshots:
            self._snapshots[sym] = deque(maxlen=self._maxlen)
        self._snapshots[sym].append(snap)
        self._latest[sym] = snap
        self._mids[sym] = snap.mid
        return snap

    def record_trade(self, trade) -> dict:
        """Cache a trade event (from on_trades callback)."""
        t = {
            "timestamp": time.time(),
            "exchange_ts": trade.timestamp,
            "product": trade.product,
            "price": trade.price,
            "volume": trade.volume,
            "buyer": trade.buyer,
            "seller": trade.seller,
        }
        self._trades.append(t)
        return t

    # ---- READ (called by strategies) ----

    @property
    def latest(self) -> dict[str, OrderBookSnapshot]:
        """Latest snapshot per product.  Usage: cache.latest["TIDE_SPOT"].mid"""
        return self._latest

    @property
    def mids(self) -> dict[str, float]:
        """Current mid prices for all known products. {symbol: mid}"""
        return dict(self._mids)

    def history(self, product: str, n: Optional[int] = None) -> pd.DataFrame:
        """Return last `n` snapshots for `product` as a DataFrame.

        Only scalar fields are included (no nested lists).
        If n is None, returns full history.
        """
        buf = self._snapshots.get(product)
        if not buf:
            return pd.DataFrame()
        data = buf if n is None else list(buf)[-n:]
        return pd.DataFrame([snapshot_to_dict(s) for s in data])

    def trade_history(self, product: Optional[str] = None,
                      n: Optional[int] = None) -> pd.DataFrame:
        """Return trade history as DataFrame. Filter by product optionally."""
        data = list(self._trades)
        if product:
            data = [t for t in data if t["product"] == product]
        if n is not None:
            data = data[-n:]
        return pd.DataFrame(data) if data else pd.DataFrame()

    @property
    def products(self) -> list[str]:
        """List of all products seen so far."""
        return list(self._snapshots.keys())

    # ---- ARBITRAGE HELPERS ----

    # LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT
    _ETF_COMPONENTS = ("TIDE_SPOT", "WX_SPOT", "LHR_COUNT")

    @property
    def fair_etf(self) -> float:
        """Fair value of LON_ETF from component mid prices."""
        try:
            return sum(self._mids[c] for c in self._ETF_COMPONENTS)
        except KeyError:
            return float("nan")

    @property
    def etf_arb_gap(self) -> float:
        """LON_ETF market mid minus its fair (component-implied) value.

        Positive = ETF is overpriced vs components.
        Negative = ETF is cheap vs components.
        """
        etf_mid = self._mids.get("LON_ETF", float("nan"))
        return etf_mid - self.fair_etf

    def arb_snapshot(self) -> dict:
        """Full cross-product arbitrage state in one dict.

        Returns {
            "etf_mid", "fair_etf", "gap",
            "tide_mid", "wx_mid", "lhr_mid",
            "fly_mid", "fly_fair"  (placeholder),
            "timestamp"
        }
        """
        etf_mid = self._mids.get("LON_ETF", float("nan"))
        fair = self.fair_etf
        fly_mid = self._mids.get("LON_FLY", float("nan"))

        return {
            "timestamp": time.time(),
            "etf_mid": etf_mid,
            "fair_etf": fair,
            "gap": etf_mid - fair,
            "tide_mid": self._mids.get("TIDE_SPOT", float("nan")),
            "wx_mid": self._mids.get("WX_SPOT", float("nan")),
            "lhr_mid": self._mids.get("LHR_COUNT", float("nan")),
            "fly_mid": fly_mid,
            # LON_FLY fair value requires ETF settlement — compute in strategy
        }

    def multi_product_df(self, n: Optional[int] = None) -> pd.DataFrame:
        """Pivot table: rows = timestamps, columns = product mids.

        Useful for correlation / co-integration analysis.
        Aligns by nearest timestamp (forward-fill).
        """
        frames = []
        for sym in self._snapshots:
            h = self.history(sym, n=n)
            if h.empty:
                continue
            frames.append(
                h[["timestamp", "mid"]].rename(columns={"mid": sym}).set_index("timestamp")
            )
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=1).sort_index().ffill()
        return df


# ---------------------------------------------------------------------------
# 4. DATA PERSISTENCE — append-only, dedup, single file per product
# ---------------------------------------------------------------------------

def _safe_csv_append(df: "pd.DataFrame", path: "Path", write_header: bool,
                     retries: int = 3, delay: float = 0.5) -> bool:
    """Write df to CSV with retries — handles OneDrive/Windows file locks."""
    import time as _time
    for attempt in range(retries):
        try:
            df.to_csv(path, mode="a", header=write_header, index=False)
            return True
        except PermissionError:
            if attempt < retries - 1:
                _time.sleep(delay)
            else:
                return False
    return False


class DataPersistence:
    """Handles saving/loading orderbook data to/from CSV files.

    Design decisions:
    - ONE file per product (e.g. collected_data/TIDE_SPOT.csv)
    - ONE file for trades (collected_data/trades.csv)
    - APPEND mode: new data is appended, never overwritten
    - DEDUP by timestamp: duplicate timestamps are dropped on save
    - LOAD on startup: existing CSV data is available immediately
    """

    def __init__(self, data_dir: str = "collected_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Track last-saved timestamp per product to avoid duplicate writes
        self._last_saved_ts: dict[str, float] = {}
        self._last_saved_trade_ts: float = 0.0

    def _product_path(self, product: str) -> Path:
        return self.data_dir / f"{product}.csv"

    @property
    def _trade_path(self) -> Path:
        return self.data_dir / "trades.csv"

    def load_existing(self, product: str) -> pd.DataFrame:
        """Load existing CSV for a product. Returns empty DataFrame if none."""
        p = self._product_path(product)
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p)
                if "timestamp" in df.columns and not df.empty:
                    self._last_saved_ts[product] = df["timestamp"].max()
                return df
            except Exception as e:
                print(f"[PERSIST] Warning: couldn't load {p}: {e}")
        return pd.DataFrame()

    def load_existing_trades(self) -> pd.DataFrame:
        """Load existing trade CSV. Returns empty DataFrame if none."""
        p = self._trade_path
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p)
                if "timestamp" in df.columns and not df.empty:
                    self._last_saved_trade_ts = df["timestamp"].max()
                return df
            except Exception as e:
                print(f"[PERSIST] Warning: couldn't load {p}: {e}")
        return pd.DataFrame()

    def save_snapshots(self, cache: DataCache) -> int:
        """Append new snapshots to per-product CSVs. Returns rows written."""
        total_written = 0
        for product in cache.products:
            df = cache.history(product)
            if df.empty:
                continue

            # Filter to only NEW rows (timestamp > last saved)
            last_ts = self._last_saved_ts.get(product, 0.0)
            new_rows = df[df["timestamp"] > last_ts]
            if new_rows.empty:
                continue

            p = self._product_path(product)
            write_header = not p.exists() or p.stat().st_size == 0

            if _safe_csv_append(new_rows, p, write_header):
                self._last_saved_ts[product] = new_rows["timestamp"].max()
                total_written += len(new_rows)
            else:
                print(f"[PERSIST] Skipped {product} (file locked by OneDrive)")

        return total_written

    def save_trades(self, cache: DataCache) -> int:
        """Append new trades to trades.csv. Returns rows written."""
        df = cache.trade_history()
        if df.empty:
            return 0

        new_rows = df[df["timestamp"] > self._last_saved_trade_ts]
        if new_rows.empty:
            return 0

        p = self._trade_path
        write_header = not p.exists() or p.stat().st_size == 0

        if _safe_csv_append(new_rows, p, write_header):
            self._last_saved_trade_ts = new_rows["timestamp"].max()
            return len(new_rows)
        print("[PERSIST] Skipped trades (file locked by OneDrive)")
        return 0

    def save_all(self, cache: DataCache) -> tuple[int, int]:
        """Save both snapshots and trades. Returns (snap_rows, trade_rows)."""
        s = self.save_snapshots(cache)
        t = self.save_trades(cache)
        if s or t:
            print(f"[PERSIST] Saved {s} snapshot rows + {t} trade rows")
        return s, t

    def get_full_history(self, product: str) -> pd.DataFrame:
        """Load full saved CSV history for a product (from disk, not cache)."""
        return self.load_existing(product)

    def get_all_products_on_disk(self) -> list[str]:
        """List all products that have saved data."""
        return [p.stem for p in self.data_dir.glob("*.csv")
                if p.stem != "trades"]
