"""Algothon 2026 — Competition-Specialized Trading Strategies.

DESIGNED FOR: Thin books, wild price swings, irregular fills, active bots.

KEY INSIGHT FROM LIVE DATA:
  - LHR_COUNT swings from ~700 to ~1975 (extreme volatility)
  - LON_ETF trades from ~5700 to ~11200 (massive mispricing)
  - WX_SPOT is most stable (~4400-4440)
  - TIDE_SPOT moderate (~2530-2700)
  - Position limits: ±100 per product
  - Thin book = our limit orders WILL get filled at crazy prices

STRATEGY OVERVIEW:
  1. grid_strategy       — Wide price grid capturing volatility swings
  2. etf_arb_strategy    — ETF vs components when gap > threshold
  3. component_arb       — Trade components when one diverges
  4. fly_strategy        — LON_FLY vs computed fair value
  5. mean_revert         — Mean reversion on volatile products
  6. inventory_unwind    — Passively unwind large positions
  7. etf_mm_strategy     — Market-make LON_ETF off component fair value
  8. vol_mm_strategy      — Volatility-adaptive LON_ETF market making

TEAMMATE CONTRACT:
  Every function: (bot, snap) -> Signal | None
  NEVER send orders directly. NEVER worry about risk.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import TYPE_CHECKING, Optional

from signals import Signal, Side, OrderType

if TYPE_CHECKING:
    from algothon_bot import AlgothonBot
    from data_cache import OrderBookSnapshot

from dataclasses import dataclass, field


# =========================================================================
# SHARED POSITION CACHE — updated by run.py, read by all strategies
# =========================================================================
# This avoids each strategy calling bot.get_positions() (an API call).
# run.py calls update_positions() every main-loop tick (~10s).

_positions: dict[str, int] = {}
_POSITION_SOFT_LIMIT = 40   # start being careful above this
_POSITION_HARD_LIMIT = 60   # only allow risk-reducing trades above this


def update_positions(positions: dict[str, int]) -> None:
    """Called by run.py to push exchange positions into strategy module."""
    global _positions
    _positions = dict(positions)


def get_position(product: str) -> int:
    """Get cached net position for a product.  0 if unknown."""
    return _positions.get(product, 0)


def _would_increase_risk(product: str, side: Side) -> bool:
    """True if this trade would push position further from zero."""
    pos = get_position(product)
    if side == Side.BUY and pos > 0:
        return True
    if side == Side.SELL and pos < 0:
        return True
    return False


def _position_guard(product: str, side: Side) -> bool:
    """Return True if the trade is BLOCKED by position limits.
    
    - Above SOFT_LIMIT: block risk-increasing trades
    - Below SOFT_LIMIT: always allow
    """
    pos = get_position(product)
    # Above soft limit — only allow risk-reducing
    if abs(pos) >= _POSITION_SOFT_LIMIT:
        if side == Side.BUY and pos > 0:
            return True   # blocked: already long, don't buy more
        if side == Side.SELL and pos < 0:
            return True   # blocked: already short, don't sell more
    return False


# =========================================================================
# PRICE HISTORY TRACKER (shared state for all strategies)
# =========================================================================

class _PriceTracker:
    """Rolling EMA + volatility tracker per product. Ultra-fast, no pandas."""

    def __init__(self):
        self._ema_fast: dict[str, float] = {}   # ~10-tick EMA
        self._ema_slow: dict[str, float] = {}   # ~50-tick EMA
        self._vol: dict[str, float] = {}         # rolling volatility (std of returns)
        self._last_mid: dict[str, float] = {}
        self._returns: dict[str, deque] = {}     # rolling returns
        self._count: dict[str, int] = {}
        self._alpha_fast = 2 / (10 + 1)
        self._alpha_slow = 2 / (50 + 1)

    def update(self, product: str, mid: float) -> None:
        if math.isnan(mid):
            return
        if product not in self._ema_fast:
            self._ema_fast[product] = mid
            self._ema_slow[product] = mid
            self._vol[product] = 0.0
            self._last_mid[product] = mid
            self._returns[product] = deque(maxlen=100)
            self._count[product] = 0
        else:
            self._ema_fast[product] = self._alpha_fast * mid + (1 - self._alpha_fast) * self._ema_fast[product]
            self._ema_slow[product] = self._alpha_slow * mid + (1 - self._alpha_slow) * self._ema_slow[product]

            # Track returns for volatility
            last = self._last_mid[product]
            if last > 0:
                ret = (mid - last) / last
                self._returns[product].append(ret)
                # Rolling std
                if len(self._returns[product]) > 5:
                    rets = list(self._returns[product])
                    mean_r = sum(rets) / len(rets)
                    var = sum((r - mean_r) ** 2 for r in rets) / len(rets)
                    self._vol[product] = var ** 0.5

            self._last_mid[product] = mid
        self._count[product] = self._count.get(product, 0) + 1

    def ema_fast(self, product: str) -> float:
        return self._ema_fast.get(product, float("nan"))

    def ema_slow(self, product: str) -> float:
        return self._ema_slow.get(product, float("nan"))

    def volatility(self, product: str) -> float:
        return self._vol.get(product, 0.0)

    def tick_count(self, product: str) -> int:
        return self._count.get(product, 0)

    def trend(self, product: str) -> float:
        """Fast EMA / Slow EMA - 1.  Positive = uptrend."""
        f = self._ema_fast.get(product, 0)
        s = self._ema_slow.get(product, 0)
        if s > 0:
            return (f / s) - 1.0
        return 0.0


# Global tracker — updated by every strategy call
_tracker = _PriceTracker()


def _update_tracker(snap) -> None:
    """Call on every tick to update shared price tracker."""
    _tracker.update(snap.product, snap.mid)


# =========================================================================
# STRATEGY 1: GRID / VOLATILITY CAPTURE
# =========================================================================
# Place a ladder of limit orders at multiple price levels.
# When wild swings happen (LHR_COUNT drops 700->1400), our orders fill.
# This is THE #1 strategy for thin, volatile books.

_GRID_CONFIG = {
    "TIDE_SPOT":  {"center_ema": True, "width_pct": 0.08, "levels": 4, "vol": 2, "interval": 5.0},
    "WX_SPOT":    {"center_ema": True, "width_pct": 0.04, "levels": 3, "vol": 2, "interval": 5.0},
    "LHR_COUNT":  {"center_ema": True, "width_pct": 0.20, "levels": 5, "vol": 2, "interval": 4.0},
    "LHR_INDEX":  {"center_ema": True, "width_pct": 0.15, "levels": 4, "vol": 2, "interval": 5.0},
    "TIDE_SWING": {"center_ema": True, "width_pct": 0.05, "levels": 3, "vol": 1, "interval": 5.0},
    "WX_SUM":     {"center_ema": True, "width_pct": 0.06, "levels": 3, "vol": 2, "interval": 5.0},
}

_grid_last_signal: dict[str, float] = {}
_grid_level_idx: dict[str, int] = {}


def grid_strategy(bot: AlgothonBot, snap: OrderBookSnapshot) -> Optional[Signal]:
    """Wide grid strategy — places buy/sell orders across a price range.

    Each tick, we emit ONE level of the grid (round-robin) to stay within
    rate limits. Over several ticks, we build the full grid.
    """
    _update_tracker(snap)

    product = snap.product
    if product not in _GRID_CONFIG:
        return None

    cfg = _GRID_CONFIG[product]
    now = time.time()

    if now - _grid_last_signal.get(product, 0) < cfg["interval"]:
        return None

    if math.isnan(snap.mid):
        return None

    if _tracker.tick_count(product) < 10:
        return None

    center = _tracker.ema_slow(product) if cfg["center_ema"] else snap.mid
    if math.isnan(center) or center <= 0:
        return None

    width = center * cfg["width_pct"]
    levels = cfg["levels"]
    vol = cfg["vol"]

    idx = _grid_level_idx.get(product, 0) % (levels * 2)
    _grid_level_idx[product] = idx + 1

    if idx < levels:
        side = Side.BUY
        # ── Position guard: don't buy if already too long ──
        if _position_guard(product, side):
            return None
        frac = (idx + 1) / levels
        price = center - width * frac
        if price <= 0:
            return None
        _grid_last_signal[product] = now
        return Signal(
            product=product,
            side=side,
            price=price,
            volume=vol,
            order_type=OrderType.GTC,
            reason=f"GRID buy L{idx+1}/{levels} @ {price:.0f} (ctr={center:.0f})",
        )
    else:
        side = Side.SELL
        # ── Position guard: don't sell if already too short ──
        if _position_guard(product, side):
            return None
        sell_idx = idx - levels
        frac = (sell_idx + 1) / levels
        price = center + width * frac
        _grid_last_signal[product] = now
        return Signal(
            product=product,
            side=side,
            price=price,
            volume=vol,
            order_type=OrderType.GTC,
            reason=f"GRID sell L{sell_idx+1}/{levels} @ {price:.0f} (ctr={center:.0f})",
        )


# =========================================================================
# STRATEGY 2: ETF ARBITRAGE (enhanced)
# =========================================================================

ARB_THRESHOLD = 50.0             # widened: 15→50 to avoid constant micro-arb fills
ARB_AGGRESSIVE_THRESHOLD = 150.0  # widened: 80→150
ARB_VOLUME = 2
ARB_LARGE_VOLUME = 3              # reduced: 4→3
ARB_MIN_INTERVAL = 3.0            # slowed: 2→3s
ARB_MAX_ETF_POS = 30              # cap: stop arbing if ETF position > ±30

_arb_last_trade: float = 0.0


def etf_arb_strategy(bot: AlgothonBot, snap: OrderBookSnapshot) -> Optional[Signal]:
    """ETF vs components arb — IOC aggress when gap is large."""
    global _arb_last_trade

    _update_tracker(snap)

    relevant = {"LON_ETF", "TIDE_SPOT", "WX_SPOT", "LHR_COUNT"}
    if snap.product not in relevant:
        return None

    now = time.time()
    if now - _arb_last_trade < ARB_MIN_INTERVAL:
        return None

    gap = bot.etf_arb_gap
    if math.isnan(gap):
        return None

    abs_gap = abs(gap)
    if abs_gap < ARB_THRESHOLD:
        return None

    etf = bot.latest.get("LON_ETF")
    if etf is None or math.isnan(etf.best_bid) or math.isnan(etf.best_ask):
        return None

    # ── Position cap: stop piling into ETF ──
    etf_pos = get_position("LON_ETF")
    if abs(etf_pos) >= ARB_MAX_ETF_POS:
        # Only allow risk-reducing arb trades
        if gap > 0 and etf_pos < 0:  # would sell, already short → blocked
            return None
        if gap < 0 and etf_pos > 0:  # would buy, already long → blocked
            return None
        if gap > 0 and etf_pos > 0:  # sell when long → OK (reducing)
            pass
        elif gap < 0 and etf_pos < 0:  # buy when short → OK (reducing)
            pass
        else:
            return None

    vol = ARB_LARGE_VOLUME if abs_gap > ARB_AGGRESSIVE_THRESHOLD else ARB_VOLUME

    if gap > 0:
        if _position_guard("LON_ETF", Side.SELL):
            return None
        _arb_last_trade = now
        return Signal(
            product="LON_ETF",
            side=Side.SELL,
            price=etf.best_bid,
            volume=vol,
            order_type=OrderType.IOC,
            reason=f"ARB gap={gap:+.0f} SELL ETF (pos={etf_pos})",
            urgency=min(1.0, abs_gap / 100),
        )
    else:
        if _position_guard("LON_ETF", Side.BUY):
            return None
        _arb_last_trade = now
        return Signal(
            product="LON_ETF",
            side=Side.BUY,
            price=etf.best_ask,
            volume=vol,
            order_type=OrderType.IOC,
            reason=f"ARB gap={gap:+.0f} BUY ETF (pos={etf_pos})",
            urgency=min(1.0, abs_gap / 100),
        )


# =========================================================================
# STRATEGY 3: COMPONENT CROSS-ARBITRAGE
# =========================================================================

COMP_ARB_THRESHOLD = 0.015
COMP_ARB_INTERVAL = 3.0
_comp_arb_last: dict[str, float] = {}


def component_arb_strategy(bot: AlgothonBot, snap: OrderBookSnapshot) -> Optional[Signal]:
    """Trade when one component diverges from its expected contribution to ETF."""
    _update_tracker(snap)

    components = {"TIDE_SPOT", "WX_SPOT", "LHR_COUNT"}
    if snap.product not in components:
        return None

    now = time.time()
    if now - _comp_arb_last.get(snap.product, 0) < COMP_ARB_INTERVAL:
        return None

    mids = bot.mids
    if not all(c in mids for c in components):
        return None

    etf_mid = mids.get("LON_ETF", float("nan"))
    if math.isnan(etf_mid) or etf_mid <= 0:
        return None

    fair_sum = sum(mids[c] for c in components)
    if fair_sum <= 0:
        return None

    comp_mid = mids[snap.product]
    expected_share = comp_mid / fair_sum
    actual_etf_share = comp_mid / etf_mid

    divergence = actual_etf_share - expected_share
    if abs(divergence) < COMP_ARB_THRESHOLD:
        return None

    # ── Position guard ──
    if divergence < 0:
        if _position_guard(snap.product, Side.BUY):
            return None
    else:
        if _position_guard(snap.product, Side.SELL):
            return None

    _comp_arb_last[snap.product] = now

    if divergence < 0:
        return Signal(
            product=snap.product,
            side=Side.BUY,
            price=snap.best_ask,
            volume=2,
            order_type=OrderType.IOC,
            reason=f"COMP_ARB {snap.product} cheap div={divergence:.3f} (pos={get_position(snap.product)})",
        )
    else:
        return Signal(
            product=snap.product,
            side=Side.SELL,
            price=snap.best_bid,
            volume=2,
            order_type=OrderType.IOC,
            reason=f"COMP_ARB {snap.product} rich div={divergence:.3f} (pos={get_position(snap.product)})",
        )


# =========================================================================
# STRATEGY 4: LON_FLY FAIR VALUE ARBITRAGE
# =========================================================================

def compute_fly_fair(etf_value: float) -> float:
    """LON_FLY settlement: 2*Put(6200) + Call(6200) - 2*Call(6600) + 3*Call(7000)."""
    put_6200 = max(0, 6200 - etf_value)
    call_6200 = max(0, etf_value - 6200)
    call_6600 = max(0, etf_value - 6600)
    call_7000 = max(0, etf_value - 7000)
    return 2 * put_6200 + call_6200 - 2 * call_6600 + 3 * call_7000


FLY_THRESHOLD = 20.0
FLY_INTERVAL = 4.0
_fly_last: float = 0.0


def fly_strategy(bot: AlgothonBot, snap: OrderBookSnapshot) -> Optional[Signal]:
    """Trade LON_FLY when it diverges from computed fair value."""
    global _fly_last

    _update_tracker(snap)

    if snap.product != "LON_FLY":
        return None

    now = time.time()
    if now - _fly_last < FLY_INTERVAL:
        return None

    if math.isnan(snap.mid):
        return None

    etf_ema = _tracker.ema_slow("LON_ETF")
    if math.isnan(etf_ema) or etf_ema <= 0:
        return None

    fair_fly = compute_fly_fair(etf_ema)
    gap = snap.mid - fair_fly

    if abs(gap) < FLY_THRESHOLD:
        return None

    # ── Position guard ──
    if gap > 0 and _position_guard("LON_FLY", Side.SELL):
        return None
    if gap < 0 and _position_guard("LON_FLY", Side.BUY):
        return None

    _fly_last = now
    fly_pos = get_position("LON_FLY")

    if gap > 0:
        return Signal(
            product="LON_FLY",
            side=Side.SELL,
            price=snap.best_bid,
            volume=2,
            order_type=OrderType.IOC,
            reason=f"FLY gap={gap:+.0f} fair={fair_fly:.0f} SELL (pos={fly_pos})",
        )
    else:
        return Signal(
            product="LON_FLY",
            side=Side.BUY,
            price=snap.best_ask,
            volume=2,
            order_type=OrderType.IOC,
            reason=f"FLY gap={gap:+.0f} fair={fair_fly:.0f} BUY (pos={fly_pos})",
        )


# =========================================================================
# STRATEGY 5: MEAN REVERSION (for high-vol products)
# =========================================================================

MR_PRODUCTS = {"LHR_COUNT", "TIDE_SPOT", "LON_ETF"}
MR_THRESHOLD_SIGMAS = 1.5
MR_INTERVAL = 3.0
_mr_last: dict[str, float] = {}


def mean_revert_strategy(bot: AlgothonBot, snap: OrderBookSnapshot) -> Optional[Signal]:
    """Mean reversion — buy when far below EMA, sell when far above."""
    _update_tracker(snap)

    product = snap.product
    if product not in MR_PRODUCTS:
        return None

    now = time.time()
    if now - _mr_last.get(product, 0) < MR_INTERVAL:
        return None

    if _tracker.tick_count(product) < 30:
        return None

    if math.isnan(snap.mid):
        return None

    ema = _tracker.ema_slow(product)
    vol = _tracker.volatility(product)

    if math.isnan(ema) or vol <= 0:
        return None

    z_score = (snap.mid - ema) / (ema * vol) if ema * vol > 0 else 0.0

    if abs(z_score) < MR_THRESHOLD_SIGMAS:
        return None

    _mr_last[product] = now

    if z_score > MR_THRESHOLD_SIGMAS:
        return Signal(
            product=product,
            side=Side.SELL,
            price=snap.best_bid,
            volume=2,
            order_type=OrderType.GTC,
            reason=f"MEANREV z={z_score:.1f} SELL (ema={ema:.0f})",
        )
    else:
        return Signal(
            product=product,
            side=Side.BUY,
            price=snap.best_ask,
            volume=2,
            order_type=OrderType.GTC,
            reason=f"MEANREV z={z_score:.1f} BUY (ema={ema:.0f})",
        )


# =========================================================================
# STRATEGY 6: INVENTORY UNWIND
# =========================================================================

UNWIND_INTERVAL = 3.0              # faster: 5→3s
UNWIND_THRESHOLD = 15              # start unwinding above ±15 position
UNWIND_AGGRESSIVE_THRESHOLD = 40   # aggressive IOC above ±40
_unwind_last: dict[str, float] = {}


def inventory_unwind_strategy(bot: AlgothonBot, snap: OrderBookSnapshot) -> Optional[Signal]:
    """Actively unwind large positions using real net positions.

    Uses cached positions (not order book own-vol) to decide:
    - pos > +15: place sell limit orders to reduce
    - pos > +40: use IOC to aggressively reduce
    - Scales volume with position size (bigger pos → more aggressive)
    """
    _update_tracker(snap)

    product = snap.product
    now = time.time()

    if now - _unwind_last.get(product, 0) < UNWIND_INTERVAL:
        return None

    if math.isnan(snap.mid) or math.isnan(snap.spread) or snap.spread <= 0:
        return None

    pos = get_position(product)
    abs_pos = abs(pos)

    if abs_pos < UNWIND_THRESHOLD:
        return None

    # Scale volume with urgency: 1-5 based on position size
    vol = min(5, max(1, abs_pos // 15))
    aggressive = abs_pos >= UNWIND_AGGRESSIVE_THRESHOLD
    order_type = OrderType.IOC if aggressive else OrderType.GTC

    _unwind_last[product] = now

    if pos > 0:
        # Too long → sell to reduce
        if aggressive:
            price = snap.best_bid  # hit the bid to get out
        else:
            price = snap.mid + snap.spread * 0.2  # passive sell above mid
        return Signal(
            product=product,
            side=Side.SELL,
            price=price,
            volume=vol,
            order_type=order_type,
            reason=f"UNWIND pos={pos:+d} vol={vol} {'AGG' if aggressive else 'PASS'}",
        )
    else:
        # Too short → buy to reduce
        if aggressive:
            price = snap.best_ask  # lift the ask to get out
        else:
            price = snap.mid - snap.spread * 0.2  # passive buy below mid
        return Signal(
            product=product,
            side=Side.BUY,
            price=price,
            volume=vol,
            order_type=order_type,
            reason=f"UNWIND pos={pos:+d} vol={vol} {'AGG' if aggressive else 'PASS'}",
        )


# =========================================================================
# STRATEGY 7: ETF MARKET MAKING
# =========================================================================

ETF_MM_WIDTH = 15.0
ETF_MM_INTERVAL = 3.0
_etf_mm_last: float = 0.0


def etf_mm_strategy(bot: AlgothonBot, snap: OrderBookSnapshot) -> Optional[Signal]:
    """Market make LON_ETF with quotes anchored to component-implied fair value."""
    global _etf_mm_last

    _update_tracker(snap)

    if snap.product != "LON_ETF":
        return None

    now = time.time()
    if now - _etf_mm_last < ETF_MM_INTERVAL:
        return None

    fair = bot.fair_etf
    if math.isnan(fair):
        return None

    skew = snap.imbalance * 5.0
    bid = fair - ETF_MM_WIDTH + skew
    ask = fair + ETF_MM_WIDTH + skew

    if bid <= 0 or bid >= ask:
        return None

    _etf_mm_last = now

    return Signal(
        product="LON_ETF",
        side=Side.BUY,
        price=bid,
        volume=2,
        order_type=OrderType.QUOTE,
        ask_price=ask,
        ask_volume=2,
        reason=f"ETF_MM fair={fair:.0f} skew={skew:+.1f}",
    )


# =========================================================================
# TRADE LOGGER
# =========================================================================

def trade_logger(bot: AlgothonBot, trade: dict) -> None:
    """Print every fill as it happens."""
    side = "BOUGHT" if trade.get("buyer") == bot.username else "SOLD"
    print(f"  [FILL] {side} {trade['volume']}x {trade['product']} "
          f"@ {trade['price']}")


# =========================================================================
# PUBLIC: Get tracker for dashboard/external use
# =========================================================================

def get_tracker() -> _PriceTracker:
    """Access the shared price tracker for dashboard metrics."""
    return _tracker


# =========================================================================
# STRATEGY 8: VOLATILITY-ADAPTIVE LON_ETF MARKET MAKING
# =========================================================================
# A wide-biased market maker for LON_ETF that measures realised volatility
# every tick and widens/narrows the spread accordingly.  Inventory skew
# shifts quotes to mean-revert positions.
#
# This REPLACES the simpler etf_mm_strategy when registered — you can run
# both, but they will compete for the same product.
# =========================================================================

# ── Configuration ─────────────────────────────────────────────────────────
_VMM_BASE_HALF_SPREAD      = 15       # minimum half-spread (wide bias)
_VMM_VOL_SPREAD_MULT       = 3.0      # how much to widen on high vol
_VMM_MAX_HALF_SPREAD       = 80       # cap
_VMM_MIN_HALF_SPREAD       = 10       # floor
_VMM_BASE_ORDER_SIZE       = 1        # contracts per quote
_VMM_MAX_ORDER_SIZE        = 5        # hard cap
_VMM_VOL_WINDOW            = 30       # mid-price observations for vol calc
_VMM_VOL_LOW               = 5.0      # below → "low vol" regime
_VMM_VOL_HIGH              = 20.0     # above → "high vol" regime
_VMM_SKEW_PER_UNIT         = 0.5      # price skew per unit of inventory
_VMM_REQUOTE_INTERVAL      = 2.0      # min seconds between quotes
_VMM_SOFT_LIMIT            = 40       # start reducing size
_VMM_HARD_LIMIT            = 50       # stop quoting offending side


@dataclass
class _VMMState:
    """Internal state for the vol-adaptive market maker."""
    mid_history: deque = field(default_factory=lambda: deque(maxlen=_VMM_VOL_WINDOW))
    last_quote_t: float = 0.0
    tick_count: int = 0

    def update_mid(self, mid: float) -> None:
        self.mid_history.append(mid)

    def realised_vol(self) -> float:
        """MAD-based realised volatility of mid-price changes."""
        h = self.mid_history
        if len(h) < 4:
            return _VMM_VOL_LOW  # not enough data → assume low vol
        diffs = [abs(h[i] - h[i - 1]) for i in range(1, len(h))]
        mean_d = sum(diffs) / len(diffs)
        var = sum((d - mean_d) ** 2 for d in diffs) / len(diffs)
        return math.sqrt(var) + mean_d  # robust to outliers

    def half_spread(self) -> float:
        """Dynamic half-spread that widens with volatility."""
        vol = self.realised_vol()
        if vol <= _VMM_VOL_LOW:
            hs = _VMM_BASE_HALF_SPREAD
        elif vol >= _VMM_VOL_HIGH:
            hs = _VMM_BASE_HALF_SPREAD + _VMM_VOL_SPREAD_MULT * vol
        else:
            blend = (vol - _VMM_VOL_LOW) / (_VMM_VOL_HIGH - _VMM_VOL_LOW)
            hs = _VMM_BASE_HALF_SPREAD + _VMM_VOL_SPREAD_MULT * blend * vol
        return max(_VMM_MIN_HALF_SPREAD, min(_VMM_MAX_HALF_SPREAD, hs))


_vmm_state = _VMMState()


def vol_mm_strategy(bot: AlgothonBot, snap: OrderBookSnapshot) -> Optional[Signal]:
    """Volatility-adaptive market maker for LON_ETF.

    - Computes realised vol from a rolling window of mid-prices.
    - Widens spread when vol is high, narrows (but stays wide) when low.
    - Applies inventory skew to mean-revert positions.
    - Alternates bid/ask signals each tick to refresh both sides.
    """
    _update_tracker(snap)

    if snap.product != "LON_ETF":
        return None

    now = time.time()
    if now - _vmm_state.last_quote_t < _VMM_REQUOTE_INTERVAL:
        return None

    mid = snap.mid
    if math.isnan(mid):
        return None

    _vmm_state.update_mid(mid)
    _vmm_state.tick_count += 1

    # ── Compute spread ────────────────────────────────────────────────────
    hs = _vmm_state.half_spread()
    vol = _vmm_state.realised_vol()

    # ── Inventory skew (from cached positions — no API call) ─────────────
    position = get_position("LON_ETF")

    skew = position * _VMM_SKEW_PER_UNIT  # positive pos → push prices down
    adj_mid = mid - skew

    bid_price = int(round(adj_mid - hs))
    ask_price = int(round(adj_mid + hs))
    if bid_price >= ask_price:
        bid_price = ask_price - 1

    # ── Order sizing ──────────────────────────────────────────────────────
    order_size = min(_VMM_BASE_ORDER_SIZE, _VMM_MAX_ORDER_SIZE)
    if abs(position) >= _VMM_SOFT_LIMIT:
        order_size = max(1, order_size - 1)

    # ── Alternate bid/ask; respect position limits ────────────────────────
    quote_buy = (_vmm_state.tick_count % 2 == 0)
    if position >= _VMM_HARD_LIMIT:
        quote_buy = False  # too long, only post asks
    if position <= -_VMM_HARD_LIMIT:
        quote_buy = True   # too short, only post bids

    _vmm_state.last_quote_t = now

    if quote_buy:
        return Signal(
            product="LON_ETF",
            side=Side.BUY,
            price=float(bid_price),
            volume=order_size,
            order_type=OrderType.GTC,
            reason=f"VOL_MM BID mid={mid:.0f} hs={hs:.0f} vol={vol:.1f} skew={skew:+.0f} pos={position}",
        )
    else:
        return Signal(
            product="LON_ETF",
            side=Side.SELL,
            price=float(ask_price),
            volume=order_size,
            order_type=OrderType.GTC,
            reason=f"VOL_MM ASK mid={mid:.0f} hs={hs:.0f} vol={vol:.1f} skew={skew:+.0f} pos={position}",
        )
