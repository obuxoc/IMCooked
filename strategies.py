"""Algothon 2026 — Strategy functions for Market Making and Arbitrage.

TEAMMATE CONTRACT:
    Every strategy function has the signature:

        def my_strategy(bot, snap) -> Signal | None

    - Return a Signal to trade.
    - Return None to do nothing.
    - NEVER call bot.send_order() directly — the executor handles that.
    - NEVER worry about risk limits — the RiskManager handles that.

    You have full read access to:
        snap.mid, snap.spread, snap.micro_price, snap.imbalance, ...
        bot.latest["TIDE_SPOT"], bot.mids, bot.history("WX_SPOT", n=50)
        bot.fair_etf, bot.etf_arb_gap, bot.cache.arb_snapshot()

IMPORTS YOU NEED:
    from signals import Signal, Side, OrderType
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Optional

from signals import Signal, Side, OrderType

if TYPE_CHECKING:
    from algothon_bot import AlgothonBot
    from data_cache import OrderBookSnapshot


# =========================================================================
# STRATEGY 1: MARKET MAKING (returns Signal)
# =========================================================================
#
# Teammates: edit the logic below. Your job is to decide WHAT to quote.
# The executor + risk manager decide WHETHER and HOW to send it.
#
# Key data available:
#   snap.micro_price  — volume-weighted fair value (better than naive mid)
#   snap.imbalance    — order-book imbalance [-1, 1]
#   snap.spread       — current market spread
#   snap.best_bid / best_ask
#   snap.total_bid_vol / total_ask_vol
#   bot.history("TIDE_SPOT", n=50) — recent snapshots as DataFrame

# --- CONFIG (teammates can tune these) ---
MM_PRODUCTS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT", "TIDE_SWING",
               "WX_SUM", "LHR_INDEX", "LON_ETF"]
MM_WIDTH = 5.0            # half-spread around fair value
MM_SKEW_FACTOR = 2.0      # how much imbalance shifts quotes
MM_VOLUME = 3             # default quote size
MM_MIN_INTERVAL = 2.0     # seconds between requotes per product

_mm_last_quote: dict[str, float] = {}


def mm_strategy(bot: AlgothonBot, snap: OrderBookSnapshot) -> Optional[Signal]:
    """Market-making strategy — returns a QUOTE Signal or None.

    Teammates: replace / enhance the logic below.
    """
    product = snap.product
    if product not in MM_PRODUCTS:
        return None

    # Rate limit: don't requote too often
    now = time.time()
    if now - _mm_last_quote.get(product, 0) < MM_MIN_INTERVAL:
        return None

    # Skip if book is empty
    if math.isnan(snap.mid):
        return None

    # --- FAIR VALUE ESTIMATE ---
    # Start with micro_price; upgrade with external data signals later
    fair = snap.micro_price

    # --- SKEW based on order-book imbalance ---
    skew = snap.imbalance * MM_SKEW_FACTOR

    # --- COMPUTE quote prices ---
    bid_price = fair - MM_WIDTH + skew
    ask_price = fair + MM_WIDTH + skew

    if bid_price <= 0 or bid_price >= ask_price:
        return None

    # Don't re-quote if we already have volume resting
    if snap.best_bid_own >= MM_VOLUME and snap.best_ask_own >= MM_VOLUME:
        return None

    _mm_last_quote[product] = now

    # Return a two-sided QUOTE signal — executor handles the rest
    return Signal(
        product=product,
        side=Side.BUY,          # bid side
        price=bid_price,
        volume=MM_VOLUME,
        order_type=OrderType.QUOTE,
        ask_price=ask_price,    # ask side
        ask_volume=MM_VOLUME,
        reason=f"MM fair={fair:.0f} skew={skew:+.1f}",
    )


# =========================================================================
# STRATEGY 2: ETF ARBITRAGE (returns Signal)
# =========================================================================
#
# LON_ETF settles at TIDE_SPOT + WX_SPOT + LHR_COUNT.
# If the ETF trades away from component sum → trade the spread.
#
# Key data:
#   bot.fair_etf      — sum of component mids
#   bot.etf_arb_gap   — ETF mid minus fair value
#   bot.latest["LON_ETF"].best_bid / best_ask

# --- CONFIG ---
ARB_THRESHOLD = 10.0      # min gap (ticks) before acting
ARB_VOLUME = 2            # per-leg volume
ARB_MIN_INTERVAL = 3.0    # seconds between arb trades

_arb_last_trade: float = 0.0


def arb_strategy(bot: AlgothonBot, snap: OrderBookSnapshot) -> Optional[Signal]:
    """ETF vs components arbitrage — returns a Signal or None.

    Teammates: tune the threshold, add smarter entry logic, etc.
    NOTE: this currently only returns the ETF leg signal. For a full
    multi-leg arb, you'd return multiple signals or handle component
    legs separately. Keep it simple for now — one signal at a time.
    """
    global _arb_last_trade

    relevant = {"LON_ETF", "TIDE_SPOT", "WX_SPOT", "LHR_COUNT"}
    if snap.product not in relevant:
        return None

    now = time.time()
    if now - _arb_last_trade < ARB_MIN_INTERVAL:
        return None

    gap = bot.etf_arb_gap
    if math.isnan(gap):
        return None

    etf = bot.latest.get("LON_ETF")
    if etf is None or math.isnan(etf.best_bid) or math.isnan(etf.best_ask):
        return None

    # --- ETF overpriced → SELL ETF ---
    if gap > ARB_THRESHOLD:
        _arb_last_trade = now
        return Signal(
            product="LON_ETF",
            side=Side.SELL,
            price=etf.best_bid,          # aggress into best bid
            volume=ARB_VOLUME,
            order_type=OrderType.IOC,     # don't rest on book
            reason=f"ARB gap={gap:+.1f} SELL ETF",
            urgency=0.9,
        )

    # --- ETF underpriced → BUY ETF ---
    elif gap < -ARB_THRESHOLD:
        _arb_last_trade = now
        return Signal(
            product="LON_ETF",
            side=Side.BUY,
            price=etf.best_ask,          # lift the ask
            volume=ARB_VOLUME,
            order_type=OrderType.IOC,
            reason=f"ARB gap={gap:+.1f} BUY ETF",
            urgency=0.9,
        )

    return None


# =========================================================================
# UTILITY: LON_FLY FAIR VALUE
# =========================================================================

def compute_fly_fair(etf_value: float) -> float:
    """Compute LON_FLY settlement value given an ETF settlement value.

    LON_FLY = 2*Put(6200) + Call(6200) - 2*Call(6600) + 3*Call(7000)
    """
    put_6200 = max(0, 6200 - etf_value)
    call_6200 = max(0, etf_value - 6200)
    call_6600 = max(0, etf_value - 6600)
    call_7000 = max(0, etf_value - 7000)
    return 2 * put_6200 + call_6200 - 2 * call_6600 + 3 * call_7000


# =========================================================================
# TRADE LOGGER (utility — register as trade strategy)
# =========================================================================

def trade_logger(bot: AlgothonBot, trade: dict) -> None:
    """Print every fill as it happens."""
    side = "BOUGHT" if trade["buyer"] == bot.username else "SOLD"
    print(f"  [FILL] {side} {trade['volume']}x {trade['product']} "
          f"@ {trade['price']}")
