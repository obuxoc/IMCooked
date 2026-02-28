"""Algothon 2026 — Signal, Risk Management, and Execution.

ARCHITECTURE:
    Teammate strategy fn(bot, snap) → Signal | None    ← teammates own this
    Signal → RiskManager.check(signal) → approved/rejected  ← you own this
    Approved signal → Executor.execute(signal) → order sent  ← you own this

TEAMMATES ONLY NEED TO KNOW:
    from signals import Signal, Side

    def my_strategy(bot, snap) -> Signal | None:
        if <my condition>:
            return Signal(
                product="TIDE_SPOT",
                side=Side.BUY,
                price=snap.best_ask,     # or whatever price
                volume=5,
                reason="imbalance > 0.3",  # for logging
            )
        return None  # no signal = do nothing
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from algothon_bot import AlgothonBot


# =========================================================================
# SIGNAL — the contract between strategies and execution
# =========================================================================

class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """How the executor should handle the order."""
    GTC = "GTC"      # Good-til-cancel (rests on book)
    IOC = "IOC"      # Immediate-or-cancel (aggress only)
    QUOTE = "QUOTE"  # Two-sided quote (bid + ask pair)


@dataclass(frozen=True)
class Signal:
    """What your teammates return from their strategy functions.

    Required fields:
        product:  which product to trade ("TIDE_SPOT", etc.)
        side:     Side.BUY or Side.SELL
        price:    desired price (will be tick-rounded by executor)
        volume:   desired quantity

    Optional fields:
        order_type:  GTC (default), IOC, or QUOTE
        reason:      human-readable string for logging
        urgency:     0.0 (low) to 1.0 (high) — executor can prioritize
        ask_price:   only for QUOTE type — the ask side price
        ask_volume:  only for QUOTE type — the ask side volume
    """
    product: str
    side: Side
    price: float
    volume: int
    order_type: OrderType = OrderType.GTC
    reason: str = ""
    urgency: float = 0.5
    # For two-sided quotes (QUOTE type)
    ask_price: Optional[float] = None
    ask_volume: Optional[int] = None


# =========================================================================
# RISK MANAGER — you own this, teammates never touch it
# =========================================================================

@dataclass
class RiskConfig:
    """All risk limits in one place. Tune these."""
    max_position_per_product: int = 50      # max abs net position per product
    max_total_exposure: int = 200           # sum of abs positions across all products
    max_order_size: int = 20                # single order volume cap
    max_orders_per_minute: int = 50         # rate limit (exchange allows 60/min)
    min_spread_to_quote: float = 2.0        # don't market-make if spread < this
    max_loss_threshold: float = -5000.0     # PnL floor — stop trading if breached
    cooldown_after_loss: float = 30.0       # seconds to pause after hitting loss limit
    max_signals_per_product_per_sec: float = 1.0  # throttle per product


class RiskManager:
    """Checks every signal before execution. Rejects anything risky."""

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self._positions: dict[str, int] = {}  # product -> net position
        self._order_count_window: list[float] = []  # timestamps of recent orders
        self._last_signal_time: dict[str, float] = {}  # product -> last signal ts
        self._trading_halted = False
        self._halt_reason = ""
        self._halt_until: float = 0.0
        # Stats
        self.signals_received = 0
        self.signals_approved = 0
        self.signals_rejected = 0
        self._rejection_log: list[dict] = []

    def update_positions(self, positions: dict[str, int]) -> None:
        """Called periodically to sync positions from exchange."""
        self._positions = dict(positions)

    def check(self, signal: Signal, bot: AlgothonBot) -> tuple[bool, str]:
        """Evaluate a signal. Returns (approved: bool, reason: str).

        If approved=False, reason explains why.
        """
        self.signals_received += 1

        # --- HALT CHECK ---
        if self._trading_halted:
            if time.time() < self._halt_until:
                return self._reject(signal, f"HALTED: {self._halt_reason}")
            else:
                self._trading_halted = False
                print("[RISK] Trading resumed after cooldown")

        # --- ORDER SIZE ---
        vol = signal.volume
        if vol > self.config.max_order_size:
            return self._reject(signal, f"Order size {vol} > max {self.config.max_order_size}")

        if vol <= 0:
            return self._reject(signal, "Volume must be positive")

        # --- POSITION LIMITS ---
        current_pos = self._positions.get(signal.product, 0)
        if signal.side == Side.BUY:
            new_pos = current_pos + vol
        else:
            new_pos = current_pos - vol

        if abs(new_pos) > self.config.max_position_per_product:
            return self._reject(signal,
                f"Position would be {new_pos} (limit ±{self.config.max_position_per_product})")

        # --- TOTAL EXPOSURE ---
        projected_exposure = sum(
            abs(v) for k, v in self._positions.items() if k != signal.product
        ) + abs(new_pos)
        if projected_exposure > self.config.max_total_exposure:
            return self._reject(signal,
                f"Total exposure would be {projected_exposure} (limit {self.config.max_total_exposure})")

        # --- RATE LIMIT ---
        now = time.time()
        self._order_count_window = [t for t in self._order_count_window if now - t < 60]
        if len(self._order_count_window) >= self.config.max_orders_per_minute:
            return self._reject(signal, "Rate limit: too many orders this minute")

        # --- PER-PRODUCT THROTTLE ---
        last_t = self._last_signal_time.get(signal.product, 0)
        min_gap = 1.0 / self.config.max_signals_per_product_per_sec
        if now - last_t < min_gap:
            return self._reject(signal, f"Throttled: {signal.product} signal too soon")

        # --- PRICE SANITY ---
        if signal.price <= 0:
            return self._reject(signal, "Price must be positive")

        # --- SPREAD CHECK (for quotes) ---
        if signal.order_type == OrderType.QUOTE:
            if signal.ask_price is not None and signal.price >= signal.ask_price:
                return self._reject(signal, "Quote bid >= ask")

        # ✅ APPROVED
        self.signals_approved += 1
        self._last_signal_time[signal.product] = now
        self._order_count_window.append(now)

        # Track projected position
        self._positions[signal.product] = new_pos

        return True, "OK"

    def halt_trading(self, reason: str, cooldown: float | None = None) -> None:
        """Emergency stop all trading."""
        self._trading_halted = True
        self._halt_reason = reason
        self._halt_until = time.time() + (cooldown or self.config.cooldown_after_loss)
        print(f"[RISK] ⛔ TRADING HALTED: {reason}")

    def check_pnl(self, bot: AlgothonBot) -> None:
        """Check PnL and halt if threshold breached."""
        try:
            pnl_data = bot.get_pnl()
            total_pnl = pnl_data.get("totalPnL", pnl_data.get("total", 0))
            if isinstance(total_pnl, (int, float)) and total_pnl < self.config.max_loss_threshold:
                self.halt_trading(
                    f"PnL {total_pnl:.0f} < threshold {self.config.max_loss_threshold:.0f}",
                    self.config.cooldown_after_loss
                )
        except Exception:
            pass  # PnL endpoint may not always be available

    def _reject(self, signal: Signal, reason: str) -> tuple[bool, str]:
        self.signals_rejected += 1
        self._rejection_log.append({
            "time": time.time(),
            "product": signal.product,
            "side": signal.side,
            "volume": signal.volume,
            "reason": reason,
        })
        return False, reason

    def print_stats(self) -> None:
        """Print risk manager statistics."""
        total = self.signals_received or 1
        print(f"[RISK] Signals: {self.signals_received} received, "
              f"{self.signals_approved} approved ({self.signals_approved/total:.0%}), "
              f"{self.signals_rejected} rejected")
        print(f"[RISK] Positions: {self._positions}")
        if self._trading_halted:
            print(f"[RISK] ⛔ HALTED: {self._halt_reason}")


# =========================================================================
# EXECUTOR — sends orders after risk approval
# =========================================================================

class Executor:
    """Takes approved signals and converts them into exchange orders.

    You control all execution logic here — IOC vs GTC, order tracking, etc.
    """

    def __init__(self, bot: AlgothonBot, risk: RiskManager):
        self.bot = bot
        self.risk = risk
        self._active_orders: dict[str, dict] = {}  # order_id -> info
        self.fills: list[dict] = []

    def execute(self, signal: Signal) -> bool:
        """Execute an approved signal. Returns True if order was sent."""
        from bot_template import OrderRequest, Side as BotSide

        if signal.order_type == OrderType.QUOTE:
            return self._execute_quote(signal)

        # Round price to valid tick
        bot_side = BotSide.BUY if signal.side == Side.BUY else BotSide.SELL
        price = self.bot.round_price(signal.product, signal.price, bot_side)

        if price <= 0:
            return False

        order = OrderRequest(
            product=signal.product,
            price=price,
            side=bot_side,
            volume=signal.volume,
        )

        if signal.order_type == OrderType.IOC:
            resp = self.bot.send_ioc(order)
        else:
            resp = self.bot.safe_send_order(order)

        if resp:
            self._active_orders[resp.id] = {
                "signal": signal,
                "response": resp,
                "time": time.time(),
            }
            logline = (f"[EXEC] {signal.side} {signal.volume}x {signal.product} "
                       f"@ {price:.0f} ({signal.order_type})")
            if signal.reason:
                logline += f" | {signal.reason}"
            print(logline)
            return True
        return False

    def _execute_quote(self, signal: Signal) -> bool:
        """Execute a two-sided quote signal."""
        from bot_template import OrderRequest, Side as BotSide

        bid_price = self.bot.round_price(signal.product, signal.price, BotSide.BUY)
        ask_price = self.bot.round_price(
            signal.product,
            signal.ask_price or signal.price + 10,
            BotSide.SELL
        )
        bid_vol = signal.volume
        ask_vol = signal.ask_volume or signal.volume

        if bid_price <= 0 or bid_price >= ask_price:
            return False

        orders = [
            OrderRequest(signal.product, bid_price, BotSide.BUY, bid_vol),
            OrderRequest(signal.product, ask_price, BotSide.SELL, ask_vol),
        ]
        resps = self.bot.send_orders(orders)
        for r in resps:
            self._active_orders[r.id] = {
                "signal": signal,
                "response": r,
                "time": time.time(),
            }
        print(f"[EXEC] QUOTE {signal.product} "
              f"{bid_vol}@{bid_price:.0f} / {ask_vol}@{ask_price:.0f}")
        return len(resps) > 0

    def cancel_stale_orders(self, max_age: float = 30.0) -> int:
        """Cancel orders older than max_age seconds. Returns count cancelled."""
        now = time.time()
        stale = [oid for oid, info in self._active_orders.items()
                 if now - info["time"] > max_age]
        for oid in stale:
            self.bot.cancel_order(oid)
            del self._active_orders[oid]
        return len(stale)


# =========================================================================
# STRATEGY DISPATCHER — wires everything together
# =========================================================================

def make_signal_dispatcher(executor: Executor, risk: RiskManager):
    """Create a dispatcher function that wraps a teammate's strategy.

    Usage:
        def teammates_strategy(bot, snap) -> Signal | None:
            ...

        dispatch = make_signal_dispatcher(executor, risk)
        bot.register_orderbook_strategy(dispatch(teammates_strategy))
    """

    def wrap(strategy_fn):
        """Wrap a signal-producing strategy with risk + execution."""

        def dispatched(bot: AlgothonBot, snap) -> None:
            signal = strategy_fn(bot, snap)
            if signal is None:
                return

            approved, reason = risk.check(signal, bot)
            if not approved:
                # Uncomment for debugging:
                # print(f"[RISK] Rejected {signal.product} {signal.side}: {reason}")
                return

            executor.execute(signal)

        dispatched.__name__ = f"dispatch_{strategy_fn.__name__}"
        return dispatched

    return wrap
