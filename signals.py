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
from collections import deque
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
        strategy:    name of the originating strategy (auto-filled by dispatcher)
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
    strategy: str = ""
    # For two-sided quotes (QUOTE type)
    ask_price: Optional[float] = None
    ask_volume: Optional[int] = None


# =========================================================================
# RISK MANAGER — comprehensive risk controls
# =========================================================================

@dataclass
class RiskConfig:
    """All risk limits in one place. Tune these."""
    # Position limits
    max_position_per_product: int = 50      # max abs net position per product
    max_total_exposure: int = 200           # sum of abs positions across all products
    max_order_size: int = 20                # single order volume cap

    # Rate limits
    max_orders_per_minute: int = 50         # exchange allows 60/min; leave headroom
    max_signals_per_product_per_sec: float = 1.0  # throttle per product

    # Spread / price
    min_spread_to_quote: float = 2.0        # don't market-make if spread < this

    # PnL controls
    max_loss_threshold: float = -5000.0     # total PnL floor — halt all trading
    max_drawdown: float = -2000.0           # max PnL drop from peak — halt
    cooldown_after_loss: float = 30.0       # seconds to pause after halt trigger

    # Per-strategy circuit breakers
    max_consecutive_losses_per_strategy: int = 10  # halt strategy after N losses in a row
    strategy_loss_reset_time: float = 120.0        # reset loss counter after this many seconds

    # Inventory management
    inventory_skew_threshold: int = 20      # abs position above which to reduce aggressively
    inventory_hard_limit: int = 40          # abs position above which ONLY reduce allowed
    max_notional_per_order: float = 100_000.0  # price * volume cap


class RiskManager:
    """Checks every signal before execution. Rejects anything risky.

    Features:
    - Position limits (per-product and total)
    - Drawdown tracking with automatic halt
    - Per-strategy circuit breakers
    - Inventory-aware order filtering (only allow risk-reducing when overloaded)
    - Rate limiting (global and per-product)
    - Full audit trail
    """

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self._positions: dict[str, int] = {}
        self._order_count_window: list[float] = []
        self._last_signal_time: dict[str, float] = {}
        self._trading_halted = False
        self._halt_reason = ""
        self._halt_until: float = 0.0

        # PnL tracking
        self._peak_pnl: float = 0.0
        self._current_pnl: float = 0.0
        self._pnl_history: deque[tuple[float, float]] = deque(maxlen=1000)  # (time, pnl)

        # Per-strategy tracking
        self._strategy_losses: dict[str, int] = {}     # strategy_name -> consecutive losses
        self._strategy_last_loss: dict[str, float] = {}  # strategy_name -> last loss time
        self._strategy_halted: set[str] = set()

        # Stats
        self.signals_received = 0
        self.signals_approved = 0
        self.signals_rejected = 0
        self._rejection_reasons: dict[str, int] = {}   # reason_category -> count
        self._rejection_log: deque[dict] = deque(maxlen=500)

        # Execution tracking for fill monitoring
        self._pending_signals: deque[dict] = deque(maxlen=200)

    def update_positions(self, positions: dict[str, int]) -> None:
        """Called periodically to sync positions from exchange."""
        self._positions = dict(positions)

    def update_pnl(self, pnl: float) -> None:
        """Update PnL tracking. Call this regularly."""
        self._current_pnl = pnl
        self._pnl_history.append((time.time(), pnl))
        if pnl > self._peak_pnl:
            self._peak_pnl = pnl

    def report_fill(self, strategy: str, pnl_impact: float) -> None:
        """Report a fill's PnL impact for per-strategy circuit breakers."""
        if pnl_impact < 0:
            self._strategy_losses[strategy] = self._strategy_losses.get(strategy, 0) + 1
            self._strategy_last_loss[strategy] = time.time()
            # Check circuit breaker
            if self._strategy_losses[strategy] >= self.config.max_consecutive_losses_per_strategy:
                self._strategy_halted.add(strategy)
                print(f"[RISK] Strategy '{strategy}' HALTED after "
                      f"{self._strategy_losses[strategy]} consecutive losses")
        else:
            # Reset on profitable fill
            self._strategy_losses[strategy] = 0

    def check(self, signal: Signal, bot: AlgothonBot) -> tuple[bool, str]:
        """Evaluate a signal. Returns (approved: bool, reason: str)."""
        self.signals_received += 1
        now = time.time()

        # --- GLOBAL HALT ---
        if self._trading_halted:
            if now < self._halt_until:
                return self._reject(signal, "halt", f"HALTED: {self._halt_reason}")
            else:
                self._trading_halted = False
                print("[RISK] Trading resumed after cooldown")

        # --- PER-STRATEGY HALT ---
        if signal.strategy and signal.strategy in self._strategy_halted:
            # Check if reset time elapsed
            last_loss = self._strategy_last_loss.get(signal.strategy, 0)
            if now - last_loss > self.config.strategy_loss_reset_time:
                self._strategy_halted.discard(signal.strategy)
                self._strategy_losses[signal.strategy] = 0
                print(f"[RISK] Strategy '{signal.strategy}' resumed after cooldown")
            else:
                return self._reject(signal, "strategy_halt",
                    f"Strategy '{signal.strategy}' halted (consecutive losses)")

        # --- ORDER SIZE ---
        vol = signal.volume
        if vol > self.config.max_order_size:
            return self._reject(signal, "order_size",
                f"Order size {vol} > max {self.config.max_order_size}")
        if vol <= 0:
            return self._reject(signal, "order_size", "Volume must be positive")

        # --- NOTIONAL CAP ---
        notional = signal.price * signal.volume
        if notional > self.config.max_notional_per_order:
            return self._reject(signal, "notional",
                f"Notional {notional:.0f} > max {self.config.max_notional_per_order:.0f}")

        # --- POSITION LIMITS ---
        current_pos = self._positions.get(signal.product, 0)
        if signal.side == Side.BUY:
            new_pos = current_pos + vol
        else:
            new_pos = current_pos - vol

        if abs(new_pos) > self.config.max_position_per_product:
            return self._reject(signal, "position",
                f"Position would be {new_pos} (limit ±{self.config.max_position_per_product})")

        # --- INVENTORY HARD LIMIT: only allow risk-reducing trades ---
        if abs(current_pos) >= self.config.inventory_hard_limit:
            is_reducing = (
                (current_pos > 0 and signal.side == Side.SELL) or
                (current_pos < 0 and signal.side == Side.BUY)
            )
            if not is_reducing:
                return self._reject(signal, "inventory_hard",
                    f"Position {current_pos} at hard limit — only reducing trades allowed")

        # --- TOTAL EXPOSURE ---
        projected_exposure = sum(
            abs(v) for k, v in self._positions.items() if k != signal.product
        ) + abs(new_pos)
        if projected_exposure > self.config.max_total_exposure:
            return self._reject(signal, "exposure",
                f"Total exposure would be {projected_exposure} (limit {self.config.max_total_exposure})")

        # --- DRAWDOWN CHECK ---
        drawdown = self._current_pnl - self._peak_pnl
        if drawdown < self.config.max_drawdown:
            self.halt_trading(
                f"Drawdown {drawdown:.0f} < threshold {self.config.max_drawdown:.0f}",
                self.config.cooldown_after_loss
            )
            return self._reject(signal, "drawdown", f"Drawdown halt triggered: {drawdown:.0f}")

        # --- RATE LIMIT ---
        self._order_count_window = [t for t in self._order_count_window if now - t < 60]
        if len(self._order_count_window) >= self.config.max_orders_per_minute:
            return self._reject(signal, "rate_limit", "Rate limit: too many orders this minute")

        # --- PER-PRODUCT THROTTLE ---
        last_t = self._last_signal_time.get(signal.product, 0)
        min_gap = 1.0 / self.config.max_signals_per_product_per_sec
        if now - last_t < min_gap:
            return self._reject(signal, "throttle",
                f"Throttled: {signal.product} signal too soon")

        # --- PRICE SANITY ---
        if signal.price <= 0:
            return self._reject(signal, "price", "Price must be positive")

        # --- SPREAD CHECK (for quotes) ---
        if signal.order_type == OrderType.QUOTE:
            if signal.ask_price is not None and signal.price >= signal.ask_price:
                return self._reject(signal, "quote_spread", "Quote bid >= ask")

        # APPROVED
        self.signals_approved += 1
        self._last_signal_time[signal.product] = now
        self._order_count_window.append(now)
        self._positions[signal.product] = new_pos

        return True, "OK"

    def halt_trading(self, reason: str, cooldown: float | None = None) -> None:
        """Emergency stop all trading."""
        self._trading_halted = True
        self._halt_reason = reason
        self._halt_until = time.time() + (cooldown or self.config.cooldown_after_loss)
        print(f"[RISK] TRADING HALTED: {reason}")

    def check_pnl(self, bot: AlgothonBot) -> None:
        """Check PnL and halt if threshold breached. Call from main loop."""
        try:
            pnl_data = bot.get_pnl()
            total_pnl = pnl_data.get("totalPnL", pnl_data.get("total", 0))
            if isinstance(total_pnl, (int, float)):
                self.update_pnl(total_pnl)
                if total_pnl < self.config.max_loss_threshold:
                    self.halt_trading(
                        f"PnL {total_pnl:.0f} < threshold {self.config.max_loss_threshold:.0f}",
                        self.config.cooldown_after_loss
                    )
        except Exception:
            pass

    def _reject(self, signal: Signal, category: str, reason: str) -> tuple[bool, str]:
        self.signals_rejected += 1
        self._rejection_reasons[category] = self._rejection_reasons.get(category, 0) + 1
        self._rejection_log.append({
            "time": time.time(),
            "product": signal.product,
            "side": str(signal.side),
            "volume": signal.volume,
            "strategy": signal.strategy,
            "category": category,
            "reason": reason,
        })
        return False, reason

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak PnL."""
        return self._current_pnl - self._peak_pnl

    @property
    def positions(self) -> dict[str, int]:
        return dict(self._positions)

    def get_stats(self) -> dict:
        """Return full stats as a dict (for dashboard)."""
        total = self.signals_received or 1
        return {
            "signals_received": self.signals_received,
            "signals_approved": self.signals_approved,
            "signals_rejected": self.signals_rejected,
            "approval_rate": self.signals_approved / total,
            "positions": dict(self._positions),
            "current_pnl": self._current_pnl,
            "peak_pnl": self._peak_pnl,
            "drawdown": self.drawdown,
            "halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "strategies_halted": list(self._strategy_halted),
            "rejection_breakdown": dict(self._rejection_reasons),
            "pnl_history": list(self._pnl_history),
        }

    def print_stats(self) -> None:
        """Print risk manager statistics."""
        total = self.signals_received or 1
        print(f"[RISK] Signals: {self.signals_received} recv, "
              f"{self.signals_approved} approved ({self.signals_approved/total:.0%}), "
              f"{self.signals_rejected} rejected")
        print(f"[RISK] Positions: {self._positions}")
        print(f"[RISK] PnL: {self._current_pnl:.0f} | Peak: {self._peak_pnl:.0f} | "
              f"Drawdown: {self.drawdown:.0f}")
        if self._rejection_reasons:
            print(f"[RISK] Rejections: {self._rejection_reasons}")
        if self._strategy_halted:
            print(f"[RISK] Halted strategies: {self._strategy_halted}")
        if self._trading_halted:
            print(f"[RISK] TRADING HALTED: {self._halt_reason}")


# =========================================================================
# DRY-RUN SIMULATOR — simulated fills, positions, mark-to-market PnL
# =========================================================================

class DryRunSimulator:
    """Tracks hypothetical fills from dry-run mode and computes simulated PnL.

    Assumes every approved signal fills immediately at its signal price.
    Marks positions to market using live mid prices every tick.
    Provides per-strategy attribution and a PnL time series for the dashboard.
    """

    def __init__(self):
        self._positions: dict[str, int] = {}          # product -> net qty
        self._avg_costs: dict[str, float] = {}        # product -> avg entry price
        self._realized_pnl: float = 0.0                # closed trade PnL
        self._unrealized_pnl: float = 0.0              # mark-to-market open PnL
        self._total_pnl: float = 0.0                   # realized + unrealized
        self._peak_pnl: float = 0.0
        self._trade_count: int = 0
        self._pnl_history: deque[tuple[float, float]] = deque(maxlen=2000)

        # Per-strategy tracking
        self._strategy_pnl: dict[str, float] = {}      # strategy -> realized PnL
        self._strategy_trades: dict[str, int] = {}     # strategy -> trade count
        self._strategy_volume: dict[str, int] = {}     # strategy -> total volume

        # Trade log for display
        self._trades: deque[dict] = deque(maxlen=500)

    def record_fill(self, product: str, side: str, price: float, volume: int,
                    strategy: str = "", reason: str = "") -> None:
        """Record a simulated fill. Computes realized PnL on position reduction."""
        self._trade_count += 1
        signed_qty = volume if side == "BUY" else -volume
        old_pos = self._positions.get(product, 0)
        old_cost = self._avg_costs.get(product, 0.0)
        new_pos = old_pos + signed_qty

        # Compute realized PnL when reducing position
        realized = 0.0
        if old_pos != 0 and ((old_pos > 0 and signed_qty < 0) or (old_pos < 0 and signed_qty > 0)):
            # We're reducing/flipping
            close_qty = min(abs(signed_qty), abs(old_pos))
            if old_pos > 0:  # was long, selling
                realized = close_qty * (price - old_cost)
            else:            # was short, buying
                realized = close_qty * (old_cost - price)
            self._realized_pnl += realized

        # Update average cost
        if new_pos == 0:
            self._avg_costs[product] = 0.0
        elif (old_pos >= 0 and signed_qty > 0) or (old_pos <= 0 and signed_qty < 0):
            # Adding to position — blend cost
            total_cost = abs(old_pos) * old_cost + abs(signed_qty) * price
            self._avg_costs[product] = total_cost / abs(new_pos) if new_pos != 0 else 0.0
        elif abs(new_pos) > 0 and ((old_pos > 0 and new_pos < 0) or (old_pos < 0 and new_pos > 0)):
            # Flipped through zero — new cost is fill price
            self._avg_costs[product] = price
        else:
            # Partial close — keep old cost
            pass

        self._positions[product] = new_pos

        # Per-strategy
        self._strategy_pnl[strategy] = self._strategy_pnl.get(strategy, 0.0) + realized
        self._strategy_trades[strategy] = self._strategy_trades.get(strategy, 0) + 1
        self._strategy_volume[strategy] = self._strategy_volume.get(strategy, 0) + volume

        self._trades.append({
            "time": time.time(), "product": product, "side": side,
            "price": price, "volume": volume, "strategy": strategy,
            "reason": reason, "realized_pnl": round(realized, 2),
            "position_after": new_pos,
        })

    def mark_to_market(self, mids: dict[str, float]) -> None:
        """Revalue open positions using current mid prices. Call periodically."""
        unrealized = 0.0
        for product, pos in self._positions.items():
            if pos == 0:
                continue
            mid = mids.get(product)
            if mid is None or mid != mid:  # NaN check
                continue
            cost = self._avg_costs.get(product, 0.0)
            if pos > 0:
                unrealized += pos * (mid - cost)
            else:
                unrealized += abs(pos) * (cost - mid)
        self._unrealized_pnl = unrealized
        self._total_pnl = self._realized_pnl + unrealized
        if self._total_pnl > self._peak_pnl:
            self._peak_pnl = self._total_pnl
        self._pnl_history.append((time.time(), round(self._total_pnl, 2)))

    @property
    def drawdown(self) -> float:
        return self._total_pnl - self._peak_pnl

    def get_stats(self) -> dict:
        """Full stats dict for dashboard and terminal."""
        return {
            "sim_positions": dict(self._positions),
            "sim_avg_costs": {k: round(v, 1) for k, v in self._avg_costs.items() if v},
            "sim_realized_pnl": round(self._realized_pnl, 2),
            "sim_unrealized_pnl": round(self._unrealized_pnl, 2),
            "sim_total_pnl": round(self._total_pnl, 2),
            "sim_peak_pnl": round(self._peak_pnl, 2),
            "sim_drawdown": round(self.drawdown, 2),
            "sim_trade_count": self._trade_count,
            "sim_strategy_pnl": {k: round(v, 2) for k, v in self._strategy_pnl.items()},
            "sim_strategy_trades": dict(self._strategy_trades),
            "sim_strategy_volume": dict(self._strategy_volume),
            "sim_pnl_history": [(t, pnl) for t, pnl in self._pnl_history],
            "sim_recent_trades": list(self._trades)[-20:],
            "sim_exposure": sum(abs(v) for v in self._positions.values()),
        }

    def print_summary(self) -> None:
        """Print simulated performance to terminal."""
        print(f"[SIM] PnL: {self._total_pnl:+.0f} "
              f"(realized={self._realized_pnl:+.0f} unrealized={self._unrealized_pnl:+.0f}) "
              f"| Peak: {self._peak_pnl:.0f} | DD: {self.drawdown:.0f} "
              f"| Trades: {self._trade_count}")
        if self._positions:
            pos_str = "  ".join(f"{k}:{v:+d}" for k, v in sorted(self._positions.items()) if v != 0)
            if pos_str:
                print(f"[SIM] Positions: {pos_str}")
        if self._strategy_pnl:
            strat_str = "  ".join(
                f"{k}:{v:+.0f}({self._strategy_trades.get(k,0)}t)"
                for k, v in sorted(self._strategy_pnl.items())
            )
            print(f"[SIM] By strategy: {strat_str}")


# =========================================================================
# EXECUTOR — sends orders after risk approval
# =========================================================================

class Executor:
    """Takes approved signals and converts them into exchange orders.

    You control all execution logic here — IOC vs GTC, order tracking, etc.
    """

    def __init__(self, bot: AlgothonBot, risk: RiskManager, dry_run: bool = False):
        self.bot = bot
        self.risk = risk
        self.dry_run = dry_run
        self._active_orders: dict[str, dict] = {}  # order_id -> info
        self.fills: list[dict] = []
        self._dry_run_log: list[dict] = []  # logged orders in dry-run mode
        # Simulator runs in ALL modes — shadow-tracks every approved signal
        self.simulator: DryRunSimulator = DryRunSimulator()

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

        if self.dry_run:
            entry = {"side": str(signal.side), "volume": signal.volume,
                     "product": signal.product, "price": price,
                     "type": str(signal.order_type), "reason": signal.reason,
                     "strategy": signal.strategy, "time": time.time()}
            self._dry_run_log.append(entry)
            if self.simulator:
                self.simulator.record_fill(
                    signal.product, str(signal.side), price, signal.volume,
                    strategy=signal.strategy, reason=signal.reason)
            print(f"[DRY-RUN] WOULD {signal.side} {signal.volume}x "
                  f"{signal.product} @ {price:.0f} ({signal.order_type})"
                  + (f" | {signal.reason}" if signal.reason else ""))
            return True  # pretend it was sent so risk/stats update normally

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
            # Shadow-track in simulator for live PnL chart
            self.simulator.record_fill(
                signal.product, str(signal.side), price, signal.volume,
                strategy=signal.strategy, reason=signal.reason)
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

        if self.dry_run:
            self._dry_run_log.append({"type": "QUOTE", "product": signal.product,
                                       "bid": bid_price, "ask": ask_price,
                                       "bid_vol": bid_vol, "ask_vol": ask_vol,
                                       "strategy": signal.strategy, "time": time.time()})
            if self.simulator:
                self.simulator.record_fill(
                    signal.product, "BUY", bid_price, bid_vol,
                    strategy=signal.strategy, reason=signal.reason + " [bid leg]")
                self.simulator.record_fill(
                    signal.product, "SELL", ask_price, ask_vol,
                    strategy=signal.strategy, reason=signal.reason + " [ask leg]")
            print(f"[DRY-RUN] WOULD QUOTE {signal.product} "
                  f"{bid_vol}@{bid_price:.0f} bid / {ask_vol}@{ask_price:.0f} ask")
            return True

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
        # Shadow-track both legs in simulator for live PnL chart
        if resps:
            self.simulator.record_fill(
                signal.product, "BUY", bid_price, bid_vol,
                strategy=signal.strategy, reason=signal.reason + " [bid leg]")
            self.simulator.record_fill(
                signal.product, "SELL", ask_price, ask_vol,
                strategy=signal.strategy, reason=signal.reason + " [ask leg]")
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
        strategy_name = strategy_fn.__name__

        def dispatched(bot: AlgothonBot, snap) -> None:
            signal = strategy_fn(bot, snap)
            if signal is None:
                return

            # Inject strategy name if not already set
            if not signal.strategy:
                signal = Signal(
                    product=signal.product,
                    side=signal.side,
                    price=signal.price,
                    volume=signal.volume,
                    order_type=signal.order_type,
                    reason=signal.reason,
                    urgency=signal.urgency,
                    strategy=strategy_name,
                    ask_price=signal.ask_price,
                    ask_volume=signal.ask_volume,
                )

            approved, reason = risk.check(signal, bot)
            if not approved:
                return

            executor.execute(signal)

        dispatched.__name__ = f"dispatch_{strategy_fn.__name__}"
        return dispatched

    return wrap
