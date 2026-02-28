"""Algothon 2026 — Main trading bot with integrated data cache.

This is the bot your team actually runs.  It:
  1. Connects to the CMI Exchange via SSE
  2. On every orderbook tick → extracts a snapshot → stores in DataCache
  3. On every trade → records it
  4. Exposes the DataCache to strategy modules via simple properties
  5. Calls registered strategy callbacks so teammates plug in independently

QUICK START:
    from algothon_bot import AlgothonBot

    bot = AlgothonBot(URL, USER, PASS)
    bot.start()

    # Read data any time:
    print(bot.latest["TIDE_SPOT"].mid)
    print(bot.mids)
    print(bot.history("WX_SPOT", n=20))
    print(bot.etf_arb_gap)

    # Register strategy callbacks:
    bot.register_orderbook_strategy(my_market_maker)
    bot.register_orderbook_strategy(my_arb_strategy)

    # Stop:
    bot.stop()
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

from bot_template import (
    BaseBot, OrderBook, Order, OrderRequest, OrderResponse, Trade, Side, Product
)
from data_cache import DataCache, OrderBookSnapshot


class AlgothonBot(BaseBot):
    """Production bot with built-in data caching and strategy dispatch.

    Strategies are plain functions registered as callbacks:
        def my_strategy(bot: AlgothonBot, snap: OrderBookSnapshot):
            ...
    """

    def __init__(self, cmi_url: str, username: str, password: str,
                 cache_size: int = 5_000):
        super().__init__(cmi_url, username, password)
        self.cache = DataCache(maxlen=cache_size)

        # Strategy callbacks — executed in order on every orderbook tick
        self._ob_strategies: list[Callable[[AlgothonBot, OrderBookSnapshot], None]] = []
        # Trade callbacks
        self._trade_strategies: list[Callable[[AlgothonBot, dict], None]] = []

        # Rate-limit guard: track last order send time
        self._last_request_time: float = 0.0

        # Products cache
        self._products: dict[str, Product] = {}

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to exchange and start consuming SSE."""
        # Pre-fetch products on start
        for p in self.get_products():
            self._products[p.symbol] = p
        super().start()
        print(f"[AlgothonBot] Connected as '{self.username}', "
              f"tracking {len(self._products)} products")

    # ------------------------------------------------------------------
    # SSE CALLBACKS (called by _SSEThread) 
    # ------------------------------------------------------------------

    def on_orderbook(self, orderbook: OrderBook) -> None:
        """Called on every SSE orderbook event. Do NOT override — register strategies instead."""
        snap = self.cache.update_orderbook(orderbook)

        # Dispatch to registered strategies
        for strategy_fn in self._ob_strategies:
            try:
                strategy_fn(self, snap)
            except Exception as e:
                print(f"[Strategy Error] {strategy_fn.__name__}: {e}")

    def on_trades(self, trade: Trade) -> None:
        """Called on every trade event."""
        t = self.cache.record_trade(trade)

        for strategy_fn in self._trade_strategies:
            try:
                strategy_fn(self, t)
            except Exception as e:
                print(f"[Trade Strategy Error] {strategy_fn.__name__}: {e}")

    # ------------------------------------------------------------------
    # STRATEGY REGISTRATION
    # ------------------------------------------------------------------

    def register_orderbook_strategy(
        self, fn: Callable[[AlgothonBot, OrderBookSnapshot], None]
    ) -> None:
        """Register a function to be called on every orderbook update.

        Signature: fn(bot: AlgothonBot, snap: OrderBookSnapshot) -> None
        """
        self._ob_strategies.append(fn)
        print(f"[AlgothonBot] Registered OB strategy: {fn.__name__}")

    def register_trade_strategy(
        self, fn: Callable[[AlgothonBot, dict], None]
    ) -> None:
        """Register a function to be called on every trade.

        Signature: fn(bot: AlgothonBot, trade_dict: dict) -> None
        """
        self._trade_strategies.append(fn)
        print(f"[AlgothonBot] Registered trade strategy: {fn.__name__}")

    # ------------------------------------------------------------------
    # CONVENIENCE ACCESSORS (delegates to cache)
    # ------------------------------------------------------------------

    @property
    def latest(self) -> dict[str, OrderBookSnapshot]:
        """Latest snapshot per product. Usage: bot.latest["TIDE_SPOT"].mid"""
        return self.cache.latest

    @property
    def mids(self) -> dict[str, float]:
        """Current mid prices: {"TIDE_SPOT": 3500.0, ...}"""
        return self.cache.mids

    @property
    def fair_etf(self) -> float:
        """Fair LON_ETF value from component mids."""
        return self.cache.fair_etf

    @property
    def etf_arb_gap(self) -> float:
        """LON_ETF mid - fair value. Positive = ETF overpriced."""
        return self.cache.etf_arb_gap

    def history(self, product: str, n: int | None = None):
        """Last n orderbook snapshots for a product as DataFrame."""
        return self.cache.history(product, n)

    def trade_history(self, product: str | None = None, n: int | None = None):
        """Trade history as DataFrame."""
        return self.cache.trade_history(product, n)

    # ------------------------------------------------------------------
    # SAFE ORDER HELPERS (rate-limit aware)
    # ------------------------------------------------------------------

    def safe_send_order(self, order: OrderRequest) -> OrderResponse | None:
        """Send order with built-in 1-second rate limit guard."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()
        return self.send_order(order)

    def send_ioc(self, order: OrderRequest) -> OrderResponse | None:
        """Immediate-or-cancel: send + cancel remainder."""
        resp = self.safe_send_order(order)
        if resp and resp.volume > 0:
            self.cancel_order(resp.id)
        return resp

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    def round_price(self, product: str, price: float, side: Side) -> float:
        """Round price to valid tick. BUY rounds down, SELL rounds up."""
        tick = self._products[product].tickSize if product in self._products else 1.0
        if side == Side.BUY:
            return math.floor(price / tick) * tick
        else:
            return math.ceil(price / tick) * tick

    def print_state(self) -> None:
        """Print a human-readable summary of all products."""
        print(f"\n{'='*70}")
        print(f"  {'Product':<15} {'Bid':>8} {'Ask':>8} {'Mid':>8} {'Sprd':>6} "
              f"{'Imbal':>7} {'BidVol':>7} {'AskVol':>7}")
        print(f"  {'-'*70}")
        for sym in sorted(self.cache.products):
            s = self.cache.latest.get(sym)
            if s:
                print(f"  {sym:<15} {s.best_bid:>8.1f} {s.best_ask:>8.1f} "
                      f"{s.mid:>8.1f} {s.spread:>6.1f} {s.imbalance:>7.3f} "
                      f"{s.total_bid_vol:>7} {s.total_ask_vol:>7}")

        gap = self.etf_arb_gap
        if not math.isnan(gap):
            print(f"\n  ETF arb gap: {gap:+.1f}  (fair={self.fair_etf:.1f})")
        print(f"{'='*70}\n")
