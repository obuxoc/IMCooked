"""Algothon 2026 — Offline Backtester with Parameter Sweep.

Replays collected CSV data through strategies with configurable parameters.
No exchange connection needed — instant, deterministic, repeatable.

USAGE:
    # Quick single run (all strategies, default params):
    python backtest.py

    # Full parameter sweep across strategies:
    python backtest.py --sweep

    # Use data from a specific directory:
    python backtest.py --data-dir "%LOCALAPPDATA%\\algothon_data"
"""

from __future__ import annotations

import copy
import importlib
import itertools
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# We need data_cache.OrderBookSnapshot but NOT the exchange bot.
# Import the snapshot dataclass directly.
# ---------------------------------------------------------------------------
from data_cache import OrderBookSnapshot

# ---------------------------------------------------------------------------
# Lightweight mock bot so strategies can call bot.mids, bot.fair_etf, etc.
# ---------------------------------------------------------------------------

class MockBot:
    """Minimal stand-in for AlgothonBot — enough for strategies to run."""

    def __init__(self):
        self.username = "BACKTEST"
        self.latest: dict[str, OrderBookSnapshot] = {}
        self._mids: dict[str, float] = {}

    # -- properties strategies access --
    @property
    def mids(self) -> dict[str, float]:
        return dict(self._mids)

    @property
    def fair_etf(self) -> float:
        t = self._mids.get("TIDE_SPOT", float("nan"))
        w = self._mids.get("WX_SPOT", float("nan"))
        l = self._mids.get("LHR_COUNT", float("nan"))
        return t + w + l

    @property
    def etf_arb_gap(self) -> float:
        etf = self._mids.get("LON_ETF", float("nan"))
        fair = self.fair_etf
        if math.isnan(etf) or math.isnan(fair):
            return float("nan")
        return etf - fair

    def get_positions(self) -> dict[str, int]:
        """Return simulated positions (filled by the backtester)."""
        return dict(self._sim_positions) if hasattr(self, "_sim_positions") else {}

    def update(self, snap: OrderBookSnapshot) -> None:
        """Feed a snapshot in — updates latest & mids like the real bot."""
        self.latest[snap.product] = snap
        if not math.isnan(snap.mid):
            self._mids[snap.product] = snap.mid


# ---------------------------------------------------------------------------
# Simulated PnL tracker (subset of DryRunSimulator, self-contained)
# ---------------------------------------------------------------------------

class PnLTracker:
    """Tracks positions, avg cost, realized + unrealized PnL."""

    def __init__(self):
        self.positions: dict[str, int] = {}
        self.avg_costs: dict[str, float] = {}
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.peak_pnl: float = 0.0
        self.trade_count: int = 0
        self.strategy_pnl: dict[str, float] = {}
        self.strategy_trades: dict[str, int] = {}
        self.pnl_curve: list[tuple[float, float]] = []  # (timestamp, total_pnl)

    def record_fill(self, product: str, side: str, price: float,
                    volume: int, strategy: str = "") -> float:
        """Record a fill. Returns realized PnL from this trade."""
        self.trade_count += 1
        signed_qty = volume if side == "BUY" else -volume
        old_pos = self.positions.get(product, 0)
        old_cost = self.avg_costs.get(product, 0.0)
        new_pos = old_pos + signed_qty

        realized = 0.0
        if old_pos != 0 and (
            (old_pos > 0 and signed_qty < 0) or
            (old_pos < 0 and signed_qty > 0)
        ):
            close_qty = min(abs(signed_qty), abs(old_pos))
            if old_pos > 0:
                realized = close_qty * (price - old_cost)
            else:
                realized = close_qty * (old_cost - price)
            self.realized_pnl += realized

        # Update avg cost
        if new_pos == 0:
            self.avg_costs[product] = 0.0
        elif (old_pos >= 0 and signed_qty > 0) or (old_pos <= 0 and signed_qty < 0):
            total_cost = abs(old_pos) * old_cost + abs(signed_qty) * price
            self.avg_costs[product] = total_cost / abs(new_pos) if new_pos != 0 else 0.0
        elif abs(new_pos) > 0 and (
            (old_pos > 0 and new_pos < 0) or (old_pos < 0 and new_pos > 0)
        ):
            self.avg_costs[product] = price

        self.positions[product] = new_pos

        self.strategy_pnl[strategy] = self.strategy_pnl.get(strategy, 0.0) + realized
        self.strategy_trades[strategy] = self.strategy_trades.get(strategy, 0) + 1
        return realized

    def mark_to_market(self, mids: dict[str, float], ts: float = 0.0) -> None:
        unrealized = 0.0
        for product, pos in self.positions.items():
            if pos == 0:
                continue
            mid = mids.get(product)
            if mid is None or mid != mid:
                continue
            cost = self.avg_costs.get(product, 0.0)
            if pos > 0:
                unrealized += pos * (mid - cost)
            else:
                unrealized += abs(pos) * (cost - mid)
        self.unrealized_pnl = unrealized
        self.total_pnl = self.realized_pnl + unrealized
        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl
        self.pnl_curve.append((ts, round(self.total_pnl, 2)))

    @property
    def drawdown(self) -> float:
        return self.total_pnl - self.peak_pnl

    @property
    def exposure(self) -> int:
        return sum(abs(v) for v in self.positions.values())


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def load_snapshots(data_dir: str) -> list[OrderBookSnapshot]:
    """Load all CSVs, merge by timestamp, return chronological snapshots."""
    data_path = Path(data_dir)
    all_rows: list[dict] = []

    for csv_file in sorted(data_path.glob("*.csv")):
        if csv_file.stem == "trades":
            continue
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                all_rows.append(row.to_dict())
        except Exception as e:
            print(f"  Warning: couldn't load {csv_file}: {e}")

    # Sort by timestamp (interleave products chronologically)
    all_rows.sort(key=lambda r: r["timestamp"])
    print(f"  Loaded {len(all_rows)} snapshots from {data_path}")

    # Convert to OrderBookSnapshot objects
    snapshots = []
    nan = float("nan")
    for r in all_rows:
        try:
            snap = OrderBookSnapshot(
                timestamp=r["timestamp"],
                product=r["product"],
                best_bid=r.get("best_bid", nan),
                best_ask=r.get("best_ask", nan),
                mid=r.get("mid", nan),
                spread=r.get("spread", nan),
                micro_price=r.get("micro_price", nan),
                best_bid_vol=int(r.get("best_bid_vol", 0)),
                best_ask_vol=int(r.get("best_ask_vol", 0)),
                best_bid_own=int(r.get("best_bid_own", 0)),
                best_ask_own=int(r.get("best_ask_own", 0)),
                total_bid_vol=int(r.get("total_bid_vol", 0)),
                total_ask_vol=int(r.get("total_ask_vol", 0)),
                total_own_bid_vol=int(r.get("total_own_bid_vol", 0)),
                total_own_ask_vol=int(r.get("total_own_ask_vol", 0)),
                bid_levels=int(r.get("bid_levels", 0)),
                ask_levels=int(r.get("ask_levels", 0)),
                imbalance=r.get("imbalance", 0.0),
                top_imbalance=r.get("top_imbalance", 0.0),
                weighted_mid=r.get("weighted_mid", nan),
                tick_size=r.get("tick_size", 1.0),
                bid_prices_3=[nan, nan, nan],
                bid_vols_3=[0, 0, 0],
                ask_prices_3=[nan, nan, nan],
                ask_vols_3=[0, 0, 0],
            )
            snapshots.append(snap)
        except Exception:
            pass

    return snapshots


def _reset_strategy_module():
    """Reset strategies.py module-level state so each run is independent."""
    import strategies as strat_mod
    # Reset tracker
    strat_mod._tracker = strat_mod._PriceTracker()
    # Reset timing state
    strat_mod._grid_last_signal = {}
    strat_mod._grid_level_idx = {}
    strat_mod._arb_last_trade = 0.0
    strat_mod._comp_arb_last = {}
    strat_mod._fly_last = 0.0
    strat_mod._mr_last = {}
    strat_mod._unwind_last = {}
    strat_mod._etf_mm_last = 0.0
    # Reset vol MM state
    if hasattr(strat_mod, '_vmm_state'):
        strat_mod._vmm_state = strat_mod._VMMState()
    # Reset shared position cache
    strat_mod._positions = {}


def run_backtest(
    snapshots: list[OrderBookSnapshot],
    strategies: list[Callable],
    strategy_names: list[str] | None = None,
    params: dict | None = None,
    position_limit: int = 80,
    verbose: bool = False,
) -> dict:
    """Run strategies over historical snapshots. Returns results dict.

    Args:
        snapshots: Chronological list of OrderBookSnapshot objects.
        strategies: List of strategy callables (bot, snap) -> Signal | None.
        strategy_names: Optional names for each strategy. Auto-derived if None.
        params: Optional dict of param overrides to apply to strategies module.
                e.g. {"ARB_THRESHOLD": 20.0, "MR_THRESHOLD_SIGMAS": 2.0}
        position_limit: Max abs position per product.
        verbose: Print each signal.

    Returns:
        Dict with PnL results, stats, per-strategy attribution.
    """
    import strategies as strat_mod

    # Apply parameter overrides
    _reset_strategy_module()
    if params:
        for key, value in params.items():
            if hasattr(strat_mod, key):
                setattr(strat_mod, key, value)
            # Also handle nested _GRID_CONFIG overrides like "GRID_LHR_COUNT_width_pct"
            elif key.startswith("GRID_") and "_" in key[5:]:
                parts = key[5:].rsplit("_", 1)
                product = parts[0]
                param_name = parts[1]
                # Try to find the product key (e.g., GRID_LHR_COUNT_width might be split wrong)
                for gp in strat_mod._GRID_CONFIG:
                    gp_prefix = gp.replace(" ", "_")
                    if key[5:].startswith(gp_prefix + "_"):
                        param_name = key[5 + len(gp_prefix) + 1:]
                        if param_name in strat_mod._GRID_CONFIG[gp]:
                            strat_mod._GRID_CONFIG[gp][param_name] = value
                        break

    if strategy_names is None:
        strategy_names = [getattr(fn, "__name__", f"strat_{i}") for i, fn in enumerate(strategies)]

    bot = MockBot()
    pnl = PnLTracker()
    bot._sim_positions = pnl.positions  # wire up so strategies see positions

    # Also push positions into strategy module's shared cache
    strat_mod._positions = pnl.positions

    signals_emitted = 0
    signals_by_strategy: dict[str, int] = {}
    start_time = time.time()

    for snap in snapshots:
        bot.update(snap)

        for fn, name in zip(strategies, strategy_names):
            try:
                signal = fn(bot, snap)
            except Exception as e:
                if verbose:
                    print(f"  [ERR] {name}: {e}")
                continue

            if signal is None:
                continue

            signals_emitted += 1
            signals_by_strategy[name] = signals_by_strategy.get(name, 0) + 1

            # --- Simple fill simulation ---
            # Assume the signal fills at its price if the price is marketable
            # (bid >= best_ask for buys, ask <= best_bid for sells)
            # For GTC, we assume aggressive enough to fill at signal price
            side_str = str(signal.side)
            fill_price = signal.price

            # Position limit check
            old_pos = pnl.positions.get(signal.product, 0)
            signed = signal.volume if side_str == "BUY" else -signal.volume
            new_pos = old_pos + signed
            if abs(new_pos) > position_limit:
                continue  # skip, would exceed limit

            pnl.record_fill(
                product=signal.product,
                side=side_str,
                price=fill_price,
                volume=signal.volume,
                strategy=name,
            )

            # Handle QUOTE type (also fill the ask side)
            if str(signal.order_type) == "QUOTE" and signal.ask_price is not None:
                ask_pos = pnl.positions.get(signal.product, 0)
                ask_new = ask_pos - (signal.ask_volume or signal.volume)
                if abs(ask_new) <= position_limit:
                    pnl.record_fill(
                        product=signal.product,
                        side="SELL",
                        price=signal.ask_price,
                        volume=signal.ask_volume or signal.volume,
                        strategy=name,
                    )

            if verbose:
                print(f"  [{name}] {side_str} {signal.volume}x {signal.product} "
                      f"@ {fill_price:.0f} | {signal.reason}")

        # Mark-to-market after each tick
        pnl.mark_to_market(bot.mids, ts=snap.timestamp)

    elapsed = time.time() - start_time

    return {
        "total_pnl": round(pnl.total_pnl, 2),
        "realized_pnl": round(pnl.realized_pnl, 2),
        "unrealized_pnl": round(pnl.unrealized_pnl, 2),
        "peak_pnl": round(pnl.peak_pnl, 2),
        "drawdown": round(pnl.drawdown, 2),
        "trade_count": pnl.trade_count,
        "signals_emitted": signals_emitted,
        "signals_by_strategy": dict(signals_by_strategy),
        "strategy_pnl": {k: round(v, 2) for k, v in pnl.strategy_pnl.items()},
        "strategy_trades": dict(pnl.strategy_trades),
        "final_positions": {k: v for k, v in pnl.positions.items() if v != 0},
        "final_exposure": pnl.exposure,
        "pnl_curve": pnl.pnl_curve,
        "elapsed_sec": round(elapsed, 3),
        "params": params or {},
    }


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

# Define the parameter grid to sweep
PARAM_GRID = {
    # ETF Arb
    "ARB_THRESHOLD":          [15, 25, 30, 50, 80],
    "ARB_AGGRESSIVE_THRESHOLD": [50, 80, 120],

    # Mean Reversion
    "MR_THRESHOLD_SIGMAS":    [1.0, 1.5, 2.0, 2.5],

    # Component Arb
    "COMP_ARB_THRESHOLD":     [0.008, 0.012, 0.015, 0.020, 0.030],

    # Fly
    "FLY_THRESHOLD":          [10, 15, 20, 30, 50],

    # ETF MM spread
    "ETF_MM_WIDTH":           [8, 12, 15, 20, 30],

    # Vol MM
    "_VMM_BASE_HALF_SPREAD":  [10, 15, 20, 30],
    "_VMM_SKEW_PER_UNIT":     [0.3, 0.5, 1.0],
}


def generate_param_configs(
    grid: dict | None = None,
    single_param_sweep: bool = True,
) -> list[dict]:
    """Generate parameter configurations to test.

    If single_param_sweep=True (default), varies one param at a time from
    baseline. This gives N configs where N = sum of all grid values.
    If False, does full cartesian product (can be huge).
    """
    if grid is None:
        grid = PARAM_GRID

    baseline = {}  # empty = use module defaults

    if single_param_sweep:
        configs = [baseline]  # Always include baseline
        for param, values in grid.items():
            for val in values:
                configs.append({param: val})
        return configs
    else:
        # Full cartesian — use with caution
        keys = list(grid.keys())
        value_lists = [grid[k] for k in keys]
        configs = []
        for combo in itertools.product(*value_lists):
            configs.append(dict(zip(keys, combo)))
        return configs


def print_results_table(results: list[dict]) -> None:
    """Print a comparison table of backtest results."""
    # Sort by total PnL descending
    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    print("\n" + "=" * 100)
    print(f"{'#':>3} {'Total PnL':>12} {'Realized':>10} {'Peak':>10} "
          f"{'Drawdown':>10} {'Trades':>7} {'Signals':>8}  Params")
    print("-" * 100)

    for i, r in enumerate(results):
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items()) if r["params"] else "(baseline)"
        print(f"{i+1:>3} {r['total_pnl']:>12.0f} {r['realized_pnl']:>10.0f} "
              f"{r['peak_pnl']:>10.0f} {r['drawdown']:>10.0f} "
              f"{r['trade_count']:>7} {r['signals_emitted']:>8}  {params_str}")

    print("=" * 100)

    # Per-strategy breakdown for best config
    best = results[0]
    print(f"\nBest config: {best['params'] or '(baseline)'}")
    print(f"  Total PnL: {best['total_pnl']:+.0f}  |  Peak: {best['peak_pnl']:.0f}  |  "
          f"Drawdown: {best['drawdown']:.0f}  |  Trades: {best['trade_count']}")
    if best.get("strategy_pnl"):
        print("  Per-strategy PnL:")
        for name, spnl in sorted(best["strategy_pnl"].items(), key=lambda x: -x[1]):
            trades = best["strategy_trades"].get(name, 0)
            print(f"    {name:30s}  PnL: {spnl:>+10.0f}  Trades: {trades}")
    if best.get("final_positions"):
        print(f"  Final positions: {best['final_positions']}")


def _get_all_strategies():
    """Import all strategies from the strategies module."""
    from strategies import (
        grid_strategy,
        etf_arb_strategy,
        component_arb_strategy,
        fly_strategy,
        mean_revert_strategy,
        inventory_unwind_strategy,
        etf_mm_strategy,
        vol_mm_strategy,
    )
    return [
        grid_strategy,
        etf_arb_strategy,
        component_arb_strategy,
        fly_strategy,
        mean_revert_strategy,
        inventory_unwind_strategy,
        etf_mm_strategy,
        vol_mm_strategy,
    ]


def _get_individual_strategies() -> list[tuple[str, list[Callable]]]:
    """Return each strategy individually for isolated testing."""
    from strategies import (
        grid_strategy,
        etf_arb_strategy,
        component_arb_strategy,
        fly_strategy,
        mean_revert_strategy,
        inventory_unwind_strategy,
        etf_mm_strategy,
        vol_mm_strategy,
    )
    return [
        ("grid_strategy", [grid_strategy]),
        ("etf_arb_strategy", [etf_arb_strategy]),
        ("component_arb_strategy", [component_arb_strategy]),
        ("fly_strategy", [fly_strategy]),
        ("mean_revert_strategy", [mean_revert_strategy]),
        ("inventory_unwind_strategy", [inventory_unwind_strategy]),
        ("etf_mm_strategy", [etf_mm_strategy]),
        ("vol_mm_strategy", [vol_mm_strategy]),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Algothon Strategy Backtester")
    parser.add_argument("--data-dir", default=os.path.expandvars(r"%LOCALAPPDATA%\algothon_data"),
                        help="Path to collected data CSVs")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full parameter sweep")
    parser.add_argument("--individual", action="store_true",
                        help="Test each strategy in isolation first")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every signal")
    parser.add_argument("--pos-limit", type=int, default=80,
                        help="Max position per product")
    args = parser.parse_args()

    print("=" * 60)
    print("  ALGOTHON BACKTESTER")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {args.data_dir}")
    snapshots = load_snapshots(args.data_dir)
    if not snapshots:
        print("ERROR: No snapshot data found. Run the bot first to collect data.")
        print(f"  Expected CSVs in: {args.data_dir}")
        sys.exit(1)

    ts_range = snapshots[-1].timestamp - snapshots[0].timestamp
    print(f"  Time range: {ts_range/60:.1f} minutes")
    products = set(s.product for s in snapshots)
    print(f"  Products: {', '.join(sorted(products))}")

    # --- Individual strategy test ---
    if args.individual:
        print("\n--- INDIVIDUAL STRATEGY TEST ---")
        print(f"{'Strategy':>30s} {'PnL':>10} {'Trades':>7} {'Signals':>8} {'Positions':>10}")
        print("-" * 75)
        for name, strat_list in _get_individual_strategies():
            result = run_backtest(
                snapshots, strat_list,
                strategy_names=[name],
                position_limit=args.pos_limit,
                verbose=False,
            )
            pos_str = str(result["final_positions"]) if result["final_positions"] else "-"
            print(f"{name:>30s} {result['total_pnl']:>10.0f} "
                  f"{result['trade_count']:>7} {result['signals_emitted']:>8} "
                  f"{pos_str:>10}")
        print()

    # --- Baseline run with all strategies ---
    print("\n--- BASELINE (all strategies, default params) ---")
    all_strats = _get_all_strategies()
    baseline_result = run_backtest(
        snapshots, all_strats,
        position_limit=args.pos_limit,
        verbose=args.verbose,
    )
    print(f"  Total PnL: {baseline_result['total_pnl']:+.0f}")
    print(f"  Realized:  {baseline_result['realized_pnl']:+.0f}")
    print(f"  Peak:      {baseline_result['peak_pnl']:.0f}")
    print(f"  Drawdown:  {baseline_result['drawdown']:.0f}")
    print(f"  Trades:    {baseline_result['trade_count']}")
    print(f"  Time:      {baseline_result['elapsed_sec']:.2f}s")
    if baseline_result["strategy_pnl"]:
        print("  Per-strategy:")
        for name, spnl in sorted(baseline_result["strategy_pnl"].items(), key=lambda x: -x[1]):
            trades = baseline_result["strategy_trades"].get(name, 0)
            signals = baseline_result["signals_by_strategy"].get(name, 0)
            print(f"    {name:30s}  PnL: {spnl:>+8.0f}  Trades: {trades:>4}  Signals: {signals:>4}")

    # --- Parameter sweep ---
    if args.sweep:
        print("\n--- PARAMETER SWEEP ---")
        configs = generate_param_configs()
        print(f"  Running {len(configs)} configurations...")
        results = []
        for i, params in enumerate(configs):
            label = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "(baseline)"
            sys.stdout.write(f"\r  [{i+1}/{len(configs)}] {label:60s}")
            sys.stdout.flush()
            result = run_backtest(
                snapshots, all_strats,
                params=params,
                position_limit=args.pos_limit,
            )
            results.append(result)
        sys.stdout.write("\r" + " " * 80 + "\r")
        print_results_table(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
