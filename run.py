"""Algothon 2026 — Main entry point.

Run this script to start the bot with all strategies.
    python run.py                    # LIVE trading with risk management
    python run.py --collect-only     # data collection mode (no orders)

DATA COLLECTION IS ALWAYS ON:
    Orderbook snapshots & trades are recorded regardless of mode.
    CSVs: collected_data/{PRODUCT}.csv and collected_data/trades.csv
    APPEND-only, no overwrites, no duplicate timestamps.

FLOW:
    Teammate strategy(bot, snap) → Signal | None
    → RiskManager.check(signal)  → approved / rejected
    → Executor.execute(signal)   → order sent to exchange
"""

import sys
import time
import os
import shutil
from algothon_bot import AlgothonBot
from signals import RiskManager, RiskConfig, Executor, make_signal_dispatcher
from data_cache import DataPersistence
from dashboard import Dashboard

# Import all strategies
from strategies import (
    grid_strategy,
    etf_arb_strategy,
    component_arb_strategy,
    fly_strategy,
    mean_revert_strategy,
    inventory_unwind_strategy,
    etf_mm_strategy,
    vol_mm_strategy,
    trade_logger,
    alpha_directional,
    set_alpha,
)
from alpha import AlphaEngine

# ── CREDENTIALS ──────────────────────────────────────────────────────────
TEST_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
CHALLENGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/"

EXCHANGE_URL = CHALLENGE_URL  # ← LIVE CHALLENGE EXCHANGE
USERNAME = "IMCooked"             # <── PUT YOUR USERNAME HERE
PASSWORD = "imsocooked"             # <── PUT YOUR PASSWORD HERE

# ── RISK CONFIG (MODERATE — tuned for competition) ──────────────────────
risk_config = RiskConfig(
    max_position_per_product=80,   # exchange limit is ±100; leave headroom
    max_total_exposure=350,        # sum of abs positions across all products
    max_order_size=10,             # moderate single order cap
    max_orders_per_minute=50,
    max_loss_threshold=-50000.0,   # wide PnL floor (we're already +1.5M)
    max_drawdown=-20000.0,         # drawdown halt
    cooldown_after_loss=30.0,
    inventory_skew_threshold=30,
    inventory_hard_limit=60,
    max_notional_per_order=200_000.0,
)

# ── MODE ─────────────────────────────────────────────────────────────────
COLLECT_ONLY = "--collect-only" in sys.argv
DRY_RUN      = "--dry-run"      in sys.argv
CLEAR_DATA   = "--clear-data"   in sys.argv

# Dashboard port (default 8080; use --port=8081 for a second instance)
DASH_PORT = 8080
for arg in sys.argv:
    if arg.startswith("--port="):
        DASH_PORT = int(arg.split("=", 1)[1])

# ── CLEAR OLD DATA ───────────────────────────────────────────────────────
# Store data OUTSIDE OneDrive to avoid sync locking CSVs mid-write
data_dir = os.path.expandvars(r"%LOCALAPPDATA%\algothon_data")
if CLEAR_DATA:
    if os.path.exists(data_dir):
        failed = []
        for f in os.listdir(data_dir):
            fp = os.path.join(data_dir, f)
            try:
                os.remove(fp)
            except PermissionError:
                # File locked (OneDrive / another process) – truncate instead
                try:
                    open(fp, "w").close()
                    failed.append(f"{f} (truncated)")
                except Exception:
                    failed.append(f"{f} (LOCKED)")
            except Exception:
                pass
        if failed:
            print(f"[CLEAR] Some files locked: {', '.join(failed)}")
        else:
            shutil.rmtree(data_dir, ignore_errors=True)
            print(f"[CLEAR] Deleted {data_dir}/")
    else:
        print("[CLEAR] No data directory to clear")

# ── BOOT ─────────────────────────────────────────────────────────────────
bot = AlgothonBot(EXCHANGE_URL, USERNAME, PASSWORD)
risk = RiskManager(risk_config)
executor = Executor(bot, risk, dry_run=DRY_RUN)
dispatch = make_signal_dispatcher(executor, risk)

# Data persistence (ALWAYS active)
persist = DataPersistence(data_dir)

# ── ALPHA ENGINE ─────────────────────────────────────────────────────────
# Computes settlement fair values from real-world data (weather, tides).
# This is the #1 competitive edge: products settle based on observable data.
alpha = AlphaEngine()
set_alpha(alpha)
try:
    alpha.refresh()  # initial fetch
    print(alpha.summary())
except Exception as e:
    print(f"[ALPHA] Initial fetch failed (will retry): {e}")

if COLLECT_ONLY:
    print("=" * 60)
    print("  DATA COLLECTION MODE — no orders will be sent")
    print("  Snapshots + trades will be saved to collected_data/")
    print("=" * 60)
else:
    # Register ALL strategies with risk + execution dispatch
    # (works for both --dry-run and live — Executor handles the difference)
    bot.register_orderbook_strategy(dispatch(alpha_directional))   # #1 PRIORITY: data-driven
    bot.register_orderbook_strategy(dispatch(grid_strategy))
    bot.register_orderbook_strategy(dispatch(etf_arb_strategy))
    bot.register_orderbook_strategy(dispatch(component_arb_strategy))
    bot.register_orderbook_strategy(dispatch(fly_strategy))
    # bot.register_orderbook_strategy(dispatch(mean_revert_strategy))   # disabled: replaced by alpha
    bot.register_orderbook_strategy(dispatch(inventory_unwind_strategy))
    # bot.register_orderbook_strategy(dispatch(etf_mm_strategy))        # disabled: net loser, overlaps vol_mm
    bot.register_orderbook_strategy(dispatch(vol_mm_strategy))
    bot.register_trade_strategy(trade_logger)
    print(f"[RISK] pos={risk_config.max_position_per_product}, "
          f"exposure={risk_config.max_total_exposure}, "
          f"order_size={risk_config.max_order_size}, "
          f"loss_halt={risk_config.max_loss_threshold}, "
          f"drawdown_halt={risk_config.max_drawdown}")

    if DRY_RUN:
        print("=" * 60)
        print("  DRY-RUN MODE — strategies run, orders LOGGED not sent")
        print("  Watch [DRY-RUN] lines to verify strategy behaviour")
        print("=" * 60)

bot.start()

# ── DASHBOARD ────────────────────────────────────────────────────────────
dash = Dashboard(bot, risk, executor, persist, port=DASH_PORT,
                 simulator=executor.simulator, alpha=alpha)
dash.start()  # opens http://localhost:8080

# ── MAIN LOOP ────────────────────────────────────────────────────────────
print("Bot running. Data collection is ALWAYS ON. Press Ctrl+C to stop.\n")
SAVE_INTERVAL = 120   # save CSVs every 2 minutes
PNL_CHECK = 30        # check PnL every 30 seconds
STALE_ORDER_AGE = 20  # cancel resting orders older than 20s (was 60 — too slow)
PRINT_INTERVAL = 10   # print state every 10 seconds
ALPHA_INTERVAL = 60   # refresh alpha fair values every 60s

last_save = time.time()
last_pnl_check = time.time()
last_stale_check = time.time()
last_alpha_refresh = time.time()

try:
    while True:
        time.sleep(PRINT_INTERVAL)
        bot.print_state()
        now = time.time()

        # ── Sync positions from exchange (always, for risk tracking) ──
        try:
            positions = bot.get_positions()
            if isinstance(positions, dict):
                risk.update_positions(positions)
                # Push positions into strategy module for position guards
                from strategies import update_positions as strat_update_pos
                strat_update_pos(positions)
        except Exception:
            pass

        # ── Refresh alpha fair values ──
        if now - last_alpha_refresh > ALPHA_INTERVAL:
            try:
                alpha.refresh(bot.mids)
            except Exception as e:
                print(f"[ALPHA] Refresh error: {e}")
            last_alpha_refresh = now

        # ── PnL watchdog ──
        if not COLLECT_ONLY and now - last_pnl_check > PNL_CHECK:
            risk.check_pnl(bot)
            last_pnl_check = now

        # ── Cancel stale orders ──
        if not COLLECT_ONLY and now - last_stale_check > STALE_ORDER_AGE:
            n = executor.cancel_stale_orders(max_age=STALE_ORDER_AGE)
            if n > 0:
                print(f"[EXEC] Cancelled {n} stale orders")
            last_stale_check = now

        # ── Risk stats ──
        if not COLLECT_ONLY:
            risk.print_stats()

        # ── Simulator mark-to-market + summary (all modes) ──
        if executor.simulator:
            executor.simulator.mark_to_market(bot.cache.mids)
            executor.simulator.print_summary()

        # ── Auto-save data (ALWAYS — both modes) ──
        if now - last_save > SAVE_INTERVAL:
            persist.save_all(bot.cache)
            last_save = now

except KeyboardInterrupt:
    print("\nShutting down...")

    # Final save
    snap_rows, trade_rows = persist.save_all(bot.cache)
    print(f"[PERSIST] Final save: {snap_rows} snapshot rows + {trade_rows} trade rows")

    if not COLLECT_ONLY:
        risk.print_stats()
        bot.cancel_all_orders()
    bot.stop()
    print("Done.")
