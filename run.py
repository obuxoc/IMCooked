"""Algothon 2026 — Main entry point.

Run this script to start the bot with all strategies.
    python run.py                    # LIVE trading with risk management
    python run.py --collect-only     # data collection mode (no orders)

FLOW:
    Teammate strategy(bot, snap) → Signal | None
    → RiskManager.check(signal)  → approved / rejected
    → Executor.execute(signal)   → order sent to exchange
"""

import sys
import time
import os
from algothon_bot import AlgothonBot
from signals import RiskManager, RiskConfig, Executor, make_signal_dispatcher

# Import teammate strategies (they return Signal | None, never trade directly)
from strategies import mm_strategy, arb_strategy, trade_logger

# ── CREDENTIALS ──────────────────────────────────────────────────────────
TEST_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
CHALLENGE_URL = "REPLACE_WITH_CHALLENGE_URL"

EXCHANGE_URL = TEST_URL       # Switch to CHALLENGE_URL when competing
USERNAME = "abcd"       # <── PUT YOUR USERNAME HERE
PASSWORD = "abcd"   # <── PUT YOUR PASSWORD HERE

# ── RISK CONFIG (TUNE THESE — this is YOUR domain) ──────────────────────
risk_config = RiskConfig(
    max_position_per_product=50,   # max abs net position per product
    max_total_exposure=200,        # sum of abs positions across all products
    max_order_size=20,             # single order volume cap
    max_orders_per_minute=50,      # stay under exchange 60/min limit
    max_loss_threshold=-5000.0,    # emergency halt if PnL drops below this
    cooldown_after_loss=30.0,      # seconds to pause after halt trigger
)

# ── MODE ─────────────────────────────────────────────────────────────────
COLLECT_ONLY = "--collect-only" in sys.argv

# ── BOOT ─────────────────────────────────────────────────────────────────
bot = AlgothonBot(EXCHANGE_URL, USERNAME, PASSWORD)
risk = RiskManager(risk_config)
executor = Executor(bot, risk)
dispatch = make_signal_dispatcher(executor, risk)

if COLLECT_ONLY:
    print("=" * 60)
    print("  DATA COLLECTION MODE — no orders will be sent")
    print("=" * 60)
else:
    # Wrap each teammate strategy with risk + execution
    bot.register_orderbook_strategy(dispatch(mm_strategy))
    bot.register_orderbook_strategy(dispatch(arb_strategy))
    bot.register_trade_strategy(trade_logger)
    print(f"[RISK] Limits: pos={risk_config.max_position_per_product}, "
          f"exposure={risk_config.max_total_exposure}, "
          f"order_size={risk_config.max_order_size}, "
          f"loss_halt={risk_config.max_loss_threshold}")

bot.start()

# ── MAIN LOOP ────────────────────────────────────────────────────────────
print("Bot running. Press Ctrl+C to stop.\n")
SAVE_INTERVAL = 300   # save CSVs every 5 minutes
PNL_CHECK = 30        # check PnL every 30 seconds
STALE_ORDER_AGE = 60  # cancel resting orders older than 60s
last_save = time.time()
last_pnl_check = time.time()
last_stale_check = time.time()

try:
    while True:
        time.sleep(10)
        bot.print_state()
        now = time.time()

        # ── Sync positions from exchange ──
        if not COLLECT_ONLY:
            try:
                risk.update_positions(bot.get_positions())
            except Exception:
                pass

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

        # ── Auto-save data to CSV ──
        if now - last_save > SAVE_INTERVAL:
            save_dir = os.path.join(os.path.dirname(__file__), "collected_data")
            os.makedirs(save_dir, exist_ok=True)
            for product in bot.cache.products:
                df = bot.history(product)
                if not df.empty:
                    df.to_csv(os.path.join(save_dir, f"{product}.csv"), index=False)
            trades = bot.trade_history()
            if not trades.empty:
                trades.to_csv(os.path.join(save_dir, "trades.csv"), index=False)
            print(f"[SAVE] Data saved to {save_dir}/")
            last_save = now

except KeyboardInterrupt:
    print("\nShutting down...")
    # Final save
    save_dir = os.path.join(os.path.dirname(__file__), "collected_data")
    os.makedirs(save_dir, exist_ok=True)
    for product in bot.cache.products:
        df = bot.history(product)
        if not df.empty:
            df.to_csv(os.path.join(save_dir, f"{product}.csv"), index=False)
    trades = bot.trade_history()
    if not trades.empty:
        trades.to_csv(os.path.join(save_dir, "trades.csv"), index=False)
    print(f"[SAVE] Final data saved to {save_dir}/")

    if not COLLECT_ONLY:
        risk.print_stats()
        bot.cancel_all_orders()
    bot.stop()
    print("Done.")
